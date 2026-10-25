"""Train a shared PyTorch encoder for synthetic document and entity recognition.

The model is multi-task: one encoder, one document-type head and one token-level
entity head. Training examples are sliding windows over the documents, so a long
transcript is supervised on every row instead of only its first 160 tokens.
Evaluation is done at document level: window predictions are mapped back onto the
parent document and compared with the literal gold spans.

Metrics apply only to procedural text documents and held-out synthetic templates. They
are not claims about real-world OCR, arbitrary academic documents, or other institutions.
"""
from __future__ import annotations
import argparse, json, math, random, time
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer
from .model import BASE_MODEL, MAX_LENGTH
from .synthetic_academic import DOCUMENT_TYPES, apply_ocr_noise, generate_dataset, split_by_template

ENTITY_TYPES=["COURSE","COURSE_CODE","TEACHER","ROOM","CREDIT","GRADE_CATEGORY","WEIGHT","ASSESSMENT","DATE","TIME","PREREQUISITE","STUDENT","SCHOOL","GRADE_LEVEL","MARK","SCORE","PERFORMANCE_LEVEL","SUBJECT_AREA","GPA"]
ENTITY_LABELS=["O"]+[f"{p}-{t}" for t in ENTITY_TYPES for p in ("B","I")]
ENTITY_TO_ID={v:i for i,v in enumerate(ENTITY_LABELS)}
WINDOW_STRIDE=32
# These entity values are always a single token ("09", "337"), so the generator can
# never emit an I- continuation for them. They are masked out of multi-token span
# typing instead of leaving an untrained tag to compete for probability mass.
SINGLE_TOKEN_ENTITY_TYPES=frozenset({"GRADE_LEVEL","SCORE"})


def choice_confidence(probabilities):
    """Return Jev's Choice sharpness statistic for a normalized option distribution."""
    values=[float(value) for value in probabilities]
    if len(values)<2:raise ValueError('Choice confidence requires at least two options')
    if any(not math.isfinite(value) or value<0 or value>1 for value in values):
        raise ValueError('Choice probabilities must be finite and in [0, 1]')
    if not math.isclose(sum(values),1.0,rel_tol=0.0,abs_tol=1e-6):
        raise ValueError('Choice probabilities must sum to 1')
    n=len(values)
    return max(0.0,min(1.0,(max(values)-1.0/n)/(1.0-1.0/n)))


def entity_type_distribution(token_probabilities,token_ids):
    """Normalize BIO class likelihoods into a type Choice for a fixed proposed span boundary."""
    token_ids=list(token_ids)
    if not token_ids:raise ValueError('An entity-type distribution requires at least one span token')
    scores={}
    for entity_type in ENTITY_TYPES:
        if len(token_ids)>1 and entity_type in SINGLE_TOKEN_ENTITY_TYPES:
            # A single-token value cannot be a multi-token span; exclude it instead of
            # letting an untrained I- tag win the comparison.
            continue
        tag_ids=[ENTITY_TO_ID[('B-' if position==0 else 'I-')+entity_type] for position in range(len(token_ids))]
        scores[entity_type]=sum(math.log(max(float(token_probabilities[token_id,tag_id]),1e-12)) for token_id,tag_id in zip(token_ids,tag_ids))/len(token_ids)
    if not scores:raise ValueError('Entity-type evidence must have positive finite mass')
    peak=max(scores.values())
    weights={entity_type:math.exp(score-peak) for entity_type,score in scores.items()}
    total=sum(weights.values())
    if not math.isfinite(total) or total<=0:raise ValueError('Entity-type evidence must have positive finite mass')
    return {entity_type:score/total for entity_type,score in weights.items()}


def _document_entities(text: str, entities: list[dict[str,Any]], tokenizer: Any, max_length: int):
    enc=tokenizer(text,truncation=True,max_length=max_length,return_offsets_mapping=True,add_special_tokens=True)
    tags=["O"]*len(enc["input_ids"])
    for entity in entities:
        start=int(entity["start"]);end=int(entity["end"])
        if start<0 or end>len(text) or start>=end:continue
        # Drop spans beyond the actual tokenizer truncation; never silently shift them.
        ids=[i for i,(a,b) in enumerate(enc["offset_mapping"]) if b>a and a<end and b>start]
        if not ids:continue
        for i in ids:
            # Annotations may overlap for a prerequisite course code; the relation-role
            # tag takes precedence over the generic course-code tag for exact-span task.
            if tags[i] != "O" and entity['type'] == "COURSE_CODE":
                continue
            tags[i]=f"I-{entity['type']}"
        tags[ids[0]]=f"B-{entity['type']}"
    return enc,[ENTITY_TO_ID[t] for t in tags]


def build_window_rows(rows,tokenizer,max_length,stride=WINDOW_STRIDE):
    """Turn documents into overlapping training windows with re-based entity spans."""
    windows=[]
    for row in rows:
        text=row['ocr_text']
        clean=row.get('text',text)
        encoded=tokenizer(text,truncation=True,max_length=max_length,stride=stride,
            return_overflowing_tokens=True,return_offsets_mapping=True)
        for index,offsets in enumerate(encoded['offset_mapping']):
            visible=[(int(a),int(b)) for a,b in offsets if int(b)>int(a)]
            if not visible:continue
            start,end=visible[0][0],visible[-1][1]
            entities=[]
            for entity in row['entities']:
                if entity['start']>=start and entity['end']<=end:
                    entities.append({**entity,'start':entity['start']-start,'end':entity['end']-start})
            windows.append({
                'window_id':f"{row['document_id']}#{index}",
                'parent_id':row['document_id'],
                'document_type':row['document_type'],
                'template_family':row['template_family'],
                'world_school_id':row.get('world_school_id'),
                'ocr_text':text[start:end],
                'text':clean[start:end],
                'entities':entities,
                'doc_start':start,'doc_end':end,
                'doc_length':len(text),
                # The characters just outside the window tell whether a value at the
                # window edge was cut in half or simply ends there.
                'prev_char':text[start-1] if start>0 else '',
                'next_char':text[end] if end<len(text) else '',
            })
    return windows


class Rows(Dataset):
    def __init__(self,rows,tokenizer,max_length):self.rows=rows;self.tokenizer=tokenizer;self.max_length=max_length
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];enc,tags=_document_entities(row['ocr_text'],row['entities'],self.tokenizer,self.max_length)
        return {'input_ids':torch.tensor(enc['input_ids'],dtype=torch.long),'attention_mask':torch.tensor(enc['attention_mask'],dtype=torch.long),
                'entity_labels':torch.tensor(tags,dtype=torch.long),'doc_label':DOCUMENT_TYPES.index(row['document_type']),
                'row':row,'offsets':enc['offset_mapping']}


def collate(batch,pad_id):
    length=max(len(x['input_ids']) for x in batch)
    def pad(x,v):return torch.cat([x,torch.full((length-len(x),),v,dtype=x.dtype)]) if len(x)<length else x
    return {'input_ids':torch.stack([pad(x['input_ids'],pad_id) for x in batch]),'attention_mask':torch.stack([pad(x['attention_mask'],0) for x in batch]),
            'entity_labels':torch.stack([pad(x['entity_labels'],-100) for x in batch]),'doc_labels':torch.tensor([x['doc_label'] for x in batch]),
            'offsets':[x['offsets'] for x in batch],'rows':[x['row'] for x in batch]}


class PaceMultiTask(nn.Module):
    def __init__(self,base_model,config_dir=None,pretrained=True):
        super().__init__();from transformers import AutoModel
        if config_dir:
            config=AutoConfig.from_pretrained(config_dir,local_files_only=True)
            # A shallower encoder keeps the pretrained weights of the layers it
            # keeps; the dropped layers are simply not part of the model.
            self.encoder=AutoModel.from_pretrained(base_model,config=config) if pretrained else AutoModel.from_config(config)
        else:
            self.encoder=AutoModel.from_pretrained(base_model)
        h=int(self.encoder.config.hidden_size);self.dropout=nn.Dropout(.1);self.doc_head=nn.Linear(h,len(DOCUMENT_TYPES));self.entity_head=nn.Linear(h,len(ENTITY_LABELS))
    def forward(self,input_ids,attention_mask):
        x=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        pooled=(x*attention_mask.unsqueeze(-1)).sum(1)/attention_mask.sum(1,keepdim=True).clamp_min(1)
        return self.doc_head(self.dropout(pooled)),self.entity_head(self.dropout(x))


def spans_from_iob(tags,offsets):
    """Decode BIO tags into spans, recovering an I- tag that opens a new entity.

    Single-token entity values (grade level, score) have no I- continuation in the
    training data, so a confident I- tag on them would otherwise drop the entity.
    """
    spans=[];current=None
    for i,tag_id in enumerate(tags[:len(offsets)]):
        a,b=offsets[i]
        if int(b)<=int(a):
            if current:spans.append(current);current=None
            continue
        tag=ENTITY_LABELS[int(tag_id)] if int(tag_id)>=0 else "O"
        if tag.startswith('B-'):
            if current:spans.append(current)
            current={'type':tag[2:],'start':int(offsets[i][0]),'end':int(offsets[i][1])}
        elif tag.startswith('I-'):
            if current and current['type']==tag[2:]:
                current['end']=int(offsets[i][1])
            else:
                # An I- tag with no open span of that type starts a new entity.
                if current:spans.append(current);current=None
                current={'type':tag[2:],'start':int(offsets[i][0]),'end':int(offsets[i][1])}
        else:
            if current:spans.append(current);current=None
    if current:spans.append(current)
    return spans


def _run_model(model, rows, tokenizer, max_length, device, batch_size=64):
    loader=DataLoader(Rows(rows,tokenizer,max_length),batch_size=batch_size,shuffle=False,collate_fn=lambda b:collate(b,tokenizer.pad_token_id or 0))
    doc_logits=[];entity_preds=[];offsets=[];model.eval()
    with torch.inference_mode():
        for batch in loader:
            dx,ex=model(batch['input_ids'].to(device),batch['attention_mask'].to(device))
            doc_logits.extend(dx.cpu())
            batch_offsets=batch['offsets']
            # The model sees a padded batch, so its tag sequence is longer than the
            # tokenizer offsets of a short window. Keep the tags aligned to the
            # offsets of their own window.
            for row_index,sequence in enumerate(ex.argmax(-1).cpu().tolist()):
                entity_preds.append(sequence[:len(batch_offsets[row_index])])
            offsets.extend(batch_offsets)
    return doc_logits,entity_preds,offsets


def is_word_cut(context: str, span_start: int, span_end: int, window_start: int, window_end: int) -> bool:
    """True when a window boundary cut the value's word in half.

    A value that merely ends at a window edge is complete: its word stops there.
    Only a value whose word continues past the edge was read as a fragment, and a
    fragment was never seen whole by any window.
    """
    before = context[span_start - 1] if span_start == window_start and span_start > 0 else ""
    after = context[span_end] if span_end == window_end and span_end < len(context) else ""
    return bool((before and before.isalnum()) or (after and after.isalnum()))


def _document_level(window_rows,doc_logits,entity_preds,offsets):
    """Map window predictions back onto the parent document and aggregate the logits.

    A span whose word was cut in half by a window boundary was never read whole, so
    it is not counted as a prediction. This is the same rule the inference engine
    applies, so the measured numbers describe the behaviour a caller actually gets.
    """
    logits={};spans={};support={}
    for row,logit,tags,window_offsets in zip(window_rows,doc_logits,entity_preds,offsets):
        parent=row['parent_id']
        logits.setdefault(parent,[]).append(logit)
        bucket=spans.setdefault(parent,set())
        visible=[(int(a),int(b)) for a,b in window_offsets if int(b)>int(a)]
        if not visible:continue
        window_start,window_end=visible[0][0],visible[-1][1]
        window_text=row.get('ocr_text','')
        # The characters just outside the window are needed to tell a cut word from
        # a value that simply ends at the edge.
        context=row.get('prev_char','')+window_text+row.get('next_char','')
        for span in spans_from_iob(tags,window_offsets):
            start=span['start']+row['doc_start'];end=span['end']+row['doc_start']
            if start<end and end<=row['doc_end']:
                key=(parent,span['type'],start,end)
                fragment=is_word_cut(context,span['start']+1,span['end']+1,window_start+1,window_end+1)
                bucket.add((span['type'],start,end))
                seen=support.setdefault(key,{'complete':False,'truncated':False})
                if fragment:seen['truncated']=True
                else:seen['complete']=True
    for key,seen in support.items():
        if not seen['complete']:
            parent,entity_type,start,end=key
            spans.get(parent,set()).discard((entity_type,start,end))
    return logits,spans


def exact_entity_metrics(doc_rows,per_doc_spans):
    """Exact span precision/recall/f1 over whole documents."""
    gold_set=set();pred_set=set()
    for row in doc_rows:
        document_id=row['document_id']
        for entity in row['entities']:
            gold_set.add((document_id,entity['type'],int(entity['start']),int(entity['end'])))
        for entity_type,start,end in per_doc_spans.get(document_id,set()):
            pred_set.add((document_id,entity_type,start,end))
    tp=len(gold_set & pred_set);fp=len(pred_set-gold_set);fn=len(gold_set-pred_set)
    per={}
    for typ in ENTITY_TYPES:
        g={s for s in gold_set if s[1]==typ};p={s for s in pred_set if s[1]==typ};t=len(g&p)
        # A type with no gold span and no prediction has no score to report: a zero
        # would read as a total failure of a label the document never contains.
        empty=not g and not p
        per[typ]={'support':len(g),'predicted':len(p),'true_positive':t,
            'precision':None if empty else t/max(len(p),1),
            'recall':None if empty else t/max(len(g),1),
            'f1':None if empty else 2*t/max(len(g)+len(p),1)}
    return {'true_spans':len(gold_set),'predicted_spans':len(pred_set),'true_positive_spans':tp,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),'false_positives':fp,'false_negatives':fn,'per_entity_type':per}


def _document_type_metrics(doc_rows,per_doc_logits):
    truth=[row['document_type'] for row in doc_rows]
    preds=[]
    for row in doc_rows:
        logits=per_doc_logits.get(row['document_id'])
        if not logits:
            preds.append('unknown');continue
        mean=torch.stack(logits).mean(0)
        preds.append(DOCUMENT_TYPES[int(mean.argmax(-1))])
    labels=list(DOCUMENT_TYPES)
    return {'accuracy':float(accuracy_score(truth,preds)),'macro_f1':float(f1_score(truth,preds,average='macro',zero_division=0)),
        'classification_report':classification_report(truth,preds,labels=labels,output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(truth,preds,labels=labels).tolist(),
        'errors':[{'true':row['document_type'],'predicted':pred,'template_family':row['template_family']} for row,pred in zip(doc_rows,preds) if row['document_type']!=pred]}


def evaluate_checkpoint(checkpoint_dir,rows=None,batch_size=64):
    p=Path(checkpoint_dir);meta=torch.load(p/'model.pt',map_location='cpu',weights_only=False)
    tokenizer=AutoTokenizer.from_pretrained(p/meta['tokenizer_dir'],local_files_only=True)
    if rows is None:rows=[json.loads(l) for l in (p/'test.jsonl').read_text(encoding='utf-8').splitlines() if l]
    model=PaceMultiTask(meta['base_model'],str(p/meta['encoder_config']));model.load_state_dict(meta['state_dict']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    windows=build_window_rows(rows,tokenizer,int(meta['max_length']))
    doc_logits,entity_preds,offsets=_run_model(model,windows,tokenizer,int(meta['max_length']),device,batch_size)
    per_doc_logits,per_doc_spans=_document_level(windows,doc_logits,entity_preds,offsets)
    return {'document_predictions':_document_type_metrics(rows,per_doc_logits),'entity_predictions':exact_entity_metrics(rows,per_doc_spans),'device':str(device)}


def _parser_transcript_signal(text: str) -> dict[str, Any] | None:
    """Parse the text with the deterministic transcript parser, if it applies."""
    from .transcript_extract import parse_transcript

    parsed = parse_transcript(text)
    checks = parsed["checks"]
    if not (checks["course_record_count"] or checks["assessment_record_count"]):
        return None
    return parsed


def attach_transcript_ir(result: dict[str, Any], text: str) -> dict[str, Any]:
    """Attach the deterministic transcript IR when the document is a transcript.

    The neural head only proposes document type and entity spans; the record-level
    transcript IR comes from the deterministic parser, so every field carries exact
    source offsets. The trigger is reported so a reader can see which signal fired.
    """
    parsed = _parser_transcript_signal(text)
    model_label = result.get("document_type", {}).get("label")
    model_says = model_label == "transcript"
    if parsed is None and not model_says:
        return result
    from .transcript_extract import summarize

    trigger = "both" if (parsed is not None and model_says) else (
        "model_document_type" if model_says else "deterministic_parser_signal"
    )
    result["transcript"] = {
        "parser": "deterministic_rule_based_no_model",
        "triggered_by": trigger,
        "summary": summarize(parsed) if parsed is not None else None,
        "parsed": parsed,
    }
    if model_says and parsed is None:
        result["transcript"]["summary"] = None
        result["transcript"]["parse_failed"] = "no transcript records found in text"
    return result


def _quantize(model):
    """Dynamic int8 quantization of the transformer heads and encoder linears."""
    return torch.ao.quantization.quantize_dynamic(model,{nn.Linear},dtype=torch.qint8)


def train_eval(output='artifacts/synthetic-multitask',count=30000,seed=20261019,epochs=8,batch_size=64,max_length=160,base_model=BASE_MODEL,corruption_rate=.02,encoder_layers=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    docs=generate_dataset(count,seed,corruption_rate);splits=split_by_template(docs,seed);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for name,rows in splits.items():(out/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=100000,strip_accents='unicode');x=vec.fit_transform([r['ocr_text'] for r in splits['train']])
    clf=LogisticRegression(max_iter=1200,class_weight='balanced',random_state=seed).fit(x,[r['document_type'] for r in splits['train']]);sparse_preds=clf.predict(vec.transform([r['ocr_text'] for r in splits['test']])).tolist()
    tok=AutoTokenizer.from_pretrained(base_model);tok.save_pretrained(out/'tokenizer')
    encoder_config=AutoConfig.from_pretrained(base_model)
    if encoder_layers:
        encoder_config.num_hidden_layers=int(encoder_layers)
    encoder_config.save_pretrained(out/'encoder_config')
    window_splits={name:build_window_rows(rows,tok,max_length) for name,rows in splits.items()}
    datasets={k:Rows(v,tok,max_length) for k,v in window_splits.items()};collate_fn=lambda b:collate(b,tok.pad_token_id or 0)
    loaders={k:DataLoader(v,batch_size=batch_size,shuffle=k=='train',collate_fn=collate_fn,num_workers=0) for k,v in datasets.items()}
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=PaceMultiTask(base_model,config_dir=str(out/'encoder_config')).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    cnt=Counter(tag for d in datasets['train'] for tag in d['entity_labels'].tolist() if tag>=0)
    # Coverage is required per entity type (a B- tag proves the type occurs). An I-
    # continuation is only reachable for multi-token values, so a missing I- tag is a
    # property of the value, not a generator bug.
    missing=[t for t in ENTITY_TYPES if cnt.get(ENTITY_TO_ID[f'B-{t}'],0)==0]
    if missing:raise ValueError(f"Training split has no supervised tokens for entity types: {missing}; check generator and increase --count")
    single_token_types=[t for t in ENTITY_TYPES if cnt.get(ENTITY_TO_ID[f'I-{t}'],0)==0]
    weights=np.asarray([1/np.sqrt(max(cnt.get(i,0),1)) for i in range(len(ENTITY_LABELS))],dtype=np.float32)
    weights[0]=.08;weights[1:]/=max(float(weights[1:].mean()),1e-8)
    weights=torch.tensor(np.clip(weights,.05,10),device=dev)
    entity_weight=2.5
    entity_label_counts={ENTITY_LABELS[i]:int(cnt.get(i,0)) for i in range(len(ENTITY_LABELS))}
    best_entity_f1=-1.;state=None;history=[]
    for ep in range(1,epochs+1):
        model.train();t=time.perf_counter();losses=[]
        for b in loaders['train']:
            opt.zero_grad(set_to_none=True)
            dx,ex=model(b['input_ids'].to(dev),b['attention_mask'].to(dev))
            ld=nn.functional.cross_entropy(dx,b['doc_labels'].to(dev))
            le=nn.functional.cross_entropy(ex.reshape(-1,len(ENTITY_LABELS)),b['entity_labels'].to(dev).reshape(-1),weight=weights,ignore_index=-100)
            loss=ld+entity_weight*le;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        model.eval()
        val_logits,val_preds,val_offsets=_run_model(model,window_splits['validation'],tok,max_length,dev,batch_size)
        per_doc_logits,per_doc_spans=_document_level(window_splits['validation'],val_logits,val_preds,val_offsets)
        val_entity_metrics=exact_entity_metrics(splits['validation'],per_doc_spans)
        val_doc_metrics=_document_type_metrics(splits['validation'],per_doc_logits)
        rec={'epoch':ep,'mean_loss':float(np.mean(losses)),'validation_document_accuracy':val_doc_metrics['accuracy'],
             'validation_document_macro_f1':val_doc_metrics['macro_f1'],'validation_entity_span_f1':val_entity_metrics['f1'],
             'seconds':time.perf_counter()-t};history.append(rec);print(json.dumps(rec),flush=True)
        if val_entity_metrics['f1']>best_entity_f1:
            best_entity_f1=val_entity_metrics['f1'];state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if state is None:raise RuntimeError('No checkpoint selected')
    model.load_state_dict(state);model.eval()
    ckpt={'state_dict':state,'base_model':base_model,'encoder_config':'encoder_config','tokenizer_dir':'tokenizer','doc_labels':list(DOCUMENT_TYPES),'entity_labels':ENTITY_LABELS,'max_length':max_length,'seed':seed,'encoder_layers':int(encoder_config.num_hidden_layers),'template_split':{k:sorted({r['template_family'] for r in v}) for k,v in splits.items()}}
    torch.save(ckpt,out/'model.pt')
    truth=[r['document_type'] for r in splits['test']]
    test_logits,test_preds,test_offsets=_run_model(model,window_splits['test'],tok,max_length,dev,batch_size)
    per_doc_logits,per_doc_spans=_document_level(window_splits['test'],test_logits,test_preds,test_offsets)
    neural_report=_document_type_metrics(splits['test'],per_doc_logits)
    sparse_report={'accuracy':float(accuracy_score(truth,sparse_preds)),'macro_f1':float(f1_score(truth,sparse_preds,average='macro',zero_division=0))}
    clean_windows=build_window_rows([{**r,'ocr_text':r['text']} for r in splits['test']],tok,max_length)
    clean_logits,clean_preds,clean_offsets=_run_model(model,clean_windows,tok,max_length,dev,batch_size)
    clean_doc_logits,clean_doc_spans=_document_level(clean_windows,clean_logits,clean_preds,clean_offsets)
    entity_corrupted=exact_entity_metrics(splits['test'],per_doc_spans)
    entity_clean=exact_entity_metrics(splits['test'],clean_doc_spans)
    # Noise in the surrounding text only: the values themselves survive the OCR
    # noise, so this measures context robustness without punishing a value whose
    # characters were mangled and therefore tokenized differently.
    context_rows=[]
    for row in splits['test']:
        spans=[(int(e['start']),int(e['end'])) for e in row['entities']]
        context_rows.append({**row,'ocr_text':apply_ocr_noise(row['text'],row['seed']*97,corruption_rate,protect=spans)})
    context_windows=build_window_rows(context_rows,tok,max_length)
    context_logits,context_preds,context_offsets=_run_model(model,context_windows,tok,max_length,dev,batch_size)
    _,context_doc_spans=_document_level(context_windows,context_logits,context_preds,context_offsets)
    entity_context_only=exact_entity_metrics(splits['test'],context_doc_spans)
    per_family={}
    for family in sorted({r['template_family'] for r in splits['test']}):
        subset=[r for r in splits['test'] if r['template_family']==family]
        subset_ids={r['document_id'] for r in subset}
        family_metrics=exact_entity_metrics(subset,{k:v for k,v in per_doc_spans.items() if k in subset_ids})
        per_family[family]={'documents':len(subset),'gold_spans':family_metrics['true_spans'],
            'entity_span_f1':family_metrics['f1'] if family_metrics['true_spans'] else None}
    transcript_rows=[r for r in splits['test'] if r['document_type']=='transcript']
    transcript_ids={r['document_id'] for r in transcript_rows}
    transcript_metrics={'documents':len(transcript_rows),'entity_span_f1':exact_entity_metrics(transcript_rows,{k:v for k,v in per_doc_spans.items() if k in transcript_ids})['f1']}
    metrics={'task':'synthetic academic document type + entity extraction','data':'Procedural synthetic text data with length-preserving character substitutions; no images/real OCR.',
      'seed':seed,'generated_examples':count,'corruption_rate':corruption_rate,'encoder_layers':int(encoder_config.num_hidden_layers),'split_protocol':'entire synthetic school identities and document template families held out; evaluates only school/template intersections; all ten classes occur in each split',
      'template_families_by_split':ckpt['template_split'],'rows_by_split':{k:len(v) for k,v in splits.items()},'window_rows_by_split':{k:len(v) for k,v in window_splits.items()},
      'document_labels':list(DOCUMENT_TYPES),'entity_types':ENTITY_TYPES,
      'training_device':str(dev),'checkpoint_selection':'highest validation exact entity-span F1 (document level)',      'entity_loss_weight':entity_weight,'entity_label_token_counts':entity_label_counts,'entity_class_weights':{ENTITY_LABELS[i]:float(weights[i]) for i in range(len(ENTITY_LABELS))},'single_token_entity_types':single_token_types,'parameter_count':sum(p.numel() for p in model.parameters()),'checkpoint_bytes':(out/'model.pt').stat().st_size,'history':history,
      'test_ocr_text_corrupted':{'pytorch_multitask':neural_report,'tfidf_logistic':sparse_report},'test_clean_text':_document_type_metrics(splits['test'],clean_doc_logits),
      'entity_span_metrics_corrupted':entity_corrupted,'entity_span_metrics_clean':entity_clean,
      'entity_span_metrics_corrupted_context_only':entity_context_only,
      'entity_span_metric_notes':{
        'clean':'No OCR noise at all: the model reads the exact characters of the document.',
        'corrupted':'Every character of the document may be mistyped, including the characters inside a value. A mistyped value is tokenized differently, so exact-span agreement for that value is not expected even for a perfect reader.',
        'corrupted_context_only':'Only the characters outside the entity values may be mistyped. This isolates the effect of noise in the surrounding text on extraction.'},
      'entity_span_f1_by_template_family':per_family,'transcript_metrics':transcript_metrics,
      'scope_limit':'Synthetic-only metrics are not real-document, real OCR, institution-transfer or production evidence.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    _export_quantized(out,ckpt,model,window_splits['test'],tok,max_length,dev,splits['test'])
    return metrics


def _export_quantized(out,meta,model,test_windows,tokenizer,max_length,device,test_rows):
    """Write a dynamic-int8 copy of the model and measure its size and agreement.

    Dynamic int8 quantization targets CPU inference: the quantized linear kernels do
    not run on CUDA, so the agreement measurement is taken on the CPU.
    """
    try:
        # Quantize a CPU copy: dynamic int8 packing of CUDA weights produces packed
        # parameters that the CPU kernel cannot apply.
        cpu_model=PaceMultiTask(meta['base_model'],str(out/meta['encoder_config']))
        cpu_model.load_state_dict(meta['state_dict'])
        quantized=_quantize(cpu_model.to('cpu'))
    except Exception as error:  # pragma: no cover - quantization is optional tooling
        (out/'int8_report.json').write_text(json.dumps({'status':'skipped','reason':str(error)},indent=2)+'\n',encoding='utf-8')
        return None
    cpu=torch.device('cpu')
    quantized=quantized.to(cpu).eval()
    path=out/'model_int8.pt'
    torch.save({'state_dict':quantized.state_dict(),'base_model':meta['base_model'],'encoder_config':'encoder_config','tokenizer_dir':'tokenizer','doc_labels':meta['doc_labels'],'entity_labels':meta['entity_labels'],'max_length':meta['max_length'],'quantized':'dynamic_int8'},path)
    logits,preds,offsets=_run_model(quantized,test_windows,tokenizer,max_length,cpu)
    per_doc_logits,per_doc_spans=_document_level(test_windows,logits,preds,offsets)
    report={'status':'ok','quantized_bytes':path.stat().st_size,'fp32_bytes':(out/'model.pt').stat().st_size,
        'size_ratio':round(path.stat().st_size/(out/'model.pt').stat().st_size,4),
        'quantized_device':'cpu',
        'document_type_accuracy':_document_type_metrics(test_rows,per_doc_logits)['accuracy'],
        'entity_span_f1':exact_entity_metrics(test_rows,per_doc_spans)['f1']}
    (out/'int8_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/synthetic-multitask');p.add_argument('--count',type=int,default=30000);p.add_argument('--seed',type=int,default=20261019);p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--max-length',type=int,default=MAX_LENGTH);p.add_argument('--base-model',default=BASE_MODEL);p.add_argument('--corruption-rate',type=float,default=.02);p.add_argument('--encoder-layers',type=int,default=None,help='train a shallower encoder (fewer transformer layers) to trade accuracy for size');a=p.parse_args(argv)
    m=train_eval(a.output,a.count,a.seed,a.epochs,a.batch_size,a.max_length,a.base_model,a.corruption_rate,a.encoder_layers)
    print(json.dumps({'rows_by_split':m['rows_by_split'],'windows_by_split':m['window_rows_by_split'],'templates':m['template_families_by_split'],'doc_corrupt':{k:v['accuracy'] for k,v in m['test_ocr_text_corrupted'].items()},'doc_clean':m['test_clean_text']['accuracy'],'entity_corrupted':m['entity_span_metrics_corrupted'],'entity_clean':m['entity_span_metrics_clean'],'transcript':m['transcript_metrics'],'parameters':m['parameter_count'],'bytes':m['checkpoint_bytes']},indent=2))
class SyntheticDocumentEngine:
    """Load the synthetic-only multitask PyTorch artifact for local proposal generation."""
    def __init__(self,checkpoint_dir='artifacts/synthetic-multitask',device=None,max_windows=256,prefer_int8=False):
        p=Path(checkpoint_dir)
        if p.is_file():p=p.parent
        self.checkpoint=p/'model.pt'
        self.meta=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model=PaceMultiTask(self.meta['base_model'],str(p/self.meta['encoder_config']))
        self.model.load_state_dict(self.meta['state_dict'])
        self.quantized=False
        int8_path=p/'model_int8.pt'
        if prefer_int8 and int8_path.exists():
            int8=torch.load(int8_path,map_location='cpu',weights_only=False)
            self.model=_quantize(self.model)
            self.model.load_state_dict(int8['state_dict'])
            self.quantized=True
            # Quantized linear kernels are CPU-only, so the int8 artifact runs on CPU.
            self.device=torch.device('cpu')
        self.model.to(self.device).eval()
        self.tokenizer=AutoTokenizer.from_pretrained(p/self.meta['tokenizer_dir'],local_files_only=True)
        self.max_length=int(self.meta['max_length']);self.max_windows=max(1,int(max_windows))

    @torch.inference_mode()
    def analyze_document(self,text):
        if not isinstance(text,str):raise TypeError('text must be a string')
        if not text.strip():raise ValueError('text must not be empty')
        stride=min(32,max(1,self.max_length//4))
        encoded=self.tokenizer(text or ' ',truncation=True,max_length=self.max_length,stride=stride,
            return_overflowing_tokens=True,return_offsets_mapping=True,padding='max_length',return_tensors='pt')
        original_windows=len(encoded['input_ids']);count=min(original_windows,self.max_windows)
        ids=encoded['input_ids'][:count];mask=encoded['attention_mask'][:count]
        doc_logits=[];all_entities={};support={}
        tokenizer_truncated=original_windows>self.max_windows
        for start in range(0,count,16):
            end=min(start+16,count);dx,ex=self.model(ids[start:end].to(self.device),mask[start:end].to(self.device))
            doc_logits.extend(dx.cpu());probs=torch.softmax(ex,dim=-1).cpu();pred=ex.argmax(-1).cpu()
            for offset,seq,seq_probs in zip(encoded['offset_mapping'][start:end].tolist(),pred.tolist(),probs):
                visible=[(int(x),int(y)) for x,y in offset if int(y)>int(x)]
                if not visible:continue
                window_start,window_end=visible[0][0],visible[-1][1]
                for span in spans_from_iob(seq,offset):
                    a,b=span['start'],span['end']
                    if a<0 or b>len(text) or a>=b:continue
                    token_ids=[i for i,(x,y) in enumerate(offset) if x<b and y>a and y>x]
                    if not token_ids:continue
                    # A window boundary can cut a value's word in half. Such a
                    # fragment was never read whole, so it is only evidence when
                    # no window saw the whole value.
                    fragment=is_word_cut(text,a,b,window_start,window_end)
                    type_probabilities=entity_type_distribution(seq_probs,token_ids)
                    entity_type=max(type_probabilities,key=type_probabilities.get)
                    confidence=choice_confidence(type_probabilities.values())
                    candidate={**span,'type':entity_type,'text':text[a:b],
                        'type_probabilities':type_probabilities,'probabilities_calibrated':False,
                        'confidence':confidence,'confidence_semantics':'jev_choice_sharpness',
                        'source_text':text[a:b],'decision_scope':'conditional_on_proposed_span_boundary',
                        'requires_review':True,'confidence_calibrated':False}
                    key=(entity_type,a,b)
                    if key not in all_entities or confidence>all_entities[key]['confidence']:all_entities[key]=candidate
                    seen=support.setdefault(key,{'complete':False,'truncated':False})
                    if fragment:seen['truncated']=True
                    else:seen['complete']=True
        if not doc_logits:raise RuntimeError('No model windows were evaluated')
        # A value whose word every window cut in half was never read whole, so its
        # span is unreliable evidence and is dropped instead of reported as a fact.
        for key,seen in list(support.items()):
            if not seen['complete']:
                all_entities.pop(key,None)
        for key,entity in all_entities.items():entity['window_truncated']=support[key]['truncated']
        # A span with no letter or digit in it is punctuation, not a value: the tagger
        # sometimes latches onto the hyphen in a masked phone number and calls it a
        # name. No entity type has a punctuation-only value.
        for key in [key for key,entity in all_entities.items() if not any(char.isalnum() for char in entity['text'])]:
            del all_entities[key]
        # A value is written on one line. A span that crosses a line break has run
        # into the next row of the page, which is a different value.
        for key in [key for key,entity in all_entities.items() if '\n' in entity['text'] or '\r' in entity['text']]:
            del all_entities[key]
        # Overlapping windows can produce both a complete span and a truncated prefix.
        # Suppress only same-type containment alternatives; preserve partial overlaps.
        by_type={}
        for entity in all_entities.values():by_type.setdefault(entity['type'],[]).append(entity)
        selected=[]
        for candidates in by_type.values():
            accepted=[]
            # A complete reading of a value beats a truncated prefix of the same
            # value, and a longer reading beats its own prefix.
            for entity in sorted(candidates,key=lambda e:(e.get('window_truncated',False),-(e['end']-e['start']),-e['confidence'],e['start'],e['end'])):
                contained=any(
                    (other['start']<=entity['start'] and entity['end']<=other['end'])
                    or (entity['start']<=other['start'] and other['end']<=entity['end'])
                    for other in accepted
                )
                if not contained:accepted.append(entity)
            selected.extend(accepted)
        all_entities={(entity['type'],entity['start'],entity['end']):entity for entity in selected}
        # A mark is its own table cell. A mark-like token that sits inside a course
        # title ("Content English - SS") belongs to that title, so the narrower
        # reading wins and the overlapping mark proposal is dropped.
        course_spans=[(entity['start'],entity['end']) for entity in all_entities.values() if entity['type']=='COURSE']
        if course_spans:
            for key in [key for key,entity in all_entities.items()
                        if entity['type']=='MARK'
                        and any(start<=entity['start'] and entity['end']<=end for start,end in course_spans)]:
                del all_entities[key]
        logits=torch.stack(doc_logits).mean(0)
        document_probs=torch.softmax(logits,-1)
        confidence=choice_confidence(document_probs.tolist())
        label=DOCUMENT_TYPES[int(document_probs.argmax(-1))]
        from .text_structure import split_sections
        sections=[{'title':s['title'],'source_text':s['text'],'interpretation':'heading_and_paragraph_split_only'} for s in split_sections(text)]
        from .extract import extract_candidates
        literal_candidates=extract_candidates(text)
        model_spans=[(entity['start'],entity['end']) for entity in all_entities.values()]
        literal_spans=[]
        for values in literal_candidates.values():
            for item in values:
                overlaps_model=any(item['start']<end and start<item['end'] for start,end in model_spans)
                overlaps_literal=any(item['start']<end and start<item['end'] for start,end in literal_spans)
                if not overlaps_model and not overlaps_literal:
                    literal_spans.append((item['start'],item['end']))
                    all_entities[('LITERAL_CANDIDATE',item['start'],item['end'])]={
                        'type':'LITERAL_CANDIDATE','start':item['start'],'end':item['end'],
                        'text':item['text'],'confidence':item['confidence'],
                        'confidence_semantics':'literal_pattern_match_not_model_confidence',
                        'source_text':item['text'],'evidence':item['evidence'],
                        'requires_review':True,'confidence_calibrated':False,'probabilities_calibrated':False}
        return {'schema_version':'pace-synthetic-ir/0.1','model_scope':'synthetic_text_only_unvalidated_on_real_documents',
            'document_type':{'label':label,'proposed_label':label,'confidence':float(confidence),
                'probabilities':{name:float(document_probs[i]) for i,name in enumerate(DOCUMENT_TYPES)},
                'confidence_semantics':'jev_choice_sharpness','confidence_calibrated':False,
                'probabilities_calibrated':False,'requires_review':True},
            'source_text':text,
            'confidence_semantics':'confidence is Jev Choice sharpness from normalized model distributions; probabilities and confidence are uncalibrated and are not validated correctness likelihoods',
            'probabilities_calibrated':False,
            'sections':sections,'entities':sorted(all_entities.values(),key=lambda x:(x['start'],x['end'],x['type'])),
            'source_characters':len(text),'window_count':count,'window_limit':self.max_windows,'window_limit_reached':original_windows>self.max_windows,
            'entities_truncated':tokenizer_truncated,'candidate_extractor':'literal_patterns_are_untyped_proposals',
            'quantized_int8':self.quantized,
            'requires_review':True,'commit_policy':'proposals_only_user_confirmation_required'}

    def analyze(self, text: str) -> dict[str, Any]:
        """Analyze a document and attach the deterministic transcript IR when it applies."""
        result = self.analyze_document(text)
        return attach_transcript_ir(result, text)


if __name__=='__main__':main()
