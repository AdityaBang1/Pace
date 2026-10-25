"""Fine-tune compact encoders on naturally labeled USC catalog cell text."""
from __future__ import annotations
import argparse,json,random,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,AutoConfig
from .model import BASE_MODEL,MAX_LENGTH,PaceClassifier
from .real_data import load_usc_examples,split_real_groups,exclude_intrinsically_ambiguous

class Rows(Dataset):
 def __init__(self,rows,tokenizer,labels,max_length):
  self.y=torch.tensor([labels.index(r['label']) for r in rows]);self.x=tokenizer([r['text'] for r in rows],padding='max_length',truncation=True,max_length=max_length,return_tensors='pt')
 def __len__(self):return len(self.y)
 def __getitem__(self,i):return {'input_ids':self.x['input_ids'][i],'attention_mask':self.x['attention_mask'][i],'labels':self.y[i]}

def predict(model,loader,device):
 model.eval(); logits=[]; ys=[]
 with torch.inference_mode():
  for batch in loader:
   logits.append(model(batch['input_ids'].to(device),batch['attention_mask'].to(device)).float().cpu());ys.extend(batch['labels'].tolist())
 return torch.cat(logits),np.array(ys)

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--input',default='data/raw/usc_course_data.csv');p.add_argument('--output',default='artifacts/usc-miniLM');p.add_argument('--epochs',type=int,default=4);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--seed',type=int,default=20261019);p.add_argument('--base-model',default=BASE_MODEL);p.add_argument('--max-length',type=int,default=MAX_LENGTH);p.add_argument('--min-class-count',type=int,default=25);p.add_argument('--split-assignment',default=None,help='JSON file mapping course subject prefixes to fixed train/validation/test membership');a=p.parse_args(argv)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
 all_rows=load_usc_examples(a.input);rows,ambiguous,ambiguous_rows=exclude_intrinsically_ambiguous(all_rows);counts=Counter(r['label'] for r in rows);classes=sorted(k for k,v in counts.items() if v>=a.min_class_count);rows=[r for r in rows if r['label'] in classes]
 fixed_assignment=json.loads(Path(a.split_assignment).read_text(encoding='utf-8')) if a.split_assignment else None
 splits=split_real_groups(rows,a.seed,fixed_assignment=fixed_assignment)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 from .data import save_jsonl
 for name,data in splits.items():save_jsonl(out/f'{name}.jsonl',data)
 tok=AutoTokenizer.from_pretrained(a.base_model);tok.save_pretrained(out/'tokenizer');AutoConfig.from_pretrained(a.base_model).save_pretrained(out/'encoder_config')
 ds={n:Rows(v,tok,classes,a.max_length) for n,v in splits.items()};device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 loaders={n:DataLoader(d,batch_size=a.batch_size,shuffle=n=='train',num_workers=0) for n,d in ds.items()}
 model=PaceClassifier(a.base_model,num_classes=len(classes)).to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01);best=-1;beststate=None;history=[]
 print(json.dumps({'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'labels':classes,'rows':{k:len(v) for k,v in splits.items()}},indent=2),flush=True)
 for epoch in range(1,a.epochs+1):
  model.train();t=time.perf_counter();losses=[]
  for b in loaders['train']:
   opt.zero_grad(set_to_none=True);z=model(b['input_ids'].to(device),b['attention_mask'].to(device));loss=torch.nn.functional.cross_entropy(z,b['labels'].to(device));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();losses.append(float(loss.detach()))
  vz,vy=predict(model,loaders['validation'],device);vp=vz.argmax(-1).numpy();vf=float(f1_score(vy,vp,average='macro'));rec={'epoch':epoch,'loss':float(np.mean(losses)),'validation_accuracy':float(accuracy_score(vy,vp)),'validation_macro_f1':vf,'seconds':time.perf_counter()-t};history.append(rec);print(json.dumps(rec),flush=True)
  if vf>best:best=vf;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(beststate);vz,vy=predict(model,loaders['validation'],device);tz,ty=predict(model,loaders['test'],device)
 # Fit scalar temperature on validation only. Test set remains untouched for calibration.
 temperature=torch.nn.Parameter(torch.ones(()));temp_opt=torch.optim.LBFGS([temperature],lr=.1,max_iter=50)
 def calibrate():
  temp_opt.zero_grad();loss=torch.nn.functional.cross_entropy(vz/temperature.clamp(.05,20),torch.tensor(vy,dtype=torch.long));loss.backward();return loss
 temp_opt.step(calibrate);temp=float(temperature.detach().clamp(.05,20));probs=torch.softmax(tz/temp,-1);confidence,cal_pred=probs.max(-1);confidence=confidence.numpy();pred=tz.argmax(-1).numpy();cal_pred=cal_pred.numpy();correct=(cal_pred==ty)
 ece=0.0
 for bin_idx in range(10):
  lo=bin_idx/10;hi=(bin_idx+1)/10;mask=(confidence>=lo)&((confidence<hi) if bin_idx<9 else confidence<=hi)
  if mask.any():ece+=float(mask.mean()*abs(confidence[mask].mean()-correct[mask].mean()))
 risk=[]
 for threshold in [.5,.6,.7,.8,.9,.95,.99]:
  mask=confidence>=threshold;risk.append({'threshold':threshold,'coverage':float(mask.mean()),'precision':float(correct[mask].mean()) if mask.any() else None})
 high=confidence>=.9
 report=classification_report(ty,pred,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)
 ckpt={'state_dict':beststate,'base_model':a.base_model,'num_classes':len(classes),'encoder_config':'encoder_config','tokenizer_dir':'tokenizer','labels':classes,'max_length':a.max_length,'seed':a.seed,'temperature':temp,'ambiguous_values':ambiguous,'selected_epoch':max(history,key=lambda x:x['validation_macro_f1'])['epoch']}
 torch.save(ckpt,out/'model.pt')
 metrics={'task':'natural catalog cell field classification','dataset':'USC/USC-Course-Catalog','source_group':'course subject prefixes; entire groups held out','subject_split_assignment_file':a.split_assignment or None,'subject_split_assignment':{group:split for split,part in splits.items() for group in {r['group_id'] for r in part}},'rows_total_after_rare_filter':len(rows),'ambiguous_values':ambiguous,'ambiguous_text_value_count':len(ambiguous),'ambiguous_rows_excluded':len(ambiguous_rows),'rows_by_split':{k:len(v) for k,v in splits.items()},'subjects_by_split':{k:len({r['group_id'] for r in v}) for k,v in splits.items()},'labels':classes,'label_counts':dict(Counter(r['label'] for r in rows)),'test_accuracy':float(accuracy_score(ty,pred)),'test_macro_f1':float(f1_score(ty,pred,average='macro')),'test_weighted_f1':float(f1_score(ty,pred,average='weighted')),'test_nll_temperature_scaled':float(torch.nn.functional.cross_entropy(tz/temp,torch.as_tensor(ty,dtype=torch.long))),'real_validation_temperature':temp,'test_ece_10bin':ece,'high_confidence_threshold':.9,'high_confidence_coverage':float(high.mean()),'high_confidence_precision':float(correct[high].mean()) if high.any() else None,'risk_coverage':risk,'classification_report':report,'test_error_examples':[{'true':classes[int(ty[i])],'predicted':classes[int(pred[i])],'text':splits['test'][i]['text'],'subject':splits['test'][i]['group_id']} for i in range(len(ty)) if ty[i]!=pred[i]],'confusion_matrix':confusion_matrix(ty,pred).tolist(),'history':history,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'checkpoint_bytes':(out/'model.pt').stat().st_size,'parameter_count':sum(p.numel() for p in model.parameters()),'real_document_metrics':False,'data_source':'real USC course catalog cell fields; not real scanned/document OCR evaluation','semantics':'classification on natural catalog cell text; field labels are naturally available ground truth','class_diagnostics':{label:{'errors':int(((ty==i)&(pred!=i)).sum()),'false_positives':int(((ty!=i)&(pred==i)).sum())} for i,label in enumerate(classes)},'ambiguous_examples_by_field':dict(Counter(row['label'] for row in ambiguous_rows))}
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in metrics.items() if k not in {'classification_report','confusion_matrix'}},indent=2))
if __name__=='__main__':main()
