"""Detailed classification and confidence evaluation on saved synthetic splits."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .data import load_jsonl
from .model import CLASS_NAMES,PaceClassifier,MAX_LENGTH
from .train import EncodedRows


def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='artifacts/pace_classifier.pt'); p.add_argument('--test',default='data/processed/test.jsonl'); p.add_argument('--calibration',default='artifacts/calibration.json'); p.add_argument('--output',default='artifacts/detailed_metrics.json'); args=p.parse_args(argv)
 ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model=PaceClassifier(ckpt['base_model']); model.load_state_dict(ckpt['state_dict']); model.to(device).eval()
 tokenizer=AutoTokenizer.from_pretrained(ckpt['base_model']); rows=load_jsonl(args.test); ds=EncodedRows(rows,tokenizer); loader=DataLoader(ds,batch_size=64)
 logits=[]; labels=[]
 with torch.inference_mode():
  for batch in loader:
   logits.append(model(batch['input_ids'].to(device),batch['attention_mask'].to(device)).float().cpu()); labels.extend(batch['labels'].tolist())
 logits=torch.cat(logits); y=np.array(labels); temp=json.loads(Path(args.calibration).read_text())['temperature']; probs=torch.softmax(logits/temp,-1).numpy(); pred=probs.argmax(-1); conf=probs.max(-1)
 output={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'temperature':temp,'ece_10bin':0.0,'per_class':{},'confusion_matrix':confusion_matrix(y,pred).tolist(),'labels':CLASS_NAMES,'risk_coverage':[],'test_count':len(y),'source':'synthetic only'}
 for i in range(10):
  mask=(conf>=i/10)&(conf<(i+1)/10 if i<9 else conf<=1)
  if mask.any(): output['ece_10bin']+=float(mask.mean()*abs(conf[mask].mean()-(pred[mask]==y[mask]).mean()))
 for idx,name in enumerate(CLASS_NAMES):
  mask=y==idx
  if mask.any():
   precision,recall,f1,_=precision_recall_fscore_support(y,pred,labels=[idx],average=None,zero_division=0)
   output['per_class'][name]={'support':int(mask.sum()),'precision':float(precision[0]),'recall':float(recall[0]),'f1':float(f1[0])}
 for threshold in [0.5,0.6,0.7,0.8,0.9,0.95,0.99]:
  mask=conf>=threshold; output['risk_coverage'].append({'threshold':threshold,'coverage':float(mask.mean()),'precision':float((pred[mask]==y[mask]).mean()) if mask.any() else None})
 out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8'); print(json.dumps({k:v for k,v in output.items() if k not in ('per_class','confusion_matrix')},indent=2))

if __name__=='__main__': main()
