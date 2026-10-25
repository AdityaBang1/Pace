"""Benchmark local real-catalog prediction on naturally observed field texts."""
from __future__ import annotations
import argparse,json,statistics,time
from pathlib import Path
import torch
from .real_data import load_usc_examples,exclude_intrinsically_ambiguous
from .real_infer import RealCatalogEngine

def percentile(values,p):
 values=sorted(values)
 return values[min(len(values)-1,round((len(values)-1)*p))]

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='artifacts/usc-miniLM/model.pt');p.add_argument('--input',default='data/raw/usc_course_data.csv');p.add_argument('--output',default='artifacts/usc-miniLM/benchmark.json');p.add_argument('--runs',type=int,default=500);a=p.parse_args(argv)
 rows,_,_=exclude_intrinsically_ambiguous(load_usc_examples(a.input));texts=[r['text'] for r in rows[:a.runs]]
 engine=RealCatalogEngine(a.checkpoint);device=engine.device
 for text in texts[:10]:engine.classify_field(text)
 if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 durations=[]
 for text in texts:
  if device.type=='cuda':torch.cuda.synchronize()
  start=time.perf_counter();engine.classify_field(text)
  if device.type=='cuda':torch.cuda.synchronize()
  durations.append((time.perf_counter()-start)*1000)
 token_counts=[len(engine.tokenizer.tokenize(text))+2 for text in texts]
 result={'runs':len(durations),'dataset':'USC/USC-Course-Catalog natural non-ambiguous cell values','device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'latency_including_tokenization_ms_p50':statistics.median(durations),'latency_including_tokenization_ms_p95':percentile(durations,.95),'throughput_rows_per_second':1000/statistics.mean(durations),'input_tokens_p50':statistics.median(token_counts),'input_tokens_p95':percentile(token_counts,.95),'checkpoint_bytes':Path(a.checkpoint).stat().st_size,'parameter_count':sum(p.numel() for p in engine.model.parameters()),'torch_cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,'scope':'batch=1; includes Hugging Face tokenization and PyTorch forward, excludes disk/model load, OCR, document rendering, app overhead; desktop GPU result, not phone performance'}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
