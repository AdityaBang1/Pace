"""Reproducible local inference timing on synthetic academic OCR-like snippets."""
from __future__ import annotations
import argparse, json, statistics, time, tracemalloc
from pathlib import Path
import torch
from transformers import AutoTokenizer
from .model import AutoConfig
from .data import generate_examples
from .model import CLASS_NAMES, MAX_LENGTH, PaceClassifier


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='artifacts/pace_classifier.pt'); p.add_argument('--output',default='artifacts/benchmark.json'); p.add_argument('--runs',type=int,default=200); args=p.parse_args(argv)
    path=Path(args.checkpoint); ckpt=torch.load(path,map_location='cpu',weights_only=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_dir = str((path.parent / ckpt['encoder_config']).resolve()) if ckpt.get('encoder_config') else None
    model=PaceClassifier(ckpt['base_model'],config_dir=config_dir); model.load_state_dict(ckpt['state_dict']); model.to(device).eval()
    tokenizer_path = path.parent / ckpt['tokenizer_dir'] if ckpt.get('tokenizer_dir') else None
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_path),local_files_only=True) if tokenizer_path and tokenizer_path.exists() else AutoTokenizer.from_pretrained(ckpt['base_model'])
    rows=generate_examples(1000)
    texts=[r['text'] for r in rows[:args.runs]]
    enc=tokenizer(texts,padding='max_length',truncation=True,max_length=MAX_LENGTH,return_tensors='pt')
    input_ids=enc['input_ids'].to(device); mask=enc['attention_mask'].to(device)
    with torch.inference_mode():
        for _ in range(10): model(input_ids[:1],mask[:1])
        times=[]
        if device.type=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        for i in range(len(texts)):
            if device.type=='cuda': torch.cuda.synchronize()
            start=time.perf_counter(); model(input_ids[i:i+1],mask[i:i+1])
            if device.type=='cuda': torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
    stats={'runs':len(times),'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
           'latency_ms_p50':statistics.median(times),'latency_ms_p95':sorted(times)[int(.95*(len(times)-1))],
           'throughput_examples_per_second':1000/statistics.mean(times),'model_checkpoint_bytes':path.stat().st_size,
           'parameter_count':sum(p.numel() for p in model.parameters()),
           'torch_cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
           'scope':'Python/PyTorch; batch size 1; fixed 160 tokens; excludes OCR, PDF render, app and Core ML'}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8'); print(json.dumps(stats,indent=2))

if __name__=='__main__': main()
