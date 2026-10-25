"""Convert the trained fixed-length text classifier to Core ML on supported hosts."""
from __future__ import annotations
import argparse, json, platform
from pathlib import Path
import torch
from .model import PaceClassifier, MAX_LENGTH


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='artifacts/pace_classifier.pt'); p.add_argument('--output',default='artifacts/coreml/PaceClassifier.mlpackage'); args=p.parse_args(argv)
    if platform.system()!='Darwin':
        raise SystemExit('Core ML conversion is only supported by this project workflow on macOS. This Windows/Linux host cannot validate Core ML runtime or produce a verified Apple deployment artifact.')
    import coremltools as ct
    import numpy as np
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=PaceClassifier(ckpt['base_model']); model.load_state_dict(ckpt['state_dict']); model.eval()
    class Wrapper(torch.nn.Module):
        def __init__(self,m): super().__init__(); self.model=m
        def forward(self,input_ids,attention_mask): return self.model(input_ids.long(),attention_mask.long())
    wrapper=Wrapper(model).eval()
    example=(torch.ones((1,MAX_LENGTH),dtype=torch.long),torch.ones((1,MAX_LENGTH),dtype=torch.long))
    traced=torch.jit.trace(wrapper,example,strict=False)
    converted=ct.convert(traced,inputs=[ct.TensorType(name='input_ids',shape=example[0].shape,dtype=np.int32),ct.TensorType(name='attention_mask',shape=example[1].shape,dtype=np.int32)],convert_to='mlprogram',compute_precision=ct.precision.FLOAT16,minimum_deployment_target=ct.target.iOS15)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); converted.save(str(out))
    (out.parent/'coreml_metadata.json').write_text(json.dumps({'base_model':ckpt['base_model'],'max_length':MAX_LENGTH,'labels':ckpt['class_names'],'precision':'float16','minimum_ios':'15','runtime_validation':'requires Apple OS; not performed by conversion'},indent=2)+'\n',encoding='utf-8')
    print(f'Saved {out}')

if __name__=='__main__': main()
