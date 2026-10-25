"""Temporary: time forward+backward for bigger encoders at a realistic length."""
from __future__ import annotations

import time

import torch
from transformers import AutoModel, AutoTokenizer

CANDIDATES = [
    ("answerdotai/ModernBERT-base", 64),
    ("answerdotai/ModernBERT-large", 16),
    ("BAAI/bge-base-en-v1.5", 64),
]


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    for name, batch in CANDIDATES:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).cuda().train()
        head = torch.nn.Linear(model.config.hidden_size, 42).cuda()
        opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-5)
        width = 96
        ids = torch.randint(5, 2000, (batch, width)).cuda()
        mask = torch.ones_like(ids)
        for step in range(3):
            if step == 0:
                start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=mask).last_hidden_state
                loss = torch.nn.functional.cross_entropy(
                    head(out.mean(dim=1)).float(), torch.zeros(batch, dtype=torch.long).cuda())
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / 2
        print(f"{name:38s} bs{batch:>3} {seconds*1000:7.0f} ms/step "
              f"{batch/seconds:8.0f} rows/s  peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        del model, head, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
