"""Inspect named Hugging Face real datasets without altering natural records."""
from __future__ import annotations
import argparse,csv,json,statistics
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from .model import BASE_MODEL


def quantiles(values):
 if not values:return {"count":0}
 ordered=sorted(values)
 def q(p):return ordered[min(len(ordered)-1,round((len(ordered)-1)*p))]
 return {"count":len(ordered),"mean":round(statistics.mean(ordered),2),"median":q(.5),"p90":q(.9),"p95":q(.95),"max":q(1)}


def add_token_lengths(stats, texts, tokenizer):
 lengths=[]
 for text in texts:
  # Use tokenizer's unbounded tokenize method to inspect lengths beyond the model limit.
  lengths.append(len(tokenizer.tokenize(text))+int(tokenizer.num_special_tokens_to_add(pair=False)))
 return quantiles(lengths)

def inspect_gu(path,tokenizer):
 table=pq.read_table(path); rows=table.to_pylist(); fields=table.schema.names
 lengths=[]; row_texts=[]; cell_texts=[]
 for row in rows:
  values=[str(row.get(k) or "") for k in fields if k not in {"__index_level_0__"}]
  lengths.append(sum(len(v.split()) for v in values));row_texts.append(" ".join(values))
  cell_texts.extend(v for v in values if v.strip())
 return {"rows":len(rows),"fields":fields,"text_word_lengths_all_fields":quantiles(lengths),
         "row_token_lengths_all_fields_minilm":add_token_lengths(stats=None,texts=row_texts,tokenizer=tokenizer),
         "cell_token_lengths_minilm":add_token_lengths(stats=None,texts=cell_texts,tokenizer=tokenizer),
         "field_nonempty_counts":{k:sum(bool(str(r.get(k) or "").strip()) for r in rows) for k in fields},
         "sample":{k:(str(rows[0].get(k) or "")[:450]) for k in fields if k!="__index_level_0__"}}


def inspect_usc(path,tokenizer):
 with open(path,encoding="utf-8-sig",newline="") as f:
  reader=csv.DictReader(f); rows=list(reader); fields=reader.fieldnames or []
 text_fields=["Class Name","Catalogue","Prerequisites","Restrictions","Units","Time","Days","Class Type"]
 lengths=[sum(len(str(r.get(k) or "").split()) for k in text_fields) for r in rows]
 cell_texts=[str(r.get(k) or "").strip() for r in rows for k in text_fields if str(r.get(k) or "").strip().casefold() not in {"not available","n/a","none","null"}]
 row_texts=[" ".join(str(r.get(k) or "") for k in text_fields) for r in rows]
 return {"rows":len(rows),"fields":fields,"academic_text_word_lengths":quantiles(lengths),
         "academic_cell_token_lengths_minilm":add_token_lengths(None,cell_texts,tokenizer),
         "academic_row_token_lengths_minilm":add_token_lengths(None,row_texts,tokenizer),
         "nonempty_counts":{k:sum(bool((r.get(k) or "").strip()) for r in rows) for k in fields},
         "sample":{k:rows[0].get(k) for k in fields}}


def inspect_cu(path,tokenizer):
 data=json.loads(Path(path).read_text(encoding="utf-8"));
 types=Counter(); page_counts=[]; block_counts=[]; block_text_lengths=[]; syllabi=[]
 for doc in data:
  meta=doc.get("metadata",{}); gt=doc.get("ground_truth",{})
  pages=gt.get("pages",[]); page_counts.append(len(pages))
  if "syllabus" in str(meta.get("document_type","")).casefold(): syllabi.append(meta)
  for page in pages:
   blocks=page.get("blocks",[]); block_counts.append(len(blocks))
   for block in blocks:
    types[block.get("type","unknown")]+=1
    block_text_lengths.append(len((block.get("text") or "").split()))
 block_texts=[(block.get("text") or "") for doc in data for page in doc.get("ground_truth",{}).get("pages",[]) for block in page.get("blocks",[]) if (block.get("text") or "").strip()]
 page_texts=[" ".join(block.get("text") or "" for block in page.get("blocks",[])) for doc in data for page in doc.get("ground_truth",{}).get("pages",[])]
 return {"documents":len(data),"document_types":dict(Counter(str(d.get("metadata",{}).get("document_type","unknown")) for d in data)),
         "annotated_block_token_lengths_minilm":add_token_lengths(None,block_texts,tokenizer),
         "gold_page_token_lengths_minilm":add_token_lengths(None,page_texts,tokenizer),
         "documents_with_syllabus_in_type":len(syllabi),"syllabus_examples":syllabi,
         "pages_per_document":quantiles(page_counts),"blocks_per_page":quantiles(block_counts),
         "block_types":dict(types),"words_per_annotated_block":quantiles(block_text_lengths),
         "has_ground_truth":sum(bool(d.get("ground_truth")) for d in data),
         "sample_keys":list(data[0].keys()) if data else []}


def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--output',default='data/inventory.json');a=p.parse_args(argv)
 tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL)
 files={"anordkvist/gu-course-syllabus":"data/raw/gu_course_syllabus.parquet","USC/USC-Course-Catalog":"data/raw/usc_course_data.csv","iristun/CU-DocBench":"data/raw/cu_docbench.json"}
 result={"datasets":{}}
 result['datasets']["anordkvist/gu-course-syllabus"]={**inspect_gu(files["anordkvist/gu-course-syllabus"],tokenizer),"raw_size_bytes":Path(files["anordkvist/gu-course-syllabus"]).stat().st_size,"license":"No license specified in HF card; do not train/redistribute until rights are clarified.","task_fit":"Natural row-level course syllabus sections and explicit field headings; field-to-section classification and direct span supervision possible, not document-type supervision."}
 result['datasets']["USC/USC-Course-Catalog"]={**inspect_usc(files["USC/USC-Course-Catalog"],tokenizer),"raw_size_bytes":Path(files["USC/USC-Course-Catalog"]).stat().st_size,"license":"Hugging Face card declares Apache-2.0; provenance is contributor scrape of USC 2024 Spring classes, not verified USC grant. Record model-card declaration and seek confirmation before commercial redistribution.","task_fit":"Natural catalog rows with prerequisite fields; relation/field extraction and text-to-field mapping; includes instructor personal names and meeting data, exclude unnecessary identity fields."}
 result['datasets']["iristun/CU-DocBench"]={**inspect_cu(files["iristun/CU-DocBench"],tokenizer),"raw_size_bytes":Path(files["iristun/CU-DocBench"]).stat().st_size,"license":"Hugging Face card declares MIT; inspect original-document rights and benchmark conditions before republishing examples/model-derived outputs.","task_fit":"Small gold page/block layout/OCR benchmark, including syllabus pages and synthetic/mock transcript. Evaluation only; 23 document-level examples, 100 selected representative pages as card claims."}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
