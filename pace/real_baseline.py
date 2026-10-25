"""Held-out real academic catalog field classification baselines."""
from __future__ import annotations
import argparse,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from sklearn.pipeline import make_pipeline
from .real_data import load_usc_examples,split_real_groups,exclude_intrinsically_ambiguous


def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--input',default='data/raw/usc_course_data.csv');p.add_argument('--output',default='artifacts/usc-real');p.add_argument('--min-class-count',type=int,default=25);p.add_argument('--max-features',type=int,default=250000);a=p.parse_args(argv)
 all_rows=load_usc_examples(a.input);rows,ambiguous,excluded=exclude_intrinsically_ambiguous(all_rows);counts=Counter(r['label'] for r in rows);allowed={k for k,n in counts.items() if n>=a.min_class_count};rows=[r for r in rows if r['label'] in allowed];splits=split_real_groups(rows)
 fixed_assignment={group:split for split,part in splits.items() for group in {r['group_id'] for r in part}}
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 fixed_assignment={group:split for split,part in splits.items() for group in {r['group_id'] for r in part}}
 (out/'split_assignment.json').write_text(json.dumps(fixed_assignment,indent=2)+'\n',encoding='utf-8')
 for name,part in splits.items():
  from .data import save_jsonl
  save_jsonl(out/f'{name}.jsonl',part)
 train,test=splits['train'],splits['test']; labels=sorted(allowed)
 model=make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=2,max_features=a.max_features,sublinear_tf=True),LogisticRegression(C=4.0,max_iter=600,class_weight='balanced',random_state=20261019))
 started=time.perf_counter();model.fit([r['text'] for r in train],[r['label'] for r in train]);fit_s=time.perf_counter()-started
 started=time.perf_counter();pred=model.predict([r['text'] for r in test]);infer_s=time.perf_counter()-started;y=[r['label'] for r in test]
 report=classification_report(y,pred,labels=labels,output_dict=True,zero_division=0)
 import joblib;joblib.dump(model,out/'usc_char_tfidf.joblib',compress=3)
 metrics={'dataset':'USC/USC-Course-Catalog','source':'real academic catalog rows; held out complete course subject prefixes','rows_all_fields':len(all_rows),'rows_used':len(rows),'ambiguous_text_value_count':len(ambiguous),'ambiguous_examples_excluded':len(excluded),'ambiguous_values':ambiguous,'ambiguous_values_count':len(ambiguous),'ambiguous_examples_by_field':dict(Counter(r['label'] for r in excluded)),'field_counts_after_ambiguity_filter':dict(counts),'split_assignment_file':'split_assignment.json','labels_used':labels,'rare_classes_dropped':sorted(set(counts)-allowed),'split_rows':{k:len(v) for k,v in splits.items()},'split_subject_counts':{k:len({r['group_id'] for r in v}) for k,v in splits.items()},'subject_split_assignment':fixed_assignment,'test_accuracy':accuracy_score(y,pred),'test_macro_f1':f1_score(y,pred,average='macro'),'test_weighted_f1':f1_score(y,pred,average='weighted'),'per_class':report,'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(),'test_predictions':pred.tolist(),'test_true_labels':y,'fit_seconds':fit_s,'batch_inference_seconds':infer_s,'serialized_model_bytes':(out/'usc_char_tfidf.joblib').stat().st_size,'seed':20261019,'subject_split_assignment_file':'split_assignment.json','note':'USC dataset card says Apache-2.0; underlying origin is scraped university catalog, model-card permission is recorded but not independently verified.'}
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in metrics.items() if k not in {'per_class','confusion_matrix'}},indent=2))

if __name__=='__main__':main()
