"""CPU-friendly task baseline using character n-gram TF-IDF and logistic regression."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from .data import SEED, generate_examples, split_by_group


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--examples',type=int,default=6000); p.add_argument('--seed',type=int,default=SEED); p.add_argument('--output',default='artifacts'); args=p.parse_args(argv)
    splits=split_by_group(generate_examples(args.examples,args.seed),args.seed)
    train,test=splits['train'],splits['test']
    model=make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=1,max_features=150000,sublinear_tf=True),LogisticRegression(C=4.0,max_iter=500,random_state=args.seed))
    start=time.perf_counter(); model.fit([r['text'] for r in train],[f"{r['task']}::{r['label']}" for r in train]); fit_s=time.perf_counter()-start
    start=time.perf_counter(); pred=model.predict([r['text'] for r in test]); infer_s=time.perf_counter()-start
    y=[f"{r['task']}::{r['label']}" for r in test]
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    import joblib
    joblib.dump(model,out/'char_tfidf_baseline.joblib',compress=3)
    result={'model':'char-ngram TF-IDF + logistic regression','test_accuracy':float(accuracy_score(y,pred)),
            'test_macro_f1':float(f1_score(y,pred,average='macro')),'train_examples':len(train),'test_examples':len(test),
            'fit_seconds':fit_s,'batch_inference_seconds':infer_s,'saved_bytes':(out/'char_tfidf_baseline.joblib').stat().st_size,
            'split_seed':args.seed,'source':'synthetic only; paraphrase-family split'}
    (out/'baseline_metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
