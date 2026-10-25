"""What kind of document is this, and is it academic at all?

The audit's worst finding was here: the shipped model was trained on generated
documents, its artifact did not load, and it called a bank statement a transcript
at 0.9975 confidence. This module replaces it with a model trained on real text
and evaluated against real non-academic text.

Three decisions keep it honest.

* **Only classes with real support are trained.** Syllabi come from 14 real
  schools; catalog pages and schedules are built from real catalog rows (the
  text is real, the page layout is composed); ``other`` is real news prose from
  AG News plus a small set of hard non-academic documents. Transcripts are not a
  class here -- two real transcripts cannot train one -- and the deterministic
  transcript detector already covers them.
* **``other`` is a real class**, not a catch-all guessed from absence.
* **The threshold is chosen from the measured false-academic rate**, because the
  product question is "how often does this invent an academic document?", not
  "what is the F1?".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

CORPUS = Path("data/processed/field/corpus.jsonl")
SYLLABI = Path("data/raw/hf/syllabus-cleaned/ingested/syllabi_with_text.jsonl")
NEWS = Path("data/raw/ag_news/data/train-00000-of-00001.parquet")
DEFAULT_OUTPUT = Path("artifacts/doc-type")

LABELS = ["syllabus", "course_catalog", "schedule", "other"]

#: Hard negatives written to look like a document a student might photograph.
#: Training templates and test templates are worded differently on purpose.
HARD_NEGATIVES_TRAIN = [
    """BLUE BOTTLE COFFEE
Order #4471        Table 12
2 x Cortado            9.00
1 x Almond Croissant   4.25
Subtotal               13.25
Tax                     1.16
Total                  14.41
Card **** 8821   Tip 20%
Thank you""",
    """STATEMENT OF ACCOUNT
Account 4412-9931
Period 01 Aug - 31 Aug 2026
Opening balance      2,410.22
05 Aug  Payroll       3,100.00
07 Aug  Rent         -1,850.00
14 Aug  Utilities      -142.60
22 Aug  Card payment   -310.45
Closing balance      3,207.17""",
    """INVOICE 2026-118
Bill to: Northside Dental
Description            Qty    Amount
Composite filling       1     240.00
X-ray bitewing          2      90.00
Total due                    330.00
Payment terms: net 30
Please reference the invoice number.""",
    """PATIENT VISIT SUMMARY
Name: J. Alvarez
Date of service: 09/12/2026
Blood pressure 128/82
Heart rate 74 bpm
Assessment: routine follow-up, no changes.
Return in six months.""",
]

HARD_NEGATIVES_TEST = [
    """THE CORNER DELI
Sandwich board
Pastrami on rye      12.50
Half sour pickle       2.00
House soda             3.25
Counter service only, cash or card.""",
    """MONTHLY CARD SUMMARY
Statement date 30 September 2026
Previous balance       980.10
Payments received     -400.00
New purchases          212.88
Fees                     0.00
New balance            792.98
Minimum due             35.00""",
    """WORK ORDER 88-2214
Site: Warehouse B
Technician: R. Okafor
Replaced belt drive assembly.
Labour 3.5 h x 62.00 = 217.00
Parts                 148.40
Total                 365.40
Signed on completion.""",
    """BOARD MEETING MINUTES
Present: Chair, Treasurer, Secretary.
1. Minutes of the previous meeting approved.
2. Budget revision tabled to October.
3. The committee will report back on parking.
Meeting adjourned 19:40.""",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")[:40]


def _bucket(name: str, seed: int = 20261024) -> int:
    value = int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16) % 100
    return 0 if value < 70 else (1 if value < 85 else 2)


# --------------------------------------------------------------------------- #
# Real documents
# --------------------------------------------------------------------------- #

def syllabus_documents() -> list[dict[str, Any]]:
    from .field_corpus import clean_html

    documents = []
    for record in _read_jsonl(SYLLABI):
        text, _ = clean_html(record.get("text") or "")
        if len(text) < 800:
            continue
        documents.append({"text": text, "label": "syllabus",
                          "institution": record.get("school") or "unknown",
                          "source": "syllabus-cleaned"})
    return documents


from .field_corpus import clean_html  # noqa: E402  (used by syllabus_documents too)


def _catalog_pages(rows: list[dict[str, Any]], per_page: int = 5,
                   seed: int = 20261024) -> list[dict[str, Any]]:
    """A catalog page: real entries from one school, printed one after another."""
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("group_id", "")), []).append(row)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for group, entries in grouped.items():
        source = entries[0].get("source", "unknown")
        by_source.setdefault(source, []).append({"group": group, "entries": entries})
    pages = []
    for source, courses in by_source.items():
        rng.shuffle(courses)
        for index in range(0, len(courses) - per_page, per_page):
            chunk = courses[index:index + per_page]
            lines = []
            for course in chunk:
                values = {}
                for entry in course["entries"]:
                    values.setdefault(entry["label"], str(entry["text"]).strip())
                code = values.get("course_code", "")
                title = values.get("course_title", "")
                if not title:
                    continue
                head = f"{code} {title}".strip()
                body = values.get("course_description", "")
                lines.append(head + (f"\n{body}" if body else "")
                             + (f"\nCredits: {values['credits']}" if "credits" in values else ""))
            if len(lines) >= 3:
                pages.append({"text": "\n\n".join(lines), "label": "course_catalog",
                              "institution": source, "source": "catalog-composed"})
    return pages


def _schedule_pages(rows: list[dict[str, Any]], seed: int = 20261024) -> list[dict[str, Any]]:
    """A timetable: real meeting rows and real course names, one row per meeting."""
    rng = random.Random(seed)
    meetings = [row for row in rows if row.get("label") == "meeting_time"]
    titles = [row for row in rows if row.get("label") == "course_title"]
    rooms = [row for row in rows if row.get("label") == "class_meeting_room"]
    if not meetings or not titles:
        return []
    pages = []
    for index in range(0, min(len(meetings), 2400) - 6, 6):
        lines = ["Weekly Schedule", ""]
        for offset in range(6):
            meeting = meetings[index + offset]
            title = rng.choice(titles)
            room = rng.choice(rooms)["text"] if rooms else ""
            lines.append(f"{meeting['text']}  {title['text']}  {room}".strip())
        pages.append({"text": "\n".join(lines), "label": "schedule",
                      "institution": meetings[index].get("source", "unknown"),
                      "source": "schedule-composed"})
    return pages


def news_pages(limit: int = 6000, per_page: int = 4, seed: int = 20261024) -> list[dict[str, Any]]:
    """Real non-academic prose, several stories to a page."""
    import pyarrow.parquet as pq

    if not NEWS.exists():
        return []
    table = pq.read_table(NEWS, columns=["text"]).column("text").to_pylist()
    rng = random.Random(seed)
    rng.shuffle(table)
    pages = []
    for index in range(0, min(len(table), limit) - per_page, per_page):
        chunk = [text.replace("\\", " ").strip() for text in table[index:index + per_page]]
        pages.append({"text": "\n\n".join(chunk), "label": "other",
                      "institution": "ag_news", "source": "ag_news"})
    return pages


def build_dataset(seed: int = 20261024) -> list[dict[str, Any]]:
    rows = _read_jsonl(CORPUS)
    documents = syllabus_documents()
    documents += _catalog_pages(rows, seed=seed)
    documents += _schedule_pages(rows, seed=seed)
    documents += news_pages(seed=seed)
    for text in HARD_NEGATIVES_TRAIN:
        documents.append({"text": text, "label": "other",
                          "institution": "hard-negative", "source": "hard-negative-train"})
    for text in HARD_NEGATIVES_TEST:
        documents.append({"text": text, "label": "other",
                          "institution": "hard-negative-test", "source": "hard-negative-test"})
    for document in documents:
        document["text"] = re.sub(r"\n{3,}", "\n\n", document["text"]).strip()
    return [document for document in documents if len(document["text"]) >= 120]


def split_documents(documents: list[dict[str, Any]], seed: int = 20261024
                    ) -> dict[str, list[dict[str, Any]]]:
    """Split by institution, stratified so every class is present in each split.

    Hashing institutions without stratifying emptied the validation set of
    syllabi, which makes the threshold meaningless. Here whole institutions are
    still held out -- that is the point -- but which ones is chosen so each
    class keeps a usable share.
    """
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    institutions: dict[str, list[str]] = {}
    for document in documents:
        if document["label"] != "other":
            institutions.setdefault(document["label"], [])
            if document["institution"] not in institutions[document["label"]]:
                institutions[document["label"]].append(document["institution"])
    assignment: dict[tuple[str, str], str] = {}
    for label, names in institutions.items():
        names = sorted(names)
        for index, name in enumerate(names):
            if len(names) == 1:
                position = "train"
            elif index % 5 == 0:
                position = "test"
            elif index % 7 == 0:
                position = "validation"
            else:
                position = "train"
            assignment[(label, name)] = position
    for document in documents:
        if document["source"] == "hard-negative-test":
            splits["test"].append(document)
        elif document["source"] == "hard-negative-train":
            splits["train"].append(document)
        elif document["label"] == "other":
            position = _bucket(
                f"news:{hashlib.sha256(document['text'][:200].encode()).hexdigest()}", seed)
            splits[["train", "validation", "test"][position]].append(document)
        else:
            splits[assignment[(document["label"], document["institution"])]].append(document)
    return splits


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class DocumentTypeClassifier:
    """Word and character TF-IDF over a whole document, then a linear decision.

    A page is long; a bag of n-grams is a strong, tiny, fast representation for
    telling a syllabus from a bank statement. A transformer is not obviously
    better here and is 40x the size, so the baseline that wins is the one
    shipped (see the measured comparison in the model card).
    """

    def __init__(self, pipeline=None, threshold: float = 0.0):
        self.pipeline = pipeline
        self.threshold = float(threshold)

    def probabilities(self, texts: Iterable[str]) -> list[dict[str, float]]:
        matrix = self.pipeline.predict_proba([_prepare(text) for text in texts])
        classes = list(self.pipeline.classes_)
        return [dict(zip(classes, row)) for row in matrix]

    def classify(self, text: str) -> dict[str, Any]:
        probabilities = self.probabilities([text])[0]
        ranked = sorted(((str(name), float(value)) for name, value in probabilities.items()),
                        key=lambda item: item[1], reverse=True)
        label, confidence = ranked[0]
        academic = label != "other" and confidence >= self.threshold
        return {
            "label": label if academic else ("other" if label == "other" else "unknown"),
            "proposed_label": label,
            "confidence": round(float(confidence), 4),
            "academic": bool(academic),
            "abstained": not academic,
            "top": [{"label": name, "probability": round(value, 4)}
                    for name, value in ranked[:3]],
        }

    def save(self, directory: str | Path) -> None:
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold},
                    directory / "doc_type.joblib")

    @classmethod
    def load(cls, directory: str | Path = DEFAULT_OUTPUT) -> "DocumentTypeClassifier":
        import joblib

        payload = joblib.load(Path(directory) / "doc_type.joblib")
        return cls(pipeline=payload["pipeline"], threshold=payload["threshold"])


def _prepare(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _metrics(truth: list[str], predictions: list[str]) -> dict[str, Any]:
    from sklearn.metrics import classification_report, f1_score

    present = sorted(set(truth))
    report = classification_report(truth, predictions, labels=present,
                                   output_dict=True, zero_division=0)
    return {
        "rows": len(truth),
        "accuracy": round(float(sum(t == p for t, p in zip(truth, predictions)) / len(truth)), 4),
        "macro_f1": round(float(f1_score(truth, predictions, average="macro", zero_division=0)), 4),
        "per_class": {label: {key: round(float(value), 4) for key, value in values.items()
                              if key in ("precision", "recall", "f1-score", "support")}
                      for label, values in report.items()
                      if isinstance(values, dict)},
    }


def _calibration(probabilities: list[float], correct: list[bool], bins: int = 10) -> dict[str, Any]:
    ece = 0.0
    reliability = []
    for position in range(bins):
        low, high = position / bins, (position + 1) / bins
        inside = [index for index, value in enumerate(probabilities)
                  if low <= value < high or (position == bins - 1 and value == high)]
        if not inside:
            reliability.append({"bin": f"{low:.1f}-{high:.1f}", "rows": 0})
            continue
        mean_confidence = sum(probabilities[index] for index in inside) / len(inside)
        accuracy = sum(correct[index] for index in inside) / len(inside)
        ece += len(inside) / len(probabilities) * abs(mean_confidence - accuracy)
        reliability.append({"bin": f"{low:.1f}-{high:.1f}", "rows": len(inside),
                            "confidence": round(mean_confidence, 4),
                            "accuracy": round(accuracy, 4)})
    return {"ece_10bin": round(ece, 4), "reliability": reliability}


def train(documents: list[dict[str, Any]], output: Path = DEFAULT_OUTPUT,
          seed: int = 20261024) -> dict[str, Any]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    splits = split_documents(documents, seed)
    train_rows, validation_rows, test_rows = splits["train"], splits["validation"], splits["test"]
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000,
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=60000, sublinear_tf=True)),
    ])
    pipeline = Pipeline([
        ("features", features),
        ("model", CalibratedClassifierCV(
            LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0),
            method="sigmoid", cv=3)),
    ])
    pipeline.fit([_prepare(row["text"]) for row in train_rows],
                 [row["label"] for row in train_rows])
    classifier = DocumentTypeClassifier(pipeline=pipeline)

    validation_texts = [_prepare(row["text"]) for row in validation_rows]
    validation_truth = [row["label"] for row in validation_rows]
    validation_probabilities = classifier.probabilities(validation_texts)
    validation_predictions = [max(row, key=row.get) for row in validation_probabilities]

    # The threshold is the smallest one at which the false-academic rate on
    # validation is at most 1%, i.e. at most 1% of non-academic pages are called academic.
    academic_labels = {"syllabus", "course_catalog", "schedule"}
    chosen, sweep = 0.0, []
    for threshold in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        false_academic = sum(1 for truth, row in zip(validation_truth, validation_probabilities)
                             if truth == "other"
                             and max(row, key=row.get) in academic_labels
                             and row[max(row, key=row.get)] >= threshold)
        negatives = sum(1 for truth in validation_truth if truth == "other")
        rate = false_academic / max(1, negatives)
        sweep.append({"threshold": threshold, "false_academic_rate": round(rate, 4),
                      "false_academic": false_academic, "negatives": negatives})
        if rate <= 0.01:
            chosen = threshold
            break
    classifier.threshold = chosen

    def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [_prepare(row["text"]) for row in rows]
        truth = [row["label"] for row in rows]
        probabilities = classifier.probabilities(texts)
        predictions = [classifier.classify(text)["label"] for text in texts]
        top = [max(row.values()) for row in probabilities]
        top_label = [max(row, key=row.get) for row in probabilities]
        correct = [label == guess for label, guess in zip(truth, top_label)]
        academic = {"syllabus", "course_catalog", "schedule"}
        false_academic = sum(1 for label, guess, probability in zip(truth, top_label, top)
                             if label == "other" and guess in academic
                             and probability >= classifier.threshold)
        negatives = sum(1 for label in truth if label == "other")
        return {
            **_metrics(truth, predictions),
            "false_academic": false_academic,
            "false_academic_rate": round(false_academic / max(1, negatives), 4),
            "negatives": negatives,
            "abstentions": sum(1 for value in predictions if value in ("unknown", "other")),
            **_calibration(top, correct),
        }

    test = evaluate(test_rows)
    by_institution: dict[str, Any] = {}
    for row in test_rows:
        if row["label"] == "other":
            continue
        by_institution.setdefault(row["institution"], []).append(row)
    institution = {name: evaluate(items)["accuracy"]
                   for name, items in by_institution.items() if len(items) >= 2}

    result = {
        "task": "academic document type + academic/non-academic gate",
        "labels": LABELS,
        "not_trained": {
            "transcript": "two real transcripts exist; the deterministic parser covers them",
            "gradebook": "no real public gradebook text; synthetic-only would repeat the old mistake",
            "report_card": "same as gradebook",
            "graduation_requirements": "no real documents in the corpus",
        },
        "rows": {name: len(items) for name, items in splits.items()},
        "by_label": {name: dict(Counter(row["label"] for row in items))
                     for name, items in splits.items()},
        "threshold": chosen,
        "threshold_sweep": sweep,
        "validation": evaluate(validation_rows),
        "test": test,
        "test_by_institution": institution,
        "hard_negative_test": evaluate([row for row in test_rows
                                        if row["source"] == "hard-negative-test"]),
        "known_limitations": [
            "catalog and schedule pages are composed from real rows; only the layout is new",
            "hard negatives are written, not scraped; they are a hard-case probe, not a sample",
            "syllabus institutions in test are held out, but the three other syllabus schools "
            "in train come from the same corpus",
        ],
        "seed": seed,
    }
    classifier.save(output)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Academic document type classifier")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--dataset", default=None, help="Write the dataset as JSONL")
    args = parser.parse_args(argv)

    documents = build_dataset(args.seed)
    if args.dataset:
        target = Path(args.dataset)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for document in documents:
                handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    print(json.dumps({"documents": len(documents),
                      "by_label": dict(Counter(d["label"] for d in documents))}, indent=2))
    print(json.dumps(train(documents, Path(args.output), args.seed)["test"], indent=2))


if __name__ == "__main__":
    main()
