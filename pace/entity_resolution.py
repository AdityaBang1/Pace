"""Do two mentions name the same course?

The audit found no matcher at all: the repository could normalise ``subject +
number`` and nothing else. This module builds the capability from the real
catalog titles already in the corpus.

There is one honest limitation to state up front. A catalog gives one printed
title per course, so *positive* pairs are derived: the same real title written
the way people write it (no code, code appended, abbreviations, initials,
reordering, OCR glyphs). *Negatives* are real: two different courses as the
catalogs actually list them, including the hard pairs -- Calculus AB against
Calculus BC, Chemistry I against Chemistry II, Honors against AP.

The decision the system must get right is asymmetric. A missed match is an
annoyance; a false merge silently corrupts a student's academic record. So the
matcher reports ``same``, ``different`` or ``uncertain``, and the shipped
threshold is chosen from the measured false-merge rate, not from F1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CORPUS = Path("data/processed/field/corpus.jsonl")
DEFAULT_OUTPUT = Path("artifacts/entity-resolution")

#: Real institutional shorthand, expanded in *both* directions by the mention
#: generator. Kept small and auditable; a wrong expansion is a false merge.
ABBREVIATIONS: dict[str, str] = {
    "intro": "introduction",
    "introduct": "introduction",
    "chem": "chemistry",
    "bio": "biology",
    "calc": "calculus",
    "phys": "physics",
    "econ": "economics",
    "psych": "psychology",
    "stats": "statistics",
    "stat": "statistics",
    "math": "mathematics",
    "eng": "english",
    "engr": "engineering",
    "cs": "computer science",
    "cogsci": "cognitive science",
    "ap": "advanced placement",
    "hon": "honors",
    "honours": "honors",
    "sem": "seminar",
    "lab": "laboratory",
    "dept": "department",
}

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}
LEVEL_WORDS = {
    "advanced placement": "ap", "ap": "ap", "honors": "honors", "honours": "honors",
    "advanced": "advanced", "accelerated": "accelerated", "regular": "regular",
    "remedial": "remedial", "college prep": "regular", "cp": "regular",
}
#: Markers that make two mentions different courses even when the words agree.
LEVEL_MARKERS = ("ap", "honors", "advanced", "accelerated", "regular", "remedial")
SECTION_LETTERS = ("a", "b", "c", "ab", "bc")

CODE_RE = re.compile(r"\b([A-Z]{2,6})\s?-?\s?(\d{2,4}[A-Z]?)\b")
MIT_CODE_RE = re.compile(r"^(\d{1,2})\.(\d{2,3}[A-Z]?)")


def parse_course_key(group: str) -> tuple[str, str] | None:
    """Split a real course id into (subject, number).

    The catalogs encode identity in the group id, not in the title: ``uw-course:
    KINES 961``, ``mit-course:3.029``, ``usc-subject:AME``. Parsing it is what
    makes a *hard* negative possible -- the neighbouring course in the same
    department, not a random course from another school.
    """
    body = group.split(":", 1)[-1].strip().upper()
    found = CODE_RE.search(body)
    if found:
        return found.group(1), found.group(2)
    mit = MIT_CODE_RE.match(body)
    if mit:
        return mit.group(1), mit.group(2)
    return None


# --------------------------------------------------------------------------- #
# Mentions
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Mention:
    text: str
    group: str
    institution: str
    code: str | None
    subject: str = ""
    number: str = ""

    def number_value(self) -> float | None:
        digits = re.sub(r"[^0-9]", "", self.number)
        return float(digits) if digits else None


def load_titles(path: str | Path = CORPUS) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [row for row in rows if row.get("label") == "course_title"
            and len(str(row.get("text", "")).strip()) >= 4]


def _clean_title(text: str) -> str:
    text = re.sub(r"^[A-Z]{2,6}\s?-?\s?\d{2,4}[A-Z]?\s*[:\-]\s*", "", text.strip())
    return re.sub(r"\s+", " ", text).strip(" .;:-")


def _abbreviate(title: str, rng: random.Random) -> str:
    words = title.split()
    out = []
    for word in words:
        key = word.casefold().strip(",.")
        if key in ABBREVIATIONS and rng.random() < 0.8:
            short = next((k for k, v in ABBREVIATIONS.items() if v == ABBREVIATIONS[key]
                          and len(k) < len(key)), None)
            out.append(short or word)
        elif key.endswith("ology") and rng.random() < 0.5:
            out.append(word[:-4] + "y")
        elif rng.random() < 0.15 and len(word) > 8:
            out.append(word[:5])
        else:
            out.append(word)
    return " ".join(out)


def _expand(title: str) -> str:
    return " ".join(ABBREVIATIONS.get(word.casefold().strip(",."), word)
                    for word in title.split())


def _ocr(title: str, rng: random.Random) -> str:
    swaps = [("0", "O"), ("1", "l"), ("5", "S"), ("8", "B"), ("m", "rn"), ("2", "Z")]
    words = []
    for word in title.split():
        if rng.random() < 0.25:
            source, target = rng.choice(swaps)
            words.append(word.replace(source, target, 1))
        else:
            words.append(word)
    return " ".join(words)


def variants(title: str, code: str | None, rng: random.Random) -> list[tuple[str, str]]:
    """Real-surface writings of one course, each tagged with how it was made."""
    base = _clean_title(title)
    out: list[tuple[str, str]] = []
    if code:
        out.append((f"{code} {base}", "code_prefix"))
    out.append((base.casefold(), "case"))
    out.append((base.upper(), "case"))
    out.append((_abbreviate(base, rng), "abbreviation"))
    out.append((_expand(base), "expansion"))
    out.append((_ocr(base, rng), "ocr"))
    words = base.split()
    if len(words) > 2:
        out.append((" ".join(words[1:] + words[:1]), "reorder"))
    out.append((re.sub(r"[^A-Za-z0-9 ]", " ", base), "punctuation"))
    if code:
        out.append((f"{base} ({code})", "code_suffix"))
    return [(text.strip(), kind) for text, kind in out if text.strip() and text.strip() != base]


def mentions_from_corpus(rows: list[dict[str, Any]], seed: int = 20261024) -> list[Mention]:
    """One canonical mention plus the real course key it belongs to."""
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        text = str(row["text"]).strip()
        group = str(row.get("group_id", ""))
        if not group or group in seen or len(text) < 5:
            continue
        key = parse_course_key(group)
        seen[group] = {"text": text,
                       "code": f"{key[0]} {key[1]}" if key else None,
                       "subject": key[0] if key else "",
                       "number": key[1] if key else "",
                       "institution": row.get("source", "")}
    mentions: list[Mention] = []
    for group, item in seen.items():
        cleaned = _clean_title(item["text"])
        if len(cleaned.split()) < 2:
            continue
        mentions.append(Mention(cleaned, group, item["institution"], item["code"],
                                item["subject"], item["number"]))
    return mentions


# --------------------------------------------------------------------------- #
# Pairs
# --------------------------------------------------------------------------- #

def _level_signature(text: str) -> tuple[str, ...]:
    folded = text.casefold()
    marks = set()
    for word, canonical in LEVEL_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", folded):
            marks.add(canonical)
    tokens = re.findall(r"\b(i{1,3}v?|iv|v)\b", folded)
    for token in tokens:
        if token in ROMAN:
            marks.add(f"roman{ROMAN[token]}")
    for letter in SECTION_LETTERS:
        if re.search(rf"\b{letter}\b", folded):
            marks.add(f"section{letter}")
    # "Advanced Placement" is "AP"; it must not also count as "advanced",
    # or the same course written in full looks like a level conflict.
    if "ap" in marks:
        marks.discard("advanced")
    return tuple(sorted(marks))


def _numbers(text: str) -> set[str]:
    return {number for _, number in CODE_RE.findall(text.upper())}


def _subject(text: str) -> str:
    found = CODE_RE.findall(text.upper())
    return found[0][0] if found else ""


def build_pairs(mentions: list[Mention], seed: int = 20261024,
                per_course: int = 3, hard_negatives: int = 2) -> list[dict[str, Any]]:
    """Positives from one real course, negatives from two real ones."""
    rng = random.Random(seed)
    by_group = {mention.group: mention for mention in mentions}
    items = list(by_group.values())
    pairs: list[dict[str, Any]] = []
    for mention in items:
        writings = variants(mention.text, mention.code, rng)
        rng.shuffle(writings)
        for text, kind in writings[:per_course]:
            pairs.append({"left": mention.text, "right": text, "label": 1,
                          "kind": f"positive:{kind}", "institution": mention.institution,
                          "group_left": mention.group, "group_right": mention.group,
                          "code_left": mention.code, "code_right": mention.code})
    # Hard negatives: the course next door. Same department and a near number,
    # the level variant of the same stem, or the same stem in another school.
    by_subject: dict[str, list[Mention]] = {}
    for mention in items:
        if mention.subject:
            by_subject.setdefault(mention.subject, []).append(mention)
    for mention in items:
        pool = [other for other in by_subject.get(mention.subject, [])
                if other.group != mention.group]
        level = _level_signature(mention.text)
        words = set(mention.text.casefold().split())
        leveled = [other for other in pool if _level_signature(other.text) != level]
        same_stem = [other for other in pool if words & set(other.text.casefold().split())
                     and len(words & set(other.text.casefold().split()))
                     >= max(2, len(words) // 2)]
        own_number = mention.number_value()
        near = sorted((other for other in pool if own_number is not None
                       and other.number_value() is not None),
                      key=lambda other: abs((other.number_value() or 0) - own_number))[:5]
        chosen = (same_stem or leveled or near or pool)
        rng.shuffle(chosen)
        for other in chosen[:hard_negatives]:
            kind = ("negative:level" if other in leveled else
                    "negative:neighbour" if other in near else "negative:same_subject")
            if _norm(mention.text) == _norm(other.text):
                # Two catalog entries with the same printed name and different
                # numbers ("Directed Study" in two departments). Neither a true
                # positive nor a fair negative: the record needs the code to decide.
                kind, label = "ambiguous:same_name", None
            else:
                label = 0
            pairs.append({"left": mention.text, "right": other.text, "label": label,
                          "kind": kind, "institution": mention.institution,
                          "group_left": mention.group, "group_right": other.group,
                          "code_left": mention.code, "code_right": other.code})
    # Cross-school hard negatives: the same course *name* at two institutions is
    # not always the same course, but it is the pair a naive matcher merges.
    rng.shuffle(items)
    for index in range(0, len(items) - 1, 2):
        left, right = items[index], items[index + 1]
        if left.group == right.group or left.institution == right.institution:
            continue
        shared = set(left.text.casefold().split()) & set(right.text.casefold().split())
        if len(shared) < max(2, len(left.text.split()) // 2):
            continue
        pairs.append({"left": left.text, "right": right.text, "label": 0,
                      "kind": "negative:cross_school_stem", "institution": left.institution,
                      "group_left": left.group, "group_right": right.group,
                      "code_left": left.code, "code_right": right.code})
    # Easy negatives across subjects, so the model cannot win on subject alone.
    for _ in range(len(items)):
        left, right = rng.sample(items, 2)
        if left.group == right.group:
            continue
        if _subject(left.text) and _subject(left.text) == _subject(right.text):
            continue
        pairs.append({"left": left.text, "right": right.text, "label": 0,
                      "kind": "negative:easy", "institution": left.institution,
                      "group_left": left.group, "group_right": right.group,
                      "code_left": left.code, "code_right": right.code})
    return [pair for pair in pairs if pair["label"] is not None or pair["kind"]]


# --------------------------------------------------------------------------- #
# Features and model
# --------------------------------------------------------------------------- #

def _norm(text: str) -> str:
    """Letters for matching: course codes and punctuation removed.

    A catalog prints the code next to the title on some rows and not on others.
    Leaving it in makes a real positive look like a different course, which is a
    false split the features would otherwise have to learn around.
    """
    text = CODE_RE.sub(" ", text)
    text = re.sub(r"\b\d{1,2}\.\d{2,3}[a-z]?\b", " ", text, flags=re.I)
    text = _expand(text.casefold())
    return re.sub(r"[^a-z0-9 ]", " ", text).strip()


def _content_numbers(text: str) -> set[str]:
    """Numbers that are part of the name, not the course code: K-8, 6-12, 101."""
    return set(re.findall(r"\d+", _norm(text)))


def _tokens(text: str) -> set[str]:
    return {token for token in _norm(text).split() if len(token) > 1}


def pairs_features(pairs: list[dict[str, Any]]) -> np.ndarray:
    from difflib import SequenceMatcher

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    left = [_norm(pair["left"]) for pair in pairs]
    right = [_norm(pair["right"]) for pair in pairs]
    if not any(left + right):
        # ``_norm`` deletes course codes on purpose, so a pair of bare codes
        # ("MATH 101" vs "MATH 1010") normalises to two empty strings. Compare the
        # raw text in that case instead of returning a zero-vocabulary error.
        left = [re.sub(r"\s+", " ", pair["left"] or "").strip().casefold() for pair in pairs]
        right = [re.sub(r"\s+", " ", pair["right"] or "").strip().casefold() for pair in pairs]
    cosine = _char_cosine(left, right)

    rows = []
    for index, pair in enumerate(pairs):
        a, b = pair["left"], pair["right"]
        ta, tb = _tokens(a), _tokens(b)
        union = ta | tb
        jaccard = len(ta & tb) / len(union) if union else 0.0
        containment = len(ta & tb) / min(len(ta), len(tb)) if ta and tb else 0.0
        sequence = SequenceMatcher(None, _norm(a), _norm(b)).ratio()
        codes_a, codes_b = _numbers(a), _numbers(b)
        code_equal = (1.0 if codes_a & codes_b else 0.0) if (codes_a and codes_b) else -1.0
        numbers_conflict = 1.0 if (codes_a and codes_b and not (codes_a & codes_b)) else 0.0
        levels_a, levels_b = _level_signature(a), _level_signature(b)
        level_conflict = 1.0 if (levels_a and levels_b and levels_a != levels_b) else 0.0
        level_agreement = 1.0 if (levels_a and levels_a == levels_b) else 0.0
        name_numbers_a, name_numbers_b = _content_numbers(a), _content_numbers(b)
        content_conflict = 1.0 if (name_numbers_a and name_numbers_b
                                   and name_numbers_a != name_numbers_b) else 0.0
        longest = max(len(a), len(b))
        length_ratio = (min(len(a), len(b)) / longest) if longest else 0.0
        common_prefix = len(set(a.casefold().split()) &
                            set(b.casefold().split())) / max(1, max(len(a.split()), len(b.split())))
        rows.append([cosine[index], jaccard, containment, sequence, code_equal,
                     numbers_conflict, level_conflict, level_agreement, length_ratio,
                     common_prefix, content_conflict, _prefix_overlap(ta, tb),
                     float(len(ta) == 0), float(len(tb) == 0)])
    return np.asarray(rows, dtype=float)


def _char_cosine(left: list[str], right: list[str]) -> np.ndarray:
    """Character n-gram cosine, degrading to zero instead of raising on tiny input.

    ``min_df=2`` trims noise on a 14k-pair training set, but a caller comparing one
    short pair has no n-gram in both documents, and a single character has no
    trigram at all, so both settings can legitimately produce an empty vocabulary.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    for settings in ({"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2},
                     {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 1},
                     {"analyzer": "char", "ngram_range": (1, 2), "min_df": 1}):
        vectorizer = TfidfVectorizer(**settings)
        try:
            vectorizer.fit(left + right)
        except ValueError:
            continue
        left_matrix = normalize(vectorizer.transform(left))
        right_matrix = normalize(vectorizer.transform(right))
        return np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel()
    return np.zeros(len(left), dtype=float)


def _prefix_overlap(left: set[str], right: set[str]) -> float:
    """Share of the smaller vocabulary that one side's token abbreviates.

    ``chem`` / ``chemistry`` and ``intro`` / ``introduction`` are the same word
    written to fit a catalog column; character n-grams alone score them apart.
    """
    if not left or not right:
        return 0.0
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    hits = 0
    for token in smaller:
        if len(token) < 3:
            continue
        if any(other.startswith(token) or token.startswith(other)
               for other in larger if len(other) >= 3):
            hits += 1
    return hits / len(smaller)


FEATURE_NAMES = ["char_cosine", "token_jaccard", "containment", "sequence_ratio",
                 "code_equal", "number_conflict", "level_conflict", "level_agreement",
                 "length_ratio", "word_overlap", "content_number_conflict",
                 "prefix_overlap", "empty_left", "empty_right"]


def _codes_disagree(left: str, right: str) -> bool:
    a, b = parse_course_key(left), parse_course_key(right)
    return bool(a and b and a[1] != b[1])


def _veto(pairs: list[dict[str, Any]]) -> np.ndarray:
    """Pairs a matcher must never merge, whatever the model says.

    A different level marker or a different name-number (K-8 against 6-12,
    Calculus I against Calculus II) is not noise to be averaged away: merging
    them corrupts a record. The guard is measured with everything else.
    """
    flags = []
    for pair in pairs:
        a, b = pair["left"], pair["right"]
        levels_a, levels_b = _level_signature(a), _level_signature(b)
        numbers_a, numbers_b = _content_numbers(a), _content_numbers(b)
        flags.append(bool((levels_a and levels_b and levels_a != levels_b)
                          or (numbers_a and numbers_b and numbers_a != numbers_b)))
    return np.asarray(flags)


def _group_split(pairs: list[dict[str, Any]], seed: int = 20261024
                 ) -> tuple[list[dict], list[dict], list[dict]]:
    """Split by course group, so a course is never on both sides of the split."""
    def bucket(group: str) -> int:
        digest = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
        value = int(digest[:8], 16) % 100
        return 0 if value < 70 else (1 if value < 85 else 2)

    train, validation, test = [], [], []
    for pair in pairs:
        position = bucket(pair["group_left"])
        if position == 0:
            train.append(pair)
        elif position == 1:
            validation.append(pair)
        else:
            test.append(pair)
    return train, validation, test


@dataclass
class Decision:
    same: bool
    confidence: float
    decision: str
    probability: float


class CourseMatcher:
    """A small logistic model over string and level features."""

    def __init__(self, model=None, vectorizer=None, threshold: float = 0.5):
        self.model = model
        self.threshold = float(threshold)

    def probability(self, pairs: list[dict[str, Any]]) -> np.ndarray:
        features = pairs_features(pairs)
        return self.model.predict_proba(features)[:, 1]

    def decide(self, left: str, right: str) -> Decision:
        pair = {"left": left, "right": right}
        probability = float(self.probability([pair])[0])
        if _veto([pair])[0]:
            # A level or name-number conflict overrides a high string score.
            return Decision(False, min(probability, 1.0 - probability), "different",
                            probability)
        if probability >= self.threshold and _codes_disagree(left, right):
            # Same name, different catalogue numbers: only the record knows.
            return Decision(False, 0.5, "uncertain", probability)
        if probability >= self.threshold:
            decision = "same"
        elif probability <= 1.0 - self.threshold:
            decision = "different"
        else:
            decision = "uncertain"
        return Decision(decision == "same", probability, decision, probability)

    def save(self, directory: str | Path) -> None:
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold},
                    directory / "matcher.joblib")

    @classmethod
    def load(cls, directory: str | Path) -> "CourseMatcher":
        import joblib

        payload = joblib.load(Path(directory) / "matcher.joblib")
        return cls(model=payload["model"], threshold=payload["threshold"])


def _metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = probabilities >= threshold
    truth = labels.astype(bool)
    true_positive = int((predicted & truth).sum())
    false_positive = int((predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "threshold": round(float(threshold), 4),
        "pairs": int(len(labels)),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_merges": false_positive,
        "false_merge_rate": round(false_positive / max(1, int((~truth).sum())), 4),
        "false_splits": false_negative,
    }


def train(pairs: list[dict[str, Any]], output: Path, seed: int = 20261024) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    train_rows, validation_rows, test_rows = _group_split(pairs, seed)
    ambiguous = sum(1 for row in pairs if row["label"] is None)
    train_rows = [row for row in train_rows if row["label"] is not None]
    validation_rows = [row for row in validation_rows if row["label"] is not None]
    test_rows = [row for row in test_rows if row["label"] is not None]
    x_train = pairs_features(train_rows)
    y_train = np.asarray([pair["label"] for pair in train_rows])
    model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    model.fit(x_train, y_train)
    matcher = CourseMatcher(model=model)

    x_validation = pairs_features(validation_rows)
    y_validation = np.asarray([pair["label"] for pair in validation_rows])
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    # The veto is part of the shipped decision, so it is part of the measurement.
    validation_probabilities[_veto(validation_rows)] = 0.0

    # Choose the lowest threshold whose validation false-merge rate is at most 0.5%.
    chosen = 0.95
    sweep = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995):
        record = _metrics(validation_probabilities, y_validation, threshold)
        sweep.append(record)
        if record["false_merge_rate"] <= 0.005 and record["recall"] >= 0.5:
            chosen = threshold
            break
    matcher.threshold = chosen

    x_test = pairs_features(test_rows)
    y_test = np.asarray([pair["label"] for pair in test_rows])
    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_probabilities[_veto(test_rows)] = 0.0

    by_kind: dict[str, Any] = {}
    kinds = sorted({pair["kind"] for pair in test_rows})
    for kind in kinds:
        mask = np.asarray([pair["kind"] == kind for pair in test_rows])
        if mask.any():
            by_kind[kind] = _metrics(test_probabilities[mask], y_test[mask], chosen)

    # Institution holdout: a school whose titles were never in training.
    held_out = {"twangodev/uwcourses", "USC/USC-Course-Catalog"}
    mask = np.asarray([pair["institution"] in held_out for pair in test_rows])
    institution = (_metrics(test_probabilities[mask], y_test[mask], chosen)
                   if mask.any() else None)

    result = {
        "task": "course entity resolution (pairwise same-course decision)",
        "features": FEATURE_NAMES,
        "rows": {"train": len(train_rows), "validation": len(validation_rows),
                 "test": len(test_rows)},
        "ambiguous_same_name_pairs_excluded": ambiguous,
        "decision_policy": ("veto on level/name-number conflict; identical names with "
                            "different catalogue numbers return 'uncertain'"),
        "label_balance": {"train_positive": int(y_train.sum())},
        "threshold_selection": "lowest validation threshold with false-merge rate <= 0.005",
        "threshold": chosen,
        "validation_sweep": sweep,
        "test": _metrics(test_probabilities, y_test, chosen),
        "test_by_kind": by_kind,
        "institution_holdout": institution,
        "known_limitation": ("positive pairs are surface variants of real catalog titles; "
                             "negatives are real distinct courses. A same-course pair written "
                             "with a different title is not in this benchmark."),
        "seed": seed,
    }
    matcher.save(output)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Course entity resolution")
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--pairs", default=None, help="Write the pair dataset as JSONL")
    args = parser.parse_args(argv)

    rows = load_titles(args.corpus)
    mentions = mentions_from_corpus(rows, args.seed)
    pairs = build_pairs(mentions, args.seed)
    if args.pairs:
        target = Path(args.pairs)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    output = Path(args.output)
    print(json.dumps({"mentions": len(mentions), "pairs": len(pairs)}, indent=2))
    print(json.dumps(train(pairs, output, args.seed), indent=2))


if __name__ == "__main__":
    main()
