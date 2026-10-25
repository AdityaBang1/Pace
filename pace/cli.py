"""Pace command-line interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import generate_examples, save_jsonl, split_by_group
from .extract import extract_candidates, validate_academic_ir
from .schedule_extract import extract_schedule_candidates


def _read_input(path: str, parser: argparse.ArgumentParser):
    """Read a text file, PDF, image or stdin, reporting problems as CLI errors."""
    from .ingest import IngestError, read_document

    try:
        return read_document(path)
    except IngestError as exc:
        parser.error(str(exc))


def _read_input_text(path: str, parser: argparse.ArgumentParser) -> str:
    """The text of any supported input, so every path-taking command agrees."""
    return _read_input(path, parser).text


def classify_engine(checkpoint, device=None):
    """Load whichever field checkpoint the caller pointed at.

    A Jev checkpoint stores the option descriptions it was trained on; an
    encoder classifier stores ``num_classes``. The two expose the same
    ``classify``/``classify_many`` surface, so callers do not care which one the
    path names.
    """
    from pathlib import Path

    path = Path(checkpoint)
    config = path / "config.json" if path.is_dir() else path.parent / "config.json"
    is_jev = False
    if config.exists():
        try:
            is_jev = "descriptions" in json.loads(config.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            is_jev = False
    if is_jev:
        from .jev_model import JevEngine

        return JevEngine(checkpoint, device=device or "cpu")
    from .field_model import FieldEngine

    return FieldEngine(checkpoint, device=device or "cpu")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pace", description="Local academic document understanding")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-data", help="Generate deterministic synthetic research data")
    generate.add_argument("--count", type=int, default=6000)
    generate.add_argument("--seed", type=int, default=20261019)
    generate.add_argument("--output", default="data/processed")
    analyze = sub.add_parser("analyze", help="Analyze a local document with a trained model")
    analyze.add_argument("path")
    analyze.add_argument("--checkpoint", default="artifacts/pace_classifier.pt")
    analyze.add_argument("--synthetic-pytorch", action="store_true", help="Use the synthetic-only PyTorch multitask model (not validated on real documents)")
    extract = sub.add_parser("extract", help="Extract source-grounded candidate spans without a model")
    extract.add_argument("path")
    transcript = sub.add_parser(
        "parse-transcript",
        help="Parse an academic transcript into source-grounded rows, sections and totals (no model)",
    )
    transcript.add_argument("path", help="A .txt transcript, a .pdf or an image to read "
                                           "directly, or '-' to read text from stdin")
    transcript.add_argument("--summary", action="store_true", help="Print only the compact summary")
    transcript.add_argument("--ocr", choices=("auto", "always", "never"), default="auto",
                         help="For pages with no text layer: read them from pixels (auto), "
                              "read every page from pixels (always), or refuse (never)")
    ingest = sub.add_parser(
        "ingest",
        help="Show how a PDF, image or text file was read, page by page (no parsing)",
    )
    ingest.add_argument("path")
    ingest.add_argument("--ocr", choices=("auto", "always", "never"), default="auto")
    ingest.add_argument("--text", action="store_true", help="Print the extracted text too")
    schedule = sub.add_parser(
        "scan-schedule",
        help="Extract plain-text dated events and recurring weekday/time candidates (uncalibrated)",
    )
    schedule.add_argument("path")
    catalog = sub.add_parser(
        "analyze-course",
        help="Classify natural course cells, parse attributes, and extract course relations",
    )
    catalog.add_argument("cells", nargs="+", help="One or more course field strings to analyze")
    catalog.add_argument("--self-course", default=None, help="Optional course ID (e.g. CSCI 270)")
    catalog.add_argument("--checkpoint", default="artifacts/usc-miniLM/model.pt")
    catalog.add_argument("--onnx", default=None, help="Optional ONNX model path")
    fields = sub.add_parser(
        "classify-field",
        help="Classify a text snippet into an academic field type with calibrated percentages",
    )
    fields.add_argument("text", nargs="?", help="Text to classify (omit to read from --file)")
    fields.add_argument("--file", default=None, help="Read the text to classify from a UTF-8 file")
    fields.add_argument("--checkpoint", default="artifacts/field-classifier-minilm")
    fields.add_argument("--top-k", type=int, default=5)
    fields.add_argument("--device", default=None, help="cuda, cpu, or an explicit torch device")
    fields.add_argument("--no-shapes", action="store_true",
                        help="Skip the deterministic value-shape layer and ask the model "
                             "about bare values such as 'Room 303' or 'Wed 2-4pm' as well")
    extract_fields = sub.add_parser(
        "extract-fields",
        help="Read any course document's text and keep only the fields that matter "
             "(course, meetings, grading, policies), each with its evidence span",
    )
    extract_fields.add_argument("path", help="UTF-8 text file, or - for stdin")
    extract_fields.add_argument("--checkpoint", default="artifacts/field-classifier-minilm")
    extract_fields.add_argument("--context-checkpoint", default="artifacts/field-classifier-context")
    extract_fields.add_argument("--threshold", type=float, default=0.8)
    extract_fields.add_argument("--min-confidence", type=float, default=None)
    extract_fields.add_argument("--device", default=None)
    understand = sub.add_parser(
        "understand",
        help="Read an academic document into the stable application record "
             "(schema pace.academic-ir/0.2)",
    )
    understand.add_argument("path", help="UTF-8 text file, or - for stdin")
    understand.add_argument("--checkpoint", default=None,
                            help="Defaults to the reader the app should use "
                                 "(artifacts/field-classifier-minilm); the best "
                                 "whole-document reader is artifacts/jev-field-v4")
    understand.add_argument("--device", default=None)
    understand.add_argument("--min-confidence", type=float, default=0.0)
    section = sub.add_parser(
        "classify-section",
        help="Classify a document section from its heading and the text under it",
    )
    section.add_argument("--heading", required=True, help="The section heading, as written")
    section.add_argument("--value", default="", help="The text under the heading")
    section.add_argument("--checkpoint", default="artifacts/field-classifier-minilm",
                         help="Heading classifier (a model trained on bare values)")
    section.add_argument("--context-checkpoint", default="artifacts/field-classifier-context",
                         help="Heading+value classifier used when the heading is unfamiliar")
    section.add_argument("--threshold", type=float, default=0.8,
                         help="Heading confidence below which the context model takes over")
    section.add_argument("--device", default=None, help="cuda, cpu, or an explicit torch device")
    args = parser.parse_args(argv)
    if args.command == "generate-data":
        rows = generate_examples(args.count, args.seed)
        splits = split_by_group(rows, args.seed)
        for name, part in splits.items():
            save_jsonl(Path(args.output) / f"{name}.jsonl", part)
        print(json.dumps({"count": len(rows), "splits": {k: len(v) for k, v in splits.items()}}, indent=2))
    elif args.command == "extract":
        text = _read_input_text(args.path, parser)
        print(json.dumps(extract_candidates(text), indent=2, ensure_ascii=False))
    elif args.command == "scan-schedule":
        text = _read_input_text(args.path, parser)
        print(json.dumps(extract_schedule_candidates(text), indent=2, ensure_ascii=False))
    elif args.command == "parse-transcript":
        from .ingest import IngestError, parse_document, read_document, summarize_source
        from .ocr_verify import verification_summary
        from .transcript_extract import summarize

        try:
            document = read_document(args.path, ocr=args.ocr)
        except IngestError as exc:
            parser.error(str(exc))
        parsed = parse_document(document)
        if args.summary:
            summary = summarize(parsed)
            source = summarize_source(parsed)
            if source is not None:
                summary["source"] = source
            verification = verification_summary(parsed)
            if verification is not None:
                summary["verification"] = verification
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
    elif args.command == "ingest":
        from .ingest import IngestError, read_document

        try:
            document = read_document(args.path, ocr=args.ocr)
        except IngestError as exc:
            parser.error(str(exc))
        result = document.summary()
        if args.text:
            result["pages"] = [dict(page.summary(), text=page.text) for page in document.pages]
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "analyze":
        document = _read_input(args.path, parser)
        text = document.text
        if args.synthetic_pytorch:
            from .synthetic_doc_model import SyntheticDocumentEngine

            # The model proposes the document type and entity spans; the rule-based
            # parser supplies the exact transcript row structure. Both stay proposals
            # that need a human to confirm.
            result = SyntheticDocumentEngine(args.checkpoint).analyze(text)
            transcript = result.get("transcript")
            if transcript is not None:
                result["transcript_proposal"] = transcript["parsed"]
                result["transcript_summary"] = transcript["summary"]
        else:
            from .infer import PaceEngine

            engine = PaceEngine(args.checkpoint)
            result = engine.analyze_document(text)
            result["candidate_entities"] = extract_candidates(text)
            errors = validate_academic_ir(result)
            result["validation_errors"] = errors
        # A model reading OCR text should say so, exactly like the transcript path.
        result["source"] = document.summary()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "analyze-course":
        from .real_infer import RealCatalogEngine

        engine = RealCatalogEngine(args.checkpoint, onnx_path=args.onnx)
        print(json.dumps(engine.analyze_course_cells(args.cells, self_course=args.self_course), indent=2))
    elif args.command == "classify-field":
        import torch

        from .field_model import FieldEngine

        if args.file:
            text = _read_input_text(args.file, parser)
        elif args.text:
            text = args.text
        else:
            parser.error("provide the text to classify or --file")
        from .field_shapes import resolve as resolve_shape

        # A bare "Room 303" or "Wed 2-4pm" is decided by its shape; the model is
        # measurably wrong on those and is only needed once there is something
        # to interpret. See pace/field_shapes.py for the measurements.
        shape = None if args.no_shapes else resolve_shape(text)
        if shape is not None:
            result = {"label": shape["label"], "task": shape["task"],
                      "confidence": 0.99, "percent": 99.0, "top": [],
                      "source": shape["source"], "reason": shape["reason"],
                      "model": "shape"}
        else:
            device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            engine = classify_engine(args.checkpoint, device=device)
            result = engine.classify(text, top_k=args.top_k)
        result["text"] = text
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "extract-fields":
        from .field_extract import DEFAULT_MIN_CONFIDENCE, main as extract_main

        forwarded = [args.path, "--checkpoint", args.checkpoint,
                     "--threshold", str(args.threshold),
                     "--min-confidence", str(args.min_confidence
                                            if args.min_confidence is not None
                                            else DEFAULT_MIN_CONFIDENCE)]
        if args.context_checkpoint:
            forwarded += ["--context-checkpoint", args.context_checkpoint]
        if args.device:
            forwarded += ["--device", args.device]
        extract_main(forwarded)
    elif args.command == "understand":
        from .api import DEFAULT_CHECKPOINT, main as understand_main

        forwarded = [args.path, "--checkpoint", args.checkpoint or DEFAULT_CHECKPOINT,
                     "--min-confidence", str(args.min_confidence)]
        if args.device:
            forwarded += ["--device", args.device]
        understand_main(forwarded)
    elif args.command == "classify-section":
        import torch

        from pathlib import Path

        from .field_model import FieldEngine, classify_section

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        engine = FieldEngine(args.checkpoint, device=device)
        context_engine = None
        if args.context_checkpoint and Path(args.context_checkpoint).exists():
            context_engine = FieldEngine(args.context_checkpoint, device=device)
        elif args.value:
            parser.error(f"context checkpoint {args.context_checkpoint!r} is not present")
        result = classify_section(args.heading, args.value, engine, context_engine,
                                  threshold=args.threshold)
        result["heading_text"] = args.heading
        result["value_text"] = args.value
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
