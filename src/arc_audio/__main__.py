import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Local desktop audio companion. Default: demo control window.")
    commands = parser.add_subparsers(dest="command")
    ui = commands.add_parser("ui", help="Open control window (demo selected)")
    ui.add_argument("--smoke-seconds", type=float, default=0, help="Auto-start demo then close for a UI smoke check")
    commands.add_parser("devices", help="Enumerate compatible stereo WASAPI loopback devices")
    demo = commands.add_parser("make-demo", help="Explicitly write a synthetic stereo PCM WAV fixture")
    demo.add_argument("output", type=Path)
    demo.add_argument("--seconds", type=float, default=12)
    download = commands.add_parser("download-model", help="Explicit network download: ~346 MB AST weights + configs")
    download.add_argument("--accept-download", action="store_true", required=True)
    evaluate = commands.add_parser("evaluate", help="Offline WAV analysis and optional interval metrics")
    evaluate.add_argument("wav", type=Path)
    evaluate.add_argument("--annotations", type=Path)
    evaluate.add_argument("--backend", choices=["baseline", "ast", "local"], default="baseline")
    evaluate.add_argument("--sensitivity", type=float, default=1.0)
    evaluate.add_argument("--min-confidence", type=float, default=0.6)
    evaluate.add_argument("--tolerance", type=float, default=0.15)
    evaluate.add_argument("--output", type=Path, help="Local detection JSONL path")
    prepare = commands.add_parser("prepare-training", help="Inventory local media; create label manifest without audio exports")
    prepare.add_argument("folder", type=Path, nargs="?", default=Path("training"))
    labels = commands.add_parser("label-training", help="Review gunfire with Yes / No / Unsure questions")
    labels.add_argument("manifest", type=Path, nargs="?", default=Path("training/dataset.json"))
    labels.add_argument("--smoke-seconds", type=float, default=0)
    labels.add_argument("--advanced", action="store_true", help="Open the optional legacy interval editor")
    build = commands.add_parser("build-weapon-dataset", help="Explicitly acquire curated sources and propose weapon shots")
    build.add_argument("csv", type=Path)
    build.add_argument("--dataset", type=Path, default=Path("training/dataset.json"))
    build.add_argument("--storage", type=Path, help="Source cache inside dataset folder; default: sources/")
    build.add_argument("--dry-run", action="store_true", help="Validate/import preview only; no writes or downloads")
    build.add_argument("--max-sources", type=int, help="Process the first N unique URLs")
    build.add_argument("--source-url", help="Only process this exact curated URL")
    build.add_argument("--pre", type=float, default=.35, help="Preview seconds before onset")
    build.add_argument("--post", type=float, default=1., help="Preview seconds after onset")
    build.add_argument("--min-spacing", type=float, default=.07, help="Minimum onset spacing, seconds")
    build.add_argument("--ffmpeg-location", help="Optional FFmpeg executable or folder")
    weapon_status = commands.add_parser("weapon-dataset-status", help="Review progress and source diversity per weapon")
    weapon_status.add_argument("manifest", type=Path, nargs="?", default=Path("training/dataset.json"))
    discovery = commands.add_parser("discover-weapon-sources", help="Search YouTube; save unverified metadata only")
    discovery.add_argument("--weapon", required=True)
    discovery.add_argument("--max-results", type=int, default=10)
    discovery.add_argument("--output", type=Path, default=Path("artifacts/discovered-weapon-sources.json"))
    discovery.add_argument("--dataset", type=Path, help="Optionally register unverified metadata for status reporting only")
    backgrounds = commands.add_parser("suggest-weapon-background", help="Propose background ranges; explicit human review required")
    backgrounds.add_argument("manifest", type=Path, nargs="?", default=Path("training/dataset.json"))
    backgrounds.add_argument("--seconds", type=float, default=2.)
    backgrounds.add_argument("--margin", type=float, default=.75)
    backgrounds.add_argument("--max-per-source", type=int, default=8)
    suggest = commands.add_parser("suggest-labels", help="Offline cached AST proposals; human review required")
    suggest.add_argument("manifest", type=Path)
    suggest.add_argument("--model-dir", type=Path, help="Existing AST cache; never downloads weights")
    suggest.add_argument("--classes", nargs="+", choices=["gunshot", "footsteps"], default=["gunshot"])
    suggest.add_argument("--threshold", type=float, default=.2, help="Uncalibrated proposal cutoff; not accuracy")
    suggest.add_argument("--window-seconds", type=float, default=2.0, help="Short windows are feature-padded")
    suggest.add_argument("--hop-seconds", type=float, default=2.0)
    suggest.add_argument("--allow-unconfirmed-track", action="store_true",
                         help="Provisional proposals only; accepting still requires human track confirmation")
    train = commands.add_parser("train-local", help="Train only from confirmed human-reviewed labels")
    train.add_argument("manifest", type=Path, nargs="?", default=Path("training/dataset.json"))
    train.add_argument("--output", type=Path, help="Default: local application model cache")
    train.add_argument("--classes", nargs="+", help="Selected sound labels, or 'all'. Default: footsteps reload gunshot")
    status = commands.add_parser("training-status", help="Show annotation coverage without decoding audio")
    status.add_argument("manifest", type=Path, nargs="?", default=Path("training/dataset.json"))
    training_eval = commands.add_parser("evaluate-training", help="Event-level evaluation on reviewed held-out media")
    training_eval.add_argument("manifest", type=Path, nargs="?", default=Path("training/dataset.json"))
    training_eval.add_argument("--split", choices=["validation", "test"], default="test")
    training_eval.add_argument("--model", type=Path)
    training_eval.add_argument("--min-confidence", type=float, default=.6)
    training_eval.add_argument("--tolerance", type=float, default=.15)
    training_eval.add_argument("--output", type=Path, default=Path("artifacts/arc-evaluation.json"))
    args = parser.parse_args()
    try:
        if args.command in (None, "ui"):
            from .ui import run_ui
            return run_ui(getattr(args, "smoke_seconds", 0))
        if args.command == "devices":
            from dataclasses import asdict

            from .audio import loopback_devices
            print(json.dumps([asdict(d) for d in loopback_devices()], indent=2))
        elif args.command == "make-demo":
            if not 0 < args.seconds <= 3600:
                raise ValueError("Demo duration must be between 0 and 3600 seconds")
            from .audio import write_demo
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_demo(args.output, args.seconds)
            print(f"Synthetic fixture written: {args.output.resolve()}")
        elif args.command == "download-model":
            from .classifier import download_model
            print(download_model())
        elif args.command == "evaluate":
            from .evaluation import evaluate
            from .settings import Settings
            Settings(sensitivity=args.sensitivity, min_confidence=args.min_confidence)
            print(json.dumps(evaluate(args.wav, args.annotations, args.backend, args.sensitivity,
                                      args.min_confidence, args.output, args.tolerance), indent=2))
        elif args.command == "prepare-training":
            from .dataset import prepare_dataset
            path, report = prepare_dataset(args.folder)
            print(f"Dataset manifest: {path}")
            print(json.dumps(report, indent=2))
        elif args.command == "label-training":
            from .labeler import run_labeler
            return run_labeler(args.manifest, args.smoke_seconds, advanced=args.advanced)
        elif args.command == "build-weapon-dataset":
            from .weapon_pipeline import build_weapon_dataset
            report = build_weapon_dataset(args.csv, args.dataset, storage=args.storage, dry_run=args.dry_run,
                                          max_sources=args.max_sources, source_url=args.source_url, pre=args.pre,
                                          post=args.post, min_spacing=args.min_spacing,
                                          ffmpeg_location=args.ffmpeg_location)
            printable = {key: value for key, value in report.items() if key != "rows"}
            print(json.dumps(printable, indent=2))
            if not args.dry_run:
                print(f'Next: .\\.venv\\Scripts\\python.exe -m arc_audio label-training "{args.dataset}"')
            return 1 if report["failed"] else 0
        elif args.command == "weapon-dataset-status":
            from .weapon_pipeline import weapon_dataset_status
            print(json.dumps(weapon_dataset_status(args.manifest), indent=2))
        elif args.command == "discover-weapon-sources":
            from .dataset import save_dataset
            from .weapon_discovery import discover_weapon_sources
            report = discover_weapon_sources(args.weapon, args.max_results)
            save_dataset(args.output, report)
            if args.dataset:
                from .weapon_pipeline import register_discoveries
                register_discoveries(args.dataset, report)
            print(f"Saved {len(report['sources'])} UNVERIFIED source candidates: {args.output}")
            print("No media downloaded or annotations created. Verify source relevance and ranges before adding curated CSV rows.")
            if report["errors"]:
                print(json.dumps({"errors": report["errors"]}, indent=2))
            return int(not report["sources"] and bool(report["errors"]))
        elif args.command == "suggest-weapon-background":
            from .weapon_negatives import propose_background
            report = propose_background(args.manifest, seconds=args.seconds, margin=args.margin,
                                        max_per_source=args.max_per_source)
            print(json.dumps(report, indent=2))
        elif args.command == "suggest-labels":
            from .suggestions import generate_suggestions
            path, report = generate_suggestions(args.manifest, model_path=args.model_dir,
                                                threshold=args.threshold, window_seconds=args.window_seconds,
                                                hop_seconds=args.hop_seconds, classes=tuple(args.classes),
                                                allow_unconfirmed=args.allow_unconfirmed_track)
            print(f"Review-only proposals saved: {path}")
            print(json.dumps(report, indent=2))
        elif args.command == "train-local":
            from .dataset import CORE_CLASSES, TARGET_CLASSES
            from .local_model import train_model
            classes = TARGET_CLASSES if args.classes == ["all"] else tuple(args.classes or CORE_CLASSES)
            path, report = train_model(args.manifest, args.output, classes=classes)
            print(f"Model saved: {path}")
            print(json.dumps(report, indent=2))
        elif args.command == "training-status":
            from .dataset import dataset_summary, load_dataset
            print(json.dumps(dataset_summary(load_dataset(args.manifest)), indent=2))
        elif args.command == "evaluate-training":
            from .settings import Settings
            from .training_evaluation import evaluate_dataset, write_report
            Settings(min_confidence=args.min_confidence)
            report = evaluate_dataset(args.manifest, args.split, args.model, args.min_confidence, args.tolerance)
            write_report(args.output, report)
            print(json.dumps({k: v for k, v in report.items() if k != "clips"}, indent=2))
            print(f"Detailed local report: {args.output}")
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
