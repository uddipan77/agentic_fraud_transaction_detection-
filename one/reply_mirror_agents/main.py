from __future__ import annotations

import argparse

from config import PROJECT_ROOT, load_config
from orchestrator import FraudOrchestrator
from output_writer import create_source_archive, write_langfuse_sessions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reply Mirror agent-based fraud detection")
    parser.add_argument(
        "--dataset",
        choices=["train", "validation", "all"],
        default="all",
        help="Dataset split to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    config.validate()

    orchestrator = FraudOrchestrator(config)
    splits = ["train", "validation"] if args.dataset == "all" else [args.dataset]
    results = [orchestrator.run_dataset(split) for split in splits]

    sessions_payload = [
        {
            "split": result.split,
            "session_id": result.session_id,
            "trace_id": result.trace_id,
            "trace_url": result.trace_url,
            "output_file": str(result.output_path),
            "predictions_file": str(result.predictions_path),
            "fraud_count": len(result.fraud_ids),
        }
        for result in results
    ]
    write_langfuse_sessions(config.output_dir / "langfuse_sessions.json", sessions_payload)
    create_source_archive(PROJECT_ROOT, config.output_dir / "reply_mirror_agents_source.zip")

    for result in results:
        print(
            f"{result.split}: {len(result.fraud_ids)} fraud predictions | "
            f"output={result.output_path.name} | trace={result.trace_url or 'n/a'}"
        )


if __name__ == "__main__":
    main()