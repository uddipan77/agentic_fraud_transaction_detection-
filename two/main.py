"""Main entry point: run the full fraud detection pipeline."""

import sys
import time
from pathlib import Path

from fraud_detection.config import Config
from fraud_detection.preprocess import preprocess_split
from fraud_detection.orchestrator import Orchestrator
from fraud_detection.output_writer import write_output_txt, write_predictions_csv, write_full_audit


def run_pipeline(splits: list[str] | None = None):
    """Run the complete fraud detection pipeline."""
    config = Config()
    output_dir = config.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config.groq_api_key:
        print("ERROR: GROQ_API_KEY not found. Set it in .env or environment.")
        sys.exit(1)

    if splits is None:
        splits = ["train", "validation"]

    start_time = time.time()

    # Phase 1: Preprocessing
    print("=" * 60)
    print("PHASE 1: PREPROCESSING - Building evidence bundles")
    print("=" * 60)

    for split in splits:
        preprocess_split(split, config)

    preprocess_time = time.time() - start_time
    print(f"\nPreprocessing completed in {preprocess_time:.1f}s\n")

    # Phase 2: Agent decision-making
    print("=" * 60)
    print("PHASE 2: AGENT DECISIONS - Primary + Reviewer agents")
    print("=" * 60)

    orchestrator = Orchestrator(config)

    for split in splits:
        agent_start = time.time()
        results = orchestrator.run_split(split)
        agent_time = time.time() - agent_start

        # Phase 3: Output generation
        print(f"\n[Output] Writing results for {split}...")
        write_output_txt(results, output_dir / f"{split}_output.txt")
        write_predictions_csv(results, output_dir / f"{split}_predictions.csv")
        write_full_audit(results, output_dir / f"{split}_audit.json")
        print(f"  Agent processing time: {agent_time:.1f}s")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE - Total time: {total_time:.1f}s")
    print(f"Output files in: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Parse command line args
    splits = None
    if len(sys.argv) > 1:
        splits = sys.argv[1:]
        valid = {"train", "validation"}
        for s in splits:
            if s not in valid:
                print(f"Invalid split: {s}. Must be one of: {valid}")
                sys.exit(1)

    run_pipeline(splits)
