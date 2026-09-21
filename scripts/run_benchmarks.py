#!/usr/bin/env python3
"""Save benchmark output and its environment in a new results directory."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def capture(command):
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except OSError as error:
        return {"command": command, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build-gpu", help="CMake build directory")
    parser.add_argument("--output", help="New output directory; existing directories are rejected")
    args = parser.parse_args()
    build = (ROOT / args.build_dir).resolve()
    binary = build / "risk_benchmark"
    if not binary.is_file():
        parser.error(f"Build risk_benchmark first: {binary}")
    started = datetime.now(timezone.utc)
    output = (ROOT / args.output).resolve() if args.output else (
        ROOT / "benchmarks" / started.strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    commands = [["git", "rev-parse", "HEAD"], ["git", "status", "--short"],
                ["c++", "--version"], ["nvcc", "--version"], ["nvidia-smi"], ["lscpu"]]
    environment = {"started_utc": started.isoformat(), "platform": platform.platform(),
                   "binary": str(binary), "commands": [capture(cmd) for cmd in commands]}
    cache = build / "CMakeCache.txt"
    if cache.is_file():
        (output / "CMakeCache.txt").write_text(cache.read_text())
    with (output / "results.csv").open("w") as results, (output / "summary.txt").open("w") as summary:
        result = subprocess.run([str(binary)], cwd=ROOT, stdout=results, stderr=summary, check=False)
    environment["benchmark_exit_code"] = result.returncode
    environment["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print((output / "summary.txt").read_text(), end="")
    print(f"Saved results: {output}")
    if result.returncode:
        print("Benchmark failed; these results are incomplete.", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
