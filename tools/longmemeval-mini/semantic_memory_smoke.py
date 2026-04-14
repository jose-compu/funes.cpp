#!/usr/bin/env python3
"""Ad-hoc smoke tests for semantic memory plumbing in llama-cli.

These tests use fabricated facts and validate:
1) teaching a fabricated fact writes one embedding row.
2) teaching a second fabricated fact appends a second, distinct row.
3) querying a fabricated fact with memory ON triggers semantic-memory injection,
   while memory OFF does not.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

FACT_CAPITAL = "The capital city of FantasyLand is WonderCity."
FACT_CURRENCY = "The official currency of FantasyLand is StarCoin."
QUERY_CAPITAL = "What is the capital city of FantasyLand? Reply with one word."


def run_cli(
    llama_cli: str,
    model: str,
    lines: list[str],
    semantic_file: str | None,
    n_predict: int,
    temp: float,
) -> str:
    cmd = [llama_cli, "-m", model, "--temp", str(temp), "-n", str(n_predict)]
    env = dict(os.environ)
    env["LLAMA_SEMANTIC_MEMORY_DEBUG"] = "2"

    if semantic_file:
        cmd += ["--semantic-memory-file", semantic_file]

    payload = "\n".join(lines + ["/exit"]) + "\n"
    proc = subprocess.run(
        cmd,
        input=payload,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-cli failed ({proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout + "\n" + proc.stderr


def _read_rows(semantic_file: str) -> list[str]:
    p = Path(semantic_file)
    if not p.exists():
        return []
    return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _dim_from_row(row: str) -> int:
    return len(row.split())


def test_teach_writes_fabricated_fact(
    llama_cli: str,
    model: str,
    semantic_file: str,
    n_predict: int,
    temp: float,
) -> tuple[bool, str]:
    out = run_cli(
        llama_cli,
        model,
        [f"/teach {FACT_CAPITAL}"],
        semantic_file,
        n_predict,
        temp,
    )
    if "taught 1 memory row" not in out:
        return False, "missing teach confirmation in output"
    rows = _read_rows(semantic_file)
    if not rows:
        return False, "memory file was not created"
    if len(rows) != 1:
        return False, f"expected 1 row, got {len(rows)}"
    dims = _dim_from_row(rows[0])
    if dims <= 0:
        return False, "row has invalid embedding dimension"
    return True, f"fact=capital row_count=1 dims={dims}"


def test_second_fabricated_fact_appends_distinct_row(
    llama_cli: str,
    model: str,
    semantic_file: str,
    n_predict: int,
    temp: float,
) -> tuple[bool, str]:
    before_rows = _read_rows(semantic_file)
    if len(before_rows) != 1:
        return False, f"expected 1 existing row before second teach, got {len(before_rows)}"

    out = run_cli(
        llama_cli,
        model,
        [f"/teach {FACT_CURRENCY}"],
        semantic_file,
        n_predict,
        temp,
    )
    if "taught 1 memory row" not in out:
        return False, "missing second teach confirmation in output"

    rows = _read_rows(semantic_file)
    if len(rows) != 2:
        return False, f"expected 2 rows after second teach, got {len(rows)}"
    if rows[0] == rows[1]:
        return False, "row 1 and row 2 are identical (unexpected for distinct facts)"
    return True, "fact=currency row_count=2 distinct_rows=true"


def test_fabricated_fact_query_toggles_memory_path(
    llama_cli: str,
    model: str,
    semantic_file: str,
    n_predict: int,
    temp: float,
) -> tuple[bool, str]:
    on_out = run_cli(
        llama_cli,
        model,
        [QUERY_CAPITAL],
        semantic_file=semantic_file,
        n_predict=n_predict,
        temp=temp,
    )
    off_out = run_cli(
        llama_cli,
        model,
        [QUERY_CAPITAL],
        semantic_file=None,
        n_predict=n_predict,
        temp=temp,
    )

    needle = "[semantic-memory] maybe_inject_semantic_memory active"
    if needle not in on_out:
        return False, "memory ON query missing activation line"
    if needle in off_out:
        return False, "memory OFF query unexpectedly has activation line"
    return True, "fabricated-query ON=active OFF=inactive"


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic memory smoke tests")
    parser.add_argument(
        "--llama-cli",
        default=str(Path(__file__).resolve().parents[2] / "build-funes-lfm2only/bin/llama-cli"),
        help="path to llama-cli binary",
    )
    parser.add_argument(
        "--model",
        default=str(Path.home() / ".models/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"),
        help="path to GGUF model",
    )
    parser.add_argument("--n-predict", type=int, default=16, help="max generated tokens")
    parser.add_argument("--temp", type=float, default=0.0, help="temperature")
    args = parser.parse_args()

    if not Path(args.llama_cli).exists():
        raise SystemExit(f"llama-cli not found: {args.llama_cli}")
    if not Path(args.model).exists():
        raise SystemExit(f"model not found: {args.model}")

    with tempfile.TemporaryDirectory(prefix="semantic-mem-smoke-") as td:
        mem_file = str(Path(td) / "semantic.mem")
        tests = [
            ("teach_fabricated_fact_1", lambda: test_teach_writes_fabricated_fact(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("teach_fabricated_fact_2", lambda: test_second_fabricated_fact_appends_distinct_row(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("query_fabricated_fact_memory_toggle", lambda: test_fabricated_fact_query_toggles_memory_path(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
        ]

        failures = 0
        for name, fn in tests:
            ok, detail = fn()
            print(f"[{'PASS' if ok else 'FAIL'}] {name} | {detail}")
            failures += int(not ok)

        print(f"TOTAL: {len(tests) - failures}/{len(tests)} passed")
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

