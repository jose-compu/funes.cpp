#!/usr/bin/env python3
"""Extended semantic-memory smoke tests for llama-cli.

Focus: stable plumbing checks with richer flows, not semantic QA quality.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def run_cli(
    llama_cli: str,
    model: str,
    lines: list[str],
    *,
    semantic_file: str | None,
    n_predict: int,
    temp: float,
    extra_args: list[str] | None = None,
) -> str:
    cmd = [llama_cli, "-m", model, "--temp", str(temp), "-n", str(n_predict)]
    if extra_args:
        cmd += extra_args
    if semantic_file:
        cmd += ["--semantic-memory-file", semantic_file]

    env = dict(os.environ)
    env["LLAMA_SEMANTIC_MEMORY_DEBUG"] = "2"

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


def read_rows(mem_file: str) -> list[str]:
    p = Path(mem_file)
    if not p.exists():
        return []
    return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def check_rows_shape(rows: list[str]) -> tuple[bool, str]:
    if not rows:
        return False, "no rows in memory file"
    dims = [len(r.split()) for r in rows]
    if any(d <= 0 for d in dims):
        return False, "one or more rows have invalid dimension"
    if len(set(dims)) != 1:
        return False, f"inconsistent row dimensions: {dims}"
    return True, f"row_count={len(rows)} dims={dims[0]}"


def test_complex_teach_flow(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    out = run_cli(
        llama_cli,
        model,
        [
            "/teach The capital city of FantasyLand is WonderCity.",
            "<teach>The official currency of FantasyLand is StarCoin.</teach>",
            "<learn-from-response>Reply with exactly TOKEN-77.</learn-from-response>",
        ],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
    )
    if "taught 1 memory row" not in out:
        return False, "missing explicit /teach confirmation"
    if "learned 1 memory row(s) from assistant response" not in out:
        return False, "missing learn-from-response confirmation"
    rows = read_rows(mem_file)
    ok_shape, detail = check_rows_shape(rows)
    if not ok_shape:
        return False, detail
    if len(rows) < 3:
        return False, f"expected at least 3 rows after complex flow, got {len(rows)}"
    return True, f"{detail} (complex flow produced >=3 rows)"


def test_cross_session_activation_count(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    rows = read_rows(mem_file)
    expected = len(rows)
    if expected <= 0:
        return False, "memory file unexpectedly empty before query"

    out = run_cli(
        llama_cli,
        model,
        ["What is the capital city of FantasyLand? Reply with one word."],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
    )
    needle = f"n_mem={expected}"
    if "[semantic-memory] maybe_inject_semantic_memory active" not in out:
        return False, "missing semantic-memory activation debug line"
    if needle not in out:
        return False, f"activation line missing expected n_mem count ({expected})"
    return True, f"activation detected with n_mem={expected}"


def test_memory_off_no_activation(llama_cli: str, model: str, n_predict: int, temp: float) -> tuple[bool, str]:
    out = run_cli(
        llama_cli,
        model,
        ["What is the capital city of FantasyLand? Reply with one word."],
        semantic_file=None,
        n_predict=n_predict,
        temp=temp,
    )
    if "[semantic-memory] maybe_inject_semantic_memory active" in out:
        return False, "activation line present while memory OFF"
    return True, "no activation line with memory OFF"


def test_disable_teach_tags_flag(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    base_rows = len(read_rows(mem_file))
    run_cli(
        llama_cli,
        model,
        ["<teach>This should NOT be learned because tags are disabled.</teach>"],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
        extra_args=["--no-semantic-memory-teach-tags"],
    )
    after_rows = len(read_rows(mem_file))
    if after_rows != base_rows:
        return False, f"row count changed with teach tags disabled: {base_rows} -> {after_rows}"
    return True, f"row count unchanged with tags disabled ({after_rows})"


def test_teach_prefix_flow(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    base_rows = len(read_rows(mem_file))
    out = run_cli(
        llama_cli,
        model,
        ["MEM: The bird of FantasyLand is SkyFalcon."],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
        extra_args=["--semantic-memory-teach-prefix", "MEM:"],
    )
    if "taught 1 memory row(s)" not in out:
        return False, "teach prefix did not trigger memory write confirmation"
    after_rows = len(read_rows(mem_file))
    if after_rows != base_rows + 1:
        return False, f"expected row increment by prefix teach: {base_rows} -> {after_rows}"
    return True, f"prefix teach row increment ok ({after_rows})"


def test_custom_teach_tags_flow(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    base_rows = len(read_rows(mem_file))
    out = run_cli(
        llama_cli,
        model,
        ["<mem>The gemstone of FantasyLand is DawnRuby.</mem>"],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
        extra_args=[
            "--semantic-memory-teach-open-tag", "<mem>",
            "--semantic-memory-teach-close-tag", "</mem>",
        ],
    )
    if "taught 1 memory row(s)" not in out:
        return False, "custom teach tags did not trigger memory write confirmation"
    after_rows = len(read_rows(mem_file))
    if after_rows != base_rows + 1:
        return False, f"expected row increment by custom tag teach: {base_rows} -> {after_rows}"
    return True, f"custom teach tag row increment ok ({after_rows})"


def test_disable_learn_from_response_flag(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    base_rows = len(read_rows(mem_file))
    out = run_cli(
        llama_cli,
        model,
        ["<learn-from-response>Reply with exactly TOKEN-88.</learn-from-response>"],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
        extra_args=["--no-semantic-memory-learn-response-tags"],
    )
    if "learned 1 memory row(s) from assistant response" in out:
        return False, "learn-from-response happened despite disable flag"
    after_rows = len(read_rows(mem_file))
    if after_rows != base_rows:
        return False, f"row count changed with learn-from-response disabled: {base_rows} -> {after_rows}"
    return True, f"learn-from-response disable respected ({after_rows})"


def test_unclosed_teach_tag_is_ignored(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    base_rows = len(read_rows(mem_file))
    out = run_cli(
        llama_cli,
        model,
        ["<teach>This malformed tag should not be stored"],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
    )
    if "taught 1 memory row(s)" in out:
        return False, "malformed teach tag unexpectedly wrote memory"
    after_rows = len(read_rows(mem_file))
    if after_rows != base_rows:
        return False, f"row count changed for malformed tag: {base_rows} -> {after_rows}"
    return True, f"malformed tag ignored ({after_rows})"


def test_multisession_growth(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    base_rows = len(read_rows(mem_file))
    run_cli(
        llama_cli,
        model,
        ["/teach Session one fact: comet color is teal."],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
    )
    mid_rows = len(read_rows(mem_file))
    if mid_rows != base_rows + 1:
        return False, f"session-1 write mismatch: {base_rows} -> {mid_rows}"

    run_cli(
        llama_cli,
        model,
        ["/teach Session two fact: moon name is SilverOrb."],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
    )
    end_rows = len(read_rows(mem_file))
    if end_rows != mid_rows + 1:
        return False, f"session-2 write mismatch: {mid_rows} -> {end_rows}"
    return True, f"multisession growth ok ({base_rows}->{mid_rows}->{end_rows})"


def test_multitoken_multiword_fact(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    base_rows = len(read_rows(mem_file))
    out = run_cli(
        llama_cli,
        model,
        [
            "/teach The official emergency phrase for FantasyLand airships is crimson lantern protocol delta seven.",
        ],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
    )
    if "taught 1 memory row" not in out:
        return False, "multi-token /teach did not confirm write"
    after_rows = len(read_rows(mem_file))
    if after_rows != base_rows + 1:
        return False, f"row count mismatch for multi-token fact: {base_rows} -> {after_rows}"
    return True, f"multi-token fact stored ({after_rows})"


def test_multisentence_teach_block(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    base_rows = len(read_rows(mem_file))
    out = run_cli(
        llama_cli,
        model,
        [
            "<teach>FantasyLand council bulletin: The eastern gate opens at dawn. "
            "The silver bridge remains closed on rainy days. "
            "Couriers must carry a sky-blue pass before entering the archive quarter.</teach>",
        ],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
    )
    if "taught 1 memory row(s)" not in out:
        return False, "multi-sentence tag teach did not confirm write"
    after_rows = len(read_rows(mem_file))
    if after_rows != base_rows + 1:
        return False, f"row count mismatch for multi-sentence block: {base_rows} -> {after_rows}"
    return True, f"multi-sentence block stored ({after_rows})"


def test_multiparagraph_style_teach_chunks(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    base_rows = len(read_rows(mem_file))
    lines = [
        "/teach Paragraph 1: FantasyLand harbor report. Cargo manifests are checked at sunrise; lighthouse code is amber-3.",
        "/teach Paragraph 2: FantasyLand observatory memo. Night calibration begins at 22:00; telescope wing B tracks comet Helios.",
        "/teach Paragraph 3: FantasyLand civil notice. Public market closes early on moon-festival eve; tram route nine is diverted.",
    ]
    out = run_cli(
        llama_cli,
        model,
        lines,
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
    )
    taught_hits = out.lower().count("taught 1 memory row")
    if taught_hits < 3:
        return False, f"expected >=3 teach confirmations for paragraph chunks, got {taught_hits}"
    after_rows = len(read_rows(mem_file))
    if after_rows != base_rows + 3:
        return False, f"row count mismatch for paragraph chunks: {base_rows} -> {after_rows}"
    ok_shape, detail = check_rows_shape(read_rows(mem_file))
    if not ok_shape:
        return False, detail
    return True, f"paragraph-style chunks stored (+3 rows, {detail})"


def test_multiparagraph_query_activation(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    expected = len(read_rows(mem_file))
    if expected <= 0:
        return False, "no rows available before multiparagraph activation query"
    out = run_cli(
        llama_cli,
        model,
        [
            "Summarize the FantasyLand harbor, observatory, and civil notices in one sentence.",
        ],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
    )
    if "[semantic-memory] maybe_inject_semantic_memory active" not in out:
        return False, "missing activation line for multiparagraph query"
    if f"n_mem={expected}" not in out:
        return False, f"multiparagraph query missing expected n_mem ({expected})"
    return True, f"multiparagraph query activation ok (n_mem={expected})"


def test_numeric_datetime_fact_block(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    base_rows = len(read_rows(mem_file))
    out = run_cli(
        llama_cli,
        model,
        [
            "/teach Incident record: batch_id=ZX-2048, checksum=9f86d081884c7d659a2feaa0c55ad015, timestamp=2026-04-13T10:45:30Z, retry_count=3.",
        ],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
    )
    if "taught 1 memory row" not in out:
        return False, "numeric/datetime fact did not confirm memory write"
    after_rows = len(read_rows(mem_file))
    if after_rows != base_rows + 1:
        return False, f"numeric/datetime row count mismatch: {base_rows} -> {after_rows}"
    return True, f"numeric+datetime fact stored ({after_rows})"


def test_structured_json_like_fact(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    base_rows = len(read_rows(mem_file))
    out = run_cli(
        llama_cli,
        model,
        [
            '/teach Config snapshot: {"service":"fantasy-gateway","region":"east-2","limits":{"rpm":1200,"burst":40},"flags":["safe_mode","audit"]}.',
        ],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
    )
    if "taught 1 memory row" not in out:
        return False, "json-like fact did not confirm memory write"
    after_rows = len(read_rows(mem_file))
    if after_rows != base_rows + 1:
        return False, f"json-like row count mismatch: {base_rows} -> {after_rows}"
    return True, f"json-like fact stored ({after_rows})"


def test_contradictory_updates_as_multiple_rows(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    base_rows = len(read_rows(mem_file))
    out = run_cli(
        llama_cli,
        model,
        [
            "/teach The secure vault code is ALPHA-111.",
            "/teach Update: The secure vault code is now BETA-222.",
        ],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
    )
    taught_hits = out.lower().count("taught 1 memory row")
    if taught_hits < 2:
        return False, f"expected two teach confirmations for contradictory updates, got {taught_hits}"
    rows = read_rows(mem_file)
    after_rows = len(rows)
    if after_rows != base_rows + 2:
        return False, f"contradictory updates row count mismatch: {base_rows} -> {after_rows}"
    if rows[-1] == rows[-2]:
        return False, "contradictory updates produced identical rows (unexpected)"
    return True, f"contradictory updates stored as separate rows ({after_rows})"


def test_long_list_fact_and_activation(llama_cli: str, model: str, mem_file: str, n_predict: int, temp: float) -> tuple[bool, str]:
    base_rows = len(read_rows(mem_file))
    out = run_cli(
        llama_cli,
        model,
        [
            "/teach Expedition inventory list: rope, compass, sextant, field_journal, med_kit, lantern, dry_rations, signal_flare, map_case, water_filter, gloves, thermal_blanket.",
            "Summarize the expedition inventory in one sentence.",
        ],
        semantic_file=mem_file,
        n_predict=n_predict,
        temp=temp,
    )
    if "taught 1 memory row" not in out:
        return False, "long-list fact did not confirm write"
    after_rows = len(read_rows(mem_file))
    if after_rows != base_rows + 1:
        return False, f"long-list row count mismatch: {base_rows} -> {after_rows}"
    if "[semantic-memory] maybe_inject_semantic_memory active" not in out:
        return False, "long-list query missing activation line"
    if f"n_mem={after_rows}" not in out:
        return False, f"long-list query activation missing expected n_mem ({after_rows})"
    return True, f"long-list fact stored and activated (n_mem={after_rows})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended semantic memory smoke tests")
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
    parser.add_argument("--n-predict", type=int, default=20, help="max generated tokens")
    parser.add_argument("--temp", type=float, default=0.0, help="temperature")
    args = parser.parse_args()

    if not Path(args.llama_cli).exists():
        raise SystemExit(f"llama-cli not found: {args.llama_cli}")
    if not Path(args.model).exists():
        raise SystemExit(f"model not found: {args.model}")

    with tempfile.TemporaryDirectory(prefix="semantic-mem-smoke-ext-") as td:
        mem_file = str(Path(td) / "semantic.mem")

        tests = [
            ("complex_teach_flow", lambda: test_complex_teach_flow(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("cross_session_activation_count", lambda: test_cross_session_activation_count(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("memory_off_no_activation", lambda: test_memory_off_no_activation(args.llama_cli, args.model, args.n_predict, args.temp)),
            ("disable_teach_tags_flag", lambda: test_disable_teach_tags_flag(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("teach_prefix_flow", lambda: test_teach_prefix_flow(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("custom_teach_tags_flow", lambda: test_custom_teach_tags_flow(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("disable_learn_from_response_flag", lambda: test_disable_learn_from_response_flag(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("unclosed_teach_tag_is_ignored", lambda: test_unclosed_teach_tag_is_ignored(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("multisession_growth", lambda: test_multisession_growth(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("multitoken_multiword_fact", lambda: test_multitoken_multiword_fact(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("multisentence_teach_block", lambda: test_multisentence_teach_block(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("multiparagraph_style_teach_chunks", lambda: test_multiparagraph_style_teach_chunks(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("multiparagraph_query_activation", lambda: test_multiparagraph_query_activation(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("numeric_datetime_fact_block", lambda: test_numeric_datetime_fact_block(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("structured_json_like_fact", lambda: test_structured_json_like_fact(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("contradictory_updates_as_multiple_rows", lambda: test_contradictory_updates_as_multiple_rows(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
            ("long_list_fact_and_activation", lambda: test_long_list_fact_and_activation(args.llama_cli, args.model, mem_file, args.n_predict, args.temp)),
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

