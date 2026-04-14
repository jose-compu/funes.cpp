#!/usr/bin/env python3
"""Real llama-cli A/B benchmark (semantic memory ON vs OFF).

This script reuses examples from benchmark.py and runs them through llama-cli:
- memory ON: teach in one session, query in a new session (file persistence path)
- memory OFF: query only
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from benchmark import ABSTAIN, Example, build_examples


@dataclass
class ABResult:
    case_id: str
    category: str
    expected: str
    on_answer: str
    off_answer: str
    on_ok: bool
    off_ok: bool


def _apply_backspaces(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch == "\b":
            if out:
                out.pop()
            continue
        out.append(ch)
    return "".join(out)


def _clean_console(text: str) -> str:
    text = _apply_backspaces(text.replace("\r", ""))
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    return text


def _extract_answer(raw_output: str, query: str) -> str:
    clean = _clean_console(raw_output)

    start = clean.rfind(query)
    segment = clean[start + len(query):] if start >= 0 else clean

    markers = [
        "\n[ Prompt:",
        "\n> /exit",
        "\nExiting...",
        "\nllama_memory_breakdown_print:",
    ]
    cut_at = len(segment)
    for marker in markers:
        idx = segment.find(marker)
        if idx >= 0:
            cut_at = min(cut_at, idx)
    segment = segment[:cut_at]

    lines = [ln.strip() for ln in segment.splitlines()]
    filtered = []
    for ln in lines:
        if not ln:
            continue
        if ln.startswith(">"):
            continue
        if ln.startswith("[ Prompt:"):
            continue
        if "ggml_" in ln or "llama_memory_breakdown" in ln:
            continue
        filtered.append(ln)

    if not filtered:
        return ""
    # Keep the full generated segment instead of only the last line.
    # Some answers appear in early lines and later lines are generic/truncated.
    return " ".join(filtered)


_WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}


def _normalize_numbers(text: str) -> str:
    """Replace written-out numbers with digits so '1' matches 'one'."""
    words = text.split()
    out = []
    for w in words:
        clean = w.strip(".,;:!?")
        if clean in _WORD_TO_DIGIT:
            out.append(w.replace(clean, _WORD_TO_DIGIT[clean]))
        else:
            out.append(w)
    return " ".join(out)


_ABSTAIN_PHRASES = [
    "don't have enough information",
    "don't have that information",
    "don't have access",
    "can't assist with that",
    "cannot assist",
    "i'm sorry, but i can't",
    "not available in my memory",
    "no record of",
    "wasn't shared with me",
]


def _is_correct(expected: str, answer: str) -> bool:
    exp = expected.strip().lower()
    ans = answer.strip().lower()
    if exp == ABSTAIN.lower():
        return any(phrase in ans for phrase in _ABSTAIN_PHRASES)
    if exp in ans:
        return True
    ans_norm = _normalize_numbers(ans)
    return exp in ans_norm


def _semantic_meta_path(semantic_file: str) -> str:
    return f"{semantic_file}.meta.jsonl"


def _run_llama_cli(
    llama_cli: str,
    model: str,
    stdin_lines: list[str],
    n_predict: int,
    temp: float,
    semantic_file: str | None,
    strength: float | None,
    extra_cli_args: list[str] | None = None,
    system_prompt: str | None = None,
) -> str:
    cmd = [llama_cli, "-m", model, "--temp", str(temp), "-n", str(n_predict)]
    if system_prompt:
        cmd += ["-sys", system_prompt]
    if extra_cli_args:
        cmd += extra_cli_args
    if semantic_file:
        cmd += ["--semantic-memory-file", semantic_file]
        if strength is not None:
            cmd += ["--semantic-memory-strength", str(strength)]

    payload = "\n".join(stdin_lines + ["/exit"]) + "\n"
    proc = subprocess.run(
        cmd,
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-cli failed ({proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout + "\n" + proc.stderr


def _teach_lines(example: Example) -> list[str]:
    lines: list[str] = []
    for session in example.sessions:
        for turn in session:
            lines.append(f"/teach [{turn.role}] {turn.text}")
    return lines


def run_ab_case(
    example: Example,
    llama_cli: str,
    model: str,
    semantic_file: str,
    n_predict: int,
    temp: float,
    strength: float | None,
    extra_cli_args: list[str] | None = None,
    system_prompt: str | None = None,
) -> ABResult:
    if os.path.exists(semantic_file):
        os.unlink(semantic_file)
    meta_file = _semantic_meta_path(semantic_file)
    if os.path.exists(meta_file):
        os.unlink(meta_file)

    # ON: teach in one session, query in a new session to validate persistence.
    teach_output = _run_llama_cli(
        llama_cli=llama_cli,
        model=model,
        stdin_lines=_teach_lines(example),
        n_predict=n_predict,
        temp=temp,
        semantic_file=semantic_file,
        strength=strength,
        extra_cli_args=extra_cli_args,
        system_prompt=system_prompt,
    )
    if "taught" not in teach_output.lower():
        raise RuntimeError(f"teaching appears to have failed for {example.case_id}")

    on_output = _run_llama_cli(
        llama_cli=llama_cli,
        model=model,
        stdin_lines=[example.question],
        n_predict=n_predict,
        temp=temp,
        semantic_file=semantic_file,
        strength=strength,
        extra_cli_args=extra_cli_args,
        system_prompt=system_prompt,
    )
    on_answer = _extract_answer(on_output, example.question)

    # OFF: no semantic memory file.
    off_output = _run_llama_cli(
        llama_cli=llama_cli,
        model=model,
        stdin_lines=[example.question],
        n_predict=n_predict,
        temp=temp,
        semantic_file=None,
        strength=None,
        extra_cli_args=extra_cli_args,
        system_prompt=system_prompt,
    )
    off_answer = _extract_answer(off_output, example.question)

    return ABResult(
        case_id=example.case_id,
        category=example.category,
        expected=example.expected,
        on_answer=on_answer,
        off_answer=off_answer,
        on_ok=_is_correct(example.expected, on_answer),
        off_ok=_is_correct(example.expected, off_answer),
    )


def select_examples(cases_per_category: int, max_cases: int | None) -> list[Example]:
    grouped: dict[str, list[Example]] = defaultdict(list)
    for ex in build_examples():
        grouped[ex.category].append(ex)

    selected: list[Example] = []
    for category in sorted(grouped.keys()):
        selected.extend(grouped[category][:cases_per_category])

    if max_cases is not None:
        selected = selected[:max_cases]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Real llama-cli semantic memory A/B benchmark")
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
    parser.add_argument("--cases-per-category", type=int, default=11, help="cases per LongMemEval category")
    parser.add_argument("--max-cases", type=int, default=None, help="limit total cases")
    parser.add_argument("--n-predict", type=int, default=32, help="max generated tokens for answers")
    parser.add_argument("--temp", type=float, default=0.0, help="sampling temperature")
    parser.add_argument("--semantic-strength", type=float, default=None, help="optional semantic memory strength")
    args = parser.parse_args()

    if not Path(args.llama_cli).exists():
        raise SystemExit(f"llama-cli not found: {args.llama_cli}")
    if not Path(args.model).exists():
        raise SystemExit(f"model not found: {args.model}")

    examples = select_examples(args.cases_per_category, args.max_cases)
    if not examples:
        raise SystemExit("no examples selected")

    with tempfile.TemporaryDirectory(prefix="lfm2-sem-ab-") as td:
        semantic_file = str(Path(td) / "semantic.mem")
        print(f"Running real llama-cli A/B on {len(examples)} case(s)")
        print(f"llama-cli: {args.llama_cli}")
        print(f"model    : {args.model}")
        print(f"sem file : {semantic_file}")
        print("-" * 88)

        results: list[ABResult] = []
        for ex in examples:
            res = run_ab_case(
                example=ex,
                llama_cli=args.llama_cli,
                model=args.model,
                semantic_file=semantic_file,
                n_predict=args.n_predict,
                temp=args.temp,
                strength=args.semantic_strength,
            )
            results.append(res)
            print(
                f"[{ex.category}] {ex.case_id}\n"
                f"  expected: {res.expected!r}\n"
                f"  ON : {res.on_answer!r}  ({'OK' if res.on_ok else 'FAIL'})\n"
                f"  OFF: {res.off_answer!r} ({'OK' if res.off_ok else 'FAIL'})\n"
                f"  delta_on_minus_off: {int(res.on_ok) - int(res.off_ok)}"
            )
            print("-" * 88)

    on_total = sum(int(r.on_ok) for r in results)
    off_total = sum(int(r.off_ok) for r in results)
    improved = sum(int(r.on_ok and not r.off_ok) for r in results)
    regressed = sum(int((not r.on_ok) and r.off_ok) for r in results)

    print("SUMMARY")
    print(f"ON  accuracy : {on_total}/{len(results)}")
    print(f"OFF accuracy : {off_total}/{len(results)}")
    print(f"improved     : {improved}")
    print(f"regressed    : {regressed}")


if __name__ == "__main__":
    main()

