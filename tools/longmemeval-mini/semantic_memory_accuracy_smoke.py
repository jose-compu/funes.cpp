#!/usr/bin/env python3
"""Accuracy-focused semantic memory smoke tests.

This suite checks QA/output behavior (expected answer matching), not just
plumbing/injection activation.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from benchmark import ABSTAIN, Example, Turn
from llama_cli_ab_test import ABResult, run_ab_case


def build_accuracy_cases() -> list[Example]:
    return [
        Example(
            case_id="acc-single-session-user-1",
            category="single-session-user",
            sessions=[[Turn("user", "My daily commute to work takes 42 minutes each way.")]],
            question="How long is my commute to work?",
            expected="42 minutes each way",
        ),
        Example(
            case_id="acc-single-session-assistant-1",
            category="single-session-assistant",
            sessions=[[Turn("assistant", "I recommend Roscioli for a romantic dinner.")]],
            question="What romantic dinner restaurant did you recommend?",
            expected="Roscioli",
        ),
        Example(
            case_id="acc-single-session-preference-1",
            category="single-session-preference",
            sessions=[[Turn("user", "I use a Sony camera body and Sony lenses for all shoots.")]],
            question="Which brand should accessory suggestions prioritize?",
            expected="Sony",
        ),
        Example(
            case_id="acc-knowledge-update-1",
            category="knowledge-update",
            sessions=[[
                Turn("user", "I own a road bike."),
                Turn("user", "I own a mountain bike."),
                Turn("user", "I sold my road bike."),
            ]],
            question="How many bikes do I currently own?",
            expected="1",
        ),
        Example(
            case_id="acc-abstention-1",
            category="abstention",
            sessions=[[
                Turn("user", "I repainted my kitchen last weekend."),
                Turn("user", "I also replaced two light bulbs."),
            ]],
            question="What is my passport number?",
            expected=ABSTAIN,
        ),
        Example(
            case_id="acc-temporal-1",
            category="temporal-reasoning",
            sessions=[[
                Turn("user", "On 2025-01-22, I went to the Science Museum with a friend."),
                Turn("user", "Current date is 2025-06-25."),
            ]],
            question="How many months have passed since my last museum visit with a friend?",
            expected="5",
        ),
        Example(
            case_id="acc-multi-session-1",
            category="multi-session",
            sessions=[
                [Turn("user", "I own a Korg B1 keyboard.")],
                [Turn("user", "I own a Yamaha FG800 guitar.")],
                [Turn("user", "I sold my Korg B1 keyboard.")],
            ],
            question="How many musical instruments do I currently own?",
            expected="1",
        ),
    ]


def print_case_result(r: ABResult) -> None:
    print(
        f"[{r.category}] {r.case_id}\n"
        f"  expected: {r.expected!r}\n"
        f"  ON : {r.on_answer!r}  ({'OK' if r.on_ok else 'FAIL'})\n"
        f"  OFF: {r.off_answer!r} ({'OK' if r.off_ok else 'FAIL'})\n"
        f"  delta_on_minus_off: {int(r.on_ok) - int(r.off_ok)}"
    )
    print("-" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy-focused semantic memory smoke benchmark")
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
    parser.add_argument("--n-predict", type=int, default=28, help="max generated tokens")
    parser.add_argument("--temp", type=float, default=0.0, help="sampling temperature")
    parser.add_argument("--semantic-strength", type=float, default=8.0, help="semantic memory strength")
    parser.add_argument(
        "--teach-variants",
        action="store_true",
        help="enable answer-focused teach variants in llama-cli",
    )
    parser.add_argument(
        "--hint-tokens",
        type=int,
        default=0,
        help="append minimal memory hint tokens to user prompt",
    )
    parser.add_argument(
        "--hint-llm-compress",
        action="store_true",
        help="use same LLM to compress top retrieved memory rows into one short hint",
    )
    parser.add_argument(
        "--hint-llm-top-k",
        type=int,
        default=3,
        help="top-k retrieved memory rows used by LLM hint compressor",
    )
    parser.add_argument(
        "--hint-llm-n-predict",
        type=int,
        default=48,
        help="max generated tokens for compressed hint",
    )
    parser.add_argument(
        "--memory-logit-bias",
        action="store_true",
        help="apply memory-derived logit bias (no hint text injection)",
    )
    parser.add_argument(
        "--memory-logit-bias-top-k",
        type=int,
        default=2,
        help="top-k memory rows used to build dynamic logit bias",
    )
    parser.add_argument(
        "--memory-logit-bias-max-tokens",
        type=int,
        default=8,
        help="max unique tokens included in dynamic logit bias",
    )
    parser.add_argument(
        "--memory-logit-bias-strength",
        type=float,
        default=0.8,
        help="bias strength per selected memory token",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="system prompt forwarded to llama-cli (e.g. current date for temporal reasoning)",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="limit number of test cases")
    parser.add_argument(
        "--fail-on-no-lift",
        action="store_true",
        help="exit non-zero if memory ON does not beat memory OFF",
    )
    args = parser.parse_args()

    if not Path(args.llama_cli).exists():
        raise SystemExit(f"llama-cli not found: {args.llama_cli}")
    if not Path(args.model).exists():
        raise SystemExit(f"model not found: {args.model}")

    cases = build_accuracy_cases()
    if args.max_cases is not None:
        cases = cases[:args.max_cases]

    with tempfile.TemporaryDirectory(prefix="semantic-mem-acc-smoke-") as td:
        semantic_file = str(Path(td) / "semantic.mem")
        print(f"Running semantic memory accuracy smoke on {len(cases)} case(s)")
        print(f"llama-cli: {args.llama_cli}")
        print(f"model    : {args.model}")
        print(f"sem file : {semantic_file}")
        print("-" * 88)

        results: list[ABResult] = []
        for case in cases:
            extra_args: list[str] | None = []
            if args.teach_variants:
                extra_args.append("--semantic-memory-teach-variants")
            if args.hint_tokens > 0:
                extra_args += ["--semantic-memory-hint-tokens", str(args.hint_tokens)]
            if args.hint_llm_compress:
                extra_args.append("--semantic-memory-hint-llm-compress")
                extra_args += ["--semantic-memory-hint-llm-top-k", str(max(1, args.hint_llm_top_k))]
                extra_args += ["--semantic-memory-hint-llm-n-predict", str(max(8, args.hint_llm_n_predict))]
            if args.memory_logit_bias:
                extra_args.append("--semantic-memory-logit-bias")
                extra_args += ["--semantic-memory-logit-bias-top-k", str(max(1, args.memory_logit_bias_top_k))]
                extra_args += ["--semantic-memory-logit-bias-max-tokens", str(max(1, args.memory_logit_bias_max_tokens))]
                extra_args += ["--semantic-memory-logit-bias-strength", str(max(0.0, args.memory_logit_bias_strength))]
            if not extra_args:
                extra_args = None
            res = run_ab_case(
                example=case,
                llama_cli=args.llama_cli,
                model=args.model,
                semantic_file=semantic_file,
                n_predict=args.n_predict,
                temp=args.temp,
                strength=args.semantic_strength,
                extra_cli_args=extra_args,
                system_prompt=args.system_prompt,
            )
            results.append(res)
            print_case_result(res)

    on_total = sum(int(r.on_ok) for r in results)
    off_total = sum(int(r.off_ok) for r in results)
    improved = sum(int(r.on_ok and not r.off_ok) for r in results)
    regressed = sum(int((not r.on_ok) and r.off_ok) for r in results)

    print("SUMMARY")
    print(f"ON  accuracy : {on_total}/{len(results)}")
    print(f"OFF accuracy : {off_total}/{len(results)}")
    print(f"improved     : {improved}")
    print(f"regressed    : {regressed}")

    if args.fail_on_no_lift and on_total <= off_total:
        raise SystemExit("no accuracy lift: ON <= OFF")


if __name__ == "__main__":
    main()

