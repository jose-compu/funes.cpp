#!/usr/bin/env python3
"""Minimal LongMemEval-style benchmark for a persistent memory approach.

This is intentionally lightweight:
- Memory fits in RAM.
- A JSON file provides persistence across turns/sessions.
- Rule-based extraction keeps implementation small for quick iteration.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


ABSTAIN = "I don't have enough information."


@dataclass
class Turn:
    role: str
    text: str


@dataclass
class Example:
    case_id: str
    category: str
    sessions: list[list[Turn]]
    question: str
    expected: str


class PersistentMemory:
    def __init__(self, db_path: Path, enabled: bool = True):
        self.db_path = db_path
        self.enabled = enabled
        self.state: dict[str, dict[str, Any] | list[Any]] = {
            "facts": {},
            "sets": {},
            "events": [],
        }
        self._load()

    def _load(self) -> None:
        if not self.enabled:
            return
        if not self.db_path.exists():
            return
        with self.db_path.open("r", encoding="utf-8") as f:
            self.state = json.load(f)

    def save(self) -> None:
        if not self.enabled:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.db_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def _get_facts(self) -> dict[str, Any]:
        facts = self.state.get("facts")
        if isinstance(facts, dict):
            return facts
        return {}

    def _get_sets(self) -> dict[str, list[str]]:
        sets = self.state.get("sets")
        if isinstance(sets, dict):
            return sets
        return {}

    def _get_events(self) -> list[dict[str, Any]]:
        events = self.state.get("events")
        if isinstance(events, list):
            return events
        return []

    def set_fact(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        facts = self._get_facts()
        facts[key] = value
        self.state["facts"] = facts
        self.save()

    def get_fact(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        return self._get_facts().get(key)

    def add_set_item(self, set_key: str, value: str) -> None:
        if not self.enabled:
            return
        sets = self._get_sets()
        current = set(sets.get(set_key, []))
        current.add(value)
        sets[set_key] = sorted(current)
        self.state["sets"] = sets
        self.save()

    def remove_set_item(self, set_key: str, value: str) -> None:
        if not self.enabled:
            return
        sets = self._get_sets()
        current = set(sets.get(set_key, []))
        current.discard(value)
        sets[set_key] = sorted(current)
        self.state["sets"] = sets
        self.save()

    def get_set(self, set_key: str) -> list[str]:
        if not self.enabled:
            return []
        return self._get_sets().get(set_key, [])

    def add_event(self, event_type: str, when: str, metadata: dict[str, Any]) -> None:
        if not self.enabled:
            return
        events = self._get_events()
        events.append(
            {
                "event_type": event_type,
                "when": when,
                "metadata": metadata,
            }
        )
        self.state["events"] = events
        self.save()

    def latest_event(self, event_type: str, metadata_filter: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        metadata_filter = metadata_filter or {}
        for event in reversed(self._get_events()):
            evt_type = event.get("event_type")
            if evt_type != event_type:
                continue
            metadata = event.get("metadata")
            if isinstance(metadata, dict):
                if all(metadata.get(k) == v for k, v in metadata_filter.items()):
                    return event
        return None


class MemoryEngine:
    BRAND_RE = re.compile(r"\b(Sony|Canon|Nikon|Fujifilm|Panasonic)\b", re.IGNORECASE)
    COMMUTE_RE = re.compile(r"takes (\d+) minutes each way", re.IGNORECASE)
    CURRENT_DATE_RE = re.compile(r"current date is (\d{4}-\d{2}-\d{2})", re.IGNORECASE)
    DATE_PREFIX_RE = re.compile(r"on (\d{4}-\d{2}-\d{2}),", re.IGNORECASE)
    ASSISTANT_REC_RE = re.compile(r"recommend ([A-Za-z0-9' -]+)\.", re.IGNORECASE)
    OWN_BIKE_RE = re.compile(r"i (?:own|bought) a ([a-z0-9 -]+ bike)\.", re.IGNORECASE)
    SOLD_BIKE_RE = re.compile(r"i sold (?:my )?([a-z0-9 -]+ bike)\.", re.IGNORECASE)
    OWN_INSTR_RE = re.compile(r"i (?:own|bought) a ([a-z0-9' -]+ (?:keyboard|guitar|drum set|piano))\.", re.IGNORECASE)
    SOLD_INSTR_RE = re.compile(r"i sold (?:my )?([a-z0-9' -]+ (?:keyboard|guitar|drum set|piano))\.", re.IGNORECASE)
    FISH_COUNT_RE = re.compile(r"my 30-gallon tank has (\d+) fish", re.IGNORECASE)

    def __init__(self, memory: PersistentMemory):
        self.memory = memory

    def ingest(self, role: str, text: str) -> None:
        text = text.strip()

        commute = self.COMMUTE_RE.search(text)
        if commute:
            self.memory.set_fact("commute_minutes_each_way", int(commute.group(1)))

        current_date = self.CURRENT_DATE_RE.search(text)
        if current_date:
            self.memory.set_fact("current_date", current_date.group(1))

        museum_date = self.DATE_PREFIX_RE.search(text)
        if museum_date and "museum" in text.lower() and "friend" in text.lower():
            self.memory.add_event(
                "museum_visit",
                museum_date.group(1),
                {"with_friend": True},
            )

        if role == "assistant":
            match = self.ASSISTANT_REC_RE.search(text)
            if match:
                self.memory.set_fact("assistant_recommended_restaurant", match.group(1).strip())

        if role == "user":
            brand = self.BRAND_RE.search(text)
            if brand:
                self.memory.set_fact("preferred_brand", brand.group(1))

        fish_count = self.FISH_COUNT_RE.search(text)
        if fish_count:
            self.memory.set_fact("fish_count_30g", int(fish_count.group(1)))

        own_bike = self.OWN_BIKE_RE.search(text)
        if own_bike:
            self.memory.add_set_item("bikes", own_bike.group(1).strip().lower())

        sold_bike = self.SOLD_BIKE_RE.search(text)
        if sold_bike:
            self.memory.remove_set_item("bikes", sold_bike.group(1).strip().lower())

        own_instr = self.OWN_INSTR_RE.search(text)
        if own_instr:
            self.memory.add_set_item("instruments", own_instr.group(1).strip().lower())

        sold_instr = self.SOLD_INSTR_RE.search(text)
        if sold_instr:
            self.memory.remove_set_item("instruments", sold_instr.group(1).strip().lower())

    def answer(self, question: str) -> str:
        q = question.lower()

        if "commute" in q and "work" in q:
            value = self.memory.get_fact("commute_minutes_each_way")
            return f"{value} minutes each way" if value is not None else ABSTAIN

        if "romantic dinner restaurant" in q:
            value = self.memory.get_fact("assistant_recommended_restaurant")
            return str(value) if value is not None else ABSTAIN

        if "accessory" in q or "accessories" in q:
            value = self.memory.get_fact("preferred_brand")
            return f"{value}-compatible accessories" if value is not None else ABSTAIN

        if "how many bikes" in q:
            return str(len(self.memory.get_set("bikes")))

        if "how many fish" in q and "30-gallon" in q:
            value = self.memory.get_fact("fish_count_30g")
            return str(value) if value is not None else ABSTAIN

        if "how many months" in q and "museum visit with a friend" in q:
            current_date_raw = self.memory.get_fact("current_date")
            latest_friend_visit = self.memory.latest_event("museum_visit", {"with_friend": True})
            if not current_date_raw or not latest_friend_visit:
                return ABSTAIN
            current_dt = date.fromisoformat(current_date_raw)
            visit_dt = date.fromisoformat(latest_friend_visit["when"])
            delta_months = (current_dt.year - visit_dt.year) * 12 + (current_dt.month - visit_dt.month)
            if current_dt.day < visit_dt.day:
                delta_months -= 1
            return str(max(delta_months, 0))

        if "how many musical instruments" in q or "how many instruments" in q:
            return str(len(self.memory.get_set("instruments")))

        return ABSTAIN


def build_examples() -> list[Example]:
    categories: list[Example] = []

    commute_values = [45, 30, 60, 25, 50, 35, 55, 40, 20, 65, 70]
    for idx, minutes in enumerate(commute_values, start=1):
        categories.append(
            Example(
                case_id=f"single-session-user-{idx}",
                category="single-session-user",
                sessions=[
                    [
                        Turn("user", f"I've been listening to audiobooks during my daily commute, which takes {minutes} minutes each way."),
                    ]
                ],
                question="How long is my commute to work?",
                expected=f"{minutes} minutes each way",
            )
        )

    restaurants = [
        "Roscioli",
        "Da Enzo",
        "Armando al Pantheon",
        "Trattoria Monti",
        "SantoPalato",
        "Pipero",
        "Il Pagliaccio",
        "Per Me Giulio Terrinoni",
        "Retrobottega",
        "Aroma",
        "Pierluigi",
    ]
    for idx, restaurant in enumerate(restaurants, start=1):
        categories.append(
            Example(
                case_id=f"single-session-assistant-{idx}",
                category="single-session-assistant",
                sessions=[
                    [
                        Turn("user", "Which one would you say is the best for a romantic dinner?"),
                        Turn("assistant", f"I would recommend {restaurant}."),
                    ]
                ],
                question="Can you remind me of the romantic dinner restaurant you recommended?",
                expected=restaurant,
            )
        )

    brands = ["Sony", "Canon", "Nikon", "Fujifilm", "Panasonic", "Sony", "Canon", "Nikon", "Fujifilm", "Panasonic", "Sony"]
    for idx, brand in enumerate(brands, start=1):
        categories.append(
            Example(
                case_id=f"single-session-preference-{idx}",
                category="single-session-preference",
                sessions=[
                    [
                        Turn("user", f"I use a {brand} camera for most of my photography work."),
                        Turn("user", "I am thinking of upgrading my camera bag and flash setup."),
                    ]
                ],
                question="Which brand should accessory suggestions prioritize?",
                expected=f"{brand}-compatible accessories",
            )
        )

    knowledge_sequences = [
        (["I own a road bike.", "I own a mountain bike.", "I bought a hybrid bike."], "3"),
        (["I own a road bike.", "I own a mountain bike.", "I sold my mountain bike."], "1"),
        (["I own a commuter bike.", "I bought a gravel bike.", "I bought a folding bike."], "3"),
        (["I own a city bike.", "I sold my city bike.", "I bought a touring bike."], "1"),
        (["I own a road bike.", "I bought a hybrid bike.", "I sold my road bike."], "1"),
        (["I own a road bike.", "I bought a mountain bike.", "I sold my road bike.", "I bought a gravel bike."], "2"),
        (["I own a folding bike.", "I bought a city bike.", "I sold my city bike.", "I sold my folding bike."], "0"),
        (["I own a trail bike.", "I bought a commuter bike.", "I bought a BMX bike.", "I sold my commuter bike."], "2"),
        (["I own a road bike.", "I sold my road bike.", "I bought a touring bike.", "I bought a hybrid bike."], "2"),
        (["I own a mountain bike.", "I bought a gravel bike.", "I sold my gravel bike.", "I bought a city bike."], "2"),
        (["I own a commuter bike.", "I bought a road bike.", "I sold my commuter bike.", "I sold my road bike."], "0"),
    ]
    for idx, (turns, expected_count) in enumerate(knowledge_sequences, start=1):
        categories.append(
            Example(
                case_id=f"knowledge-update-{idx}",
                category="knowledge-update",
                sessions=[[Turn("user", t) for t in turns]],
                question="How many bikes do I currently own?",
                expected=expected_count,
            )
        )

    abstention_prompts = [
        (
            [
                "I upgraded my old 10-gallon tank, where my betta fish lives.",
                "I added decorations to my 20-gallon tank to create more hiding places for the fish.",
            ],
            "How many fish are in my 30-gallon tank?",
        ),
        (
            [
                "I changed my Wi-Fi router yesterday.",
                "The network name now appears as HomeOffice5G.",
            ],
            "What is my Wi-Fi password?",
        ),
        (
            [
                "I bought new running shoes this week.",
                "They are light and comfortable for long runs.",
            ],
            "What is my shoe size?",
        ),
        (
            [
                "I renewed my passport last month.",
                "The process took about two weeks.",
            ],
            "What is my passport number?",
        ),
        (
            [
                "I had a blood test done this morning.",
                "The doctor said to drink more water daily.",
            ],
            "What is my blood type?",
        ),
        (
            [
                "I replaced my office chair last week.",
                "The new one has better lumbar support.",
            ],
            "What brand is my office chair?",
        ),
        (
            [
                "I switched from coffee to tea in the evenings.",
                "It helped me sleep better.",
            ],
            "How many cups of tea do I drink daily?",
        ),
        (
            [
                "I backed up my laptop yesterday.",
                "I also cleaned up old files.",
            ],
            "What is my laptop serial number?",
        ),
        (
            [
                "I visited my dentist this month.",
                "The checkup was routine.",
            ],
            "What clinic address did I visit?",
        ),
        (
            [
                "I updated my CV this weekend.",
                "I added my latest project details.",
            ],
            "What is my private phone number?",
        ),
        (
            [
                "I moved my bookshelf to the living room.",
                "Now there is more space in my office.",
            ],
            "How many books do I own in total?",
        ),
    ]
    for idx, (turns, question) in enumerate(abstention_prompts, start=1):
        categories.append(
            Example(
                case_id=f"abstention-{idx}",
                category="abstention",
                sessions=[[Turn("user", t) for t in turns]],
                question=question,
                expected=ABSTAIN,
            )
        )

    temporal_cases = [
        ("2025-01-22", "2025-06-25", "5"),
        ("2025-03-10", "2025-07-11", "4"),
        ("2024-12-18", "2025-04-18", "4"),
        ("2025-02-01", "2025-02-28", "0"),
        ("2024-08-05", "2025-01-06", "5"),
        ("2025-01-15", "2025-03-14", "1"),
        ("2025-01-15", "2025-03-15", "2"),
        ("2024-11-30", "2025-02-28", "2"),
        ("2024-07-01", "2025-07-01", "12"),
        ("2025-05-20", "2025-06-19", "0"),
        ("2025-05-20", "2025-06-20", "1"),
    ]
    for idx, (visit_date, current_date, expected_months) in enumerate(temporal_cases, start=1):
        categories.append(
            Example(
                case_id=f"temporal-reasoning-{idx}",
                category="temporal-reasoning",
                sessions=[
                    [
                        Turn("user", f"On {visit_date}, I went to the Science Museum with a friend."),
                        Turn("user", f"Current date is {current_date}."),
                    ]
                ],
                question="How many months have passed since my last museum visit with a friend?",
                expected=expected_months,
            )
        )

    multi_session_cases = [
        (
            [
                [Turn("user", "I own a Korg B1 keyboard.")],
                [Turn("user", "I own a Fender Stratocaster guitar.")],
                [Turn("user", "I own a Yamaha FG800 guitar.")],
                [Turn("user", "I own a Pearl Export drum set."), Turn("user", "I sold my Pearl Export drum set.")],
            ],
            "3",
        ),
        (
            [
                [Turn("user", "I own a Casio PX-S1100 keyboard.")],
                [Turn("user", "I own a Gibson Les Paul guitar.")],
                [Turn("user", "I sold my Casio PX-S1100 keyboard.")],
            ],
            "1",
        ),
        (
            [
                [Turn("user", "I own a Roland FP-10 piano.")],
                [Turn("user", "I own a Yamaha P-45 piano.")],
                [Turn("user", "I own a Martin D-28 guitar.")],
            ],
            "3",
        ),
        (
            [
                [Turn("user", "I own a Nord Stage keyboard.")],
                [Turn("user", "I own a Tama Imperialstar drum set.")],
                [Turn("user", "I sold my Nord Stage keyboard.")],
                [Turn("user", "I bought a Kawai ES120 piano.")],
            ],
            "2",
        ),
        (
            [
                [Turn("user", "I own a Korg SV-2 keyboard.")],
                [Turn("user", "I own a Fender Jazzmaster guitar.")],
                [Turn("user", "I own a Gretsch Catalina drum set.")],
                [Turn("user", "I sold my Fender Jazzmaster guitar.")],
            ],
            "2",
        ),
        (
            [
                [Turn("user", "I own a Yamaha P-125 keyboard.")],
                [Turn("user", "I own a Gibson SG guitar.")],
                [Turn("user", "I sold my Yamaha P-125 keyboard.")],
                [Turn("user", "I own a Roland TD-07 drum set.")],
            ],
            "2",
        ),
        (
            [
                [Turn("user", "I own a Kawai MP11 piano.")],
                [Turn("user", "I own a PRS Custom 24 guitar.")],
                [Turn("user", "I own a Ludwig Accent drum set.")],
                [Turn("user", "I sold my Ludwig Accent drum set.")],
                [Turn("user", "I sold my PRS Custom 24 guitar.")],
            ],
            "1",
        ),
        (
            [
                [Turn("user", "I own a Casio Privia keyboard.")],
                [Turn("user", "I sold my Casio Privia keyboard.")],
                [Turn("user", "I own a Fender Precision bass guitar.")],
            ],
            "1",
        ),
        (
            [
                [Turn("user", "I own a Nord Piano keyboard.")],
                [Turn("user", "I own a Yamaha Revstar guitar.")],
                [Turn("user", "I own a Pearl Roadshow drum set.")],
                [Turn("user", "I own a Korg Kronos keyboard.")],
                [Turn("user", "I sold my Yamaha Revstar guitar.")],
            ],
            "3",
        ),
        (
            [
                [Turn("user", "I own a Roland Fantom keyboard.")],
                [Turn("user", "I own a Ibanez RG guitar.")],
                [Turn("user", "I sold my Roland Fantom keyboard.")],
                [Turn("user", "I sold my Ibanez RG guitar.")],
            ],
            "0",
        ),
        (
            [
                [Turn("user", "I own a Yamaha YC keyboard.")],
                [Turn("user", "I own a Epiphone Casino guitar.")],
                [Turn("user", "I own a Tama Club-JAM drum set.")],
                [Turn("user", "I sold my Tama Club-JAM drum set.")],
                [Turn("user", "I own a Korg Minilogue keyboard.")],
            ],
            "3",
        ),
    ]
    for idx, (sessions, expected_count) in enumerate(multi_session_cases, start=1):
        categories.append(
            Example(
                case_id=f"multi-session-{idx}",
                category="multi-session",
                sessions=sessions,
                question="How many musical instruments do I currently own?",
                expected=expected_count,
            )
        )

    return categories


def run_example(example: Example, db_path: Path, memory_enabled: bool = True) -> tuple[str, bool]:
    if db_path.exists():
        db_path.unlink()

    for session in example.sessions:
        for turn in session:
            memory = PersistentMemory(db_path, enabled=memory_enabled)
            engine = MemoryEngine(memory)
            engine.ingest(turn.role, turn.text)

    memory = PersistentMemory(db_path, enabled=memory_enabled)
    engine = MemoryEngine(memory)
    prediction = engine.answer(example.question)
    return prediction, prediction.strip().lower() == example.expected.strip().lower()


def run_suite(examples: list[Example], db_path: Path, memory_enabled: bool) -> tuple[dict[str, dict[str, int]], int]:
    per_category: dict[str, dict[str, int]] = {}
    correct_total = 0

    mode = "ENABLED" if memory_enabled else "DISABLED"
    print(f"Running LongMemEval mini benchmark ({len(examples)} cases) | memory={mode}")
    print(f"Persistent store: {db_path}")
    print("-" * 72)

    for example in examples:
        prediction, ok = run_example(example, db_path, memory_enabled=memory_enabled)
        if db_path.exists():
            db_path.unlink()

        stats = per_category.setdefault(example.category, {"total": 0, "correct": 0})
        stats["total"] += 1
        stats["correct"] += int(ok)
        correct_total += int(ok)

        marker = "OK" if ok else "FAIL"
        print(
            f"[{marker}] {example.case_id} | category={example.category} | "
            f"expected={example.expected!r} | predicted={prediction!r}"
        )

    print("-" * 72)
    for category in sorted(per_category.keys()):
        stats = per_category[category]
        acc = (stats["correct"] / stats["total"]) * 100.0
        print(f"{category:24s} -> {stats['correct']}/{stats['total']} ({acc:5.1f}%)")

    total_acc = (correct_total / len(examples)) * 100.0
    print("-" * 72)
    print(f"TOTAL -> {correct_total}/{len(examples)} ({total_acc:5.1f}%)")
    return per_category, correct_total


def run_persistence_smoke_test(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()

    memory = PersistentMemory(db_path, enabled=True)
    engine = MemoryEngine(memory)
    session_turns = [
        Turn("user", "I own a Korg B1 keyboard."),
        Turn("user", "I own a Fender Stratocaster guitar."),
        Turn("user", "I sold my Korg B1 keyboard."),
    ]
    for t in session_turns:
        engine.ingest(t.role, t.text)

    if not db_path.exists():
        raise RuntimeError("persistence smoke test failed: memory file was not created")

    reloaded = PersistentMemory(db_path, enabled=True)
    instruments = reloaded.get_set("instruments")
    print("-" * 72)
    print("PERSISTENCE SMOKE TEST")
    print(f"memory file exists: {db_path.exists()}")
    print(f"instruments after reload: {instruments}")
    print(f"expected instrument count after reload: 1 | actual: {len(instruments)}")
    if len(instruments) != 1:
        raise RuntimeError("persistence smoke test failed: session state did not persist")
    if db_path.exists():
        db_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval mini benchmark")
    parser.add_argument(
        "--engine",
        choices=["proxy", "llm"],
        default="llm",
        help="benchmark engine: proxy (rule-based) or llm (real llama-cli A/B)",
    )
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="disable external memory writes/reads (ablation baseline)",
    )
    parser.add_argument(
        "--compare-disabled",
        action="store_true",
        help="run memory-enabled and memory-disabled back-to-back",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="skip file persistence smoke test",
    )
    parser.add_argument(
        "--llama-cli",
        default=str(Path(__file__).resolve().parents[2] / "build-funes-lfm2only/bin/llama-cli"),
        help="path to llama-cli (used when --engine llm)",
    )
    parser.add_argument(
        "--model",
        default=str(Path.home() / ".models/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"),
        help="path to GGUF model (used when --engine llm)",
    )
    parser.add_argument(
        "--cases-per-category",
        type=int,
        default=11,
        help="cases per category for llm engine",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="limit cases for llm engine",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=32,
        help="max generated tokens for llm engine",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="sampling temperature for llm engine",
    )
    parser.add_argument(
        "--semantic-strength",
        type=float,
        default=None,
        help="semantic memory strength for llm engine",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    db_path = root / "persistent_memory.json"
    examples = build_examples()

    if args.engine == "llm":
        cmd = [
            "python3",
            str(root / "llama_cli_ab_test.py"),
            "--llama-cli", args.llama_cli,
            "--model", args.model,
            "--cases-per-category", str(args.cases_per_category),
            "--n-predict", str(args.n_predict),
            "--temp", str(args.temp),
        ]
        if args.max_cases is not None:
            cmd += ["--max-cases", str(args.max_cases)]
        if args.semantic_strength is not None:
            cmd += ["--semantic-strength", str(args.semantic_strength)]

        print("Running TRUE LLM benchmark via llama-cli A/B...", flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return

    if not args.skip_smoke_test:
        run_persistence_smoke_test(db_path)

    if args.compare_disabled:
        _, enabled_correct = run_suite(examples, db_path, memory_enabled=True)
        print("\n")
        _, disabled_correct = run_suite(examples, db_path, memory_enabled=False)
        print("-" * 72)
        print("COMPARISON")
        print(f"memory enabled : {enabled_correct}/{len(examples)}")
        print(f"memory disabled: {disabled_correct}/{len(examples)}")
        return

    run_suite(examples, db_path, memory_enabled=not args.disable_memory)


if __name__ == "__main__":
    main()
