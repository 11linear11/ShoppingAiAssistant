#!/usr/bin/env python3
"""
Analyze pipeline LATENCY_SUMMARY log lines and print bottlenecks.

Usage:
  python scripts/analyze_latency_logs.py
  python scripts/analyze_latency_logs.py --log-dir logs --top 30
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LatencyEvent:
    file: str
    trace_id: str
    stage: str
    component: str
    total_ms: int
    breakdown_ms: dict[str, int]
    meta: dict[str, Any]


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    idx = q * (len(values) - 1)
    low = int(idx)
    high = min(low + 1, len(values) - 1)
    frac = idx - low
    return values[low] + (values[high] - values[low]) * frac


def parse_line(line: str, file_name: str) -> LatencyEvent | None:
    if "LATENCY_SUMMARY | " not in line:
        return None

    parts = line.split("│")
    if len(parts) < 4:
        return None

    trace_id = parts[1].strip()
    stage_token = parts[2].strip()
    stage = stage_token.split()[-1] if stage_token else "UNKNOWN"

    message = parts[3].strip()
    _, _, payload_raw = message.partition("|")
    payload_raw = payload_raw.strip()
    if not payload_raw:
        return None

    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        return None

    component = str(payload.get("component") or stage.lower())
    total_ms = int(payload.get("total_ms") or 0)

    breakdown_raw = payload.get("breakdown_ms") or {}
    breakdown_ms: dict[str, int] = {}
    if isinstance(breakdown_raw, dict):
        for key, value in breakdown_raw.items():
            try:
                breakdown_ms[str(key)] = int(value)
            except (TypeError, ValueError):
                continue

    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}

    return LatencyEvent(
        file=file_name,
        trace_id=trace_id,
        stage=stage,
        component=component,
        total_ms=total_ms,
        breakdown_ms=breakdown_ms,
        meta=meta,
    )


def discover_log_files(log_dir: Path) -> list[Path]:
    patterns = [
        str(log_dir / "pipeline-*.log"),
        str(log_dir / "pipeline-*.log.*"),
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob.glob(pattern))
    return sorted(set(files))


def load_events(files: list[Path]) -> list[LatencyEvent]:
    events: list[LatencyEvent] = []
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    event = parse_line(line, path.name)
                    if event:
                        events.append(event)
        except FileNotFoundError:
            continue
    return events


def print_component_summary(events: list[LatencyEvent]) -> None:
    by_component: dict[str, list[LatencyEvent]] = {}
    for event in events:
        by_component.setdefault(event.component, []).append(event)

    print("\nComponent Summary (ms)")
    print("component,count,avg,p50,p90,p95,max")
    for component, rows in sorted(
        by_component.items(),
        key=lambda item: sum(r.total_ms for r in item[1]) / max(len(item[1]), 1),
        reverse=True,
    ):
        values = sorted(r.total_ms for r in rows)
        avg = sum(values) / len(values)
        p50 = percentile(values, 0.50)
        p90 = percentile(values, 0.90)
        p95 = percentile(values, 0.95)
        vmax = max(values)
        print(
            f"{component},{len(values)},{avg:.1f},{p50:.1f},{p90:.1f},{p95:.1f},{vmax}"
        )


def print_slowest_events(events: list[LatencyEvent], top: int) -> None:
    print(f"\nTop {top} Slowest LATENCY_SUMMARY Events")
    print("total_ms,component,stage,trace_id,file,meta")
    for event in sorted(events, key=lambda e: e.total_ms, reverse=True)[:top]:
        meta_compact = json.dumps(event.meta, ensure_ascii=False, separators=(",", ":"))
        print(
            f"{event.total_ms},{event.component},{event.stage},{event.trace_id},{event.file},{meta_compact}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze pipeline latency summaries")
    parser.add_argument("--log-dir", default="logs", help="Directory containing pipeline logs")
    parser.add_argument("--top", type=int, default=20, help="Number of slow events to print")
    parser.add_argument(
        "--component",
        default="",
        help="Optional component substring filter (e.g., search.pipeline)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    files = discover_log_files(log_dir)
    if not files:
        print(f"No pipeline logs found in: {log_dir}")
        return 1

    events = load_events(files)
    if args.component:
        needle = args.component.strip().lower()
        events = [e for e in events if needle in e.component.lower()]

    if not events:
        print("No LATENCY_SUMMARY entries found.")
        return 1

    print(f"Loaded {len(events)} latency events from {len(files)} files.")
    print_component_summary(events)
    print_slowest_events(events, max(args.top, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
