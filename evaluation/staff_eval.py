"""STAFF III evaluation helpers: ground truth, positive-label matching, scoring.

Graph Visualizer functions are imported, not copied. This module only knows
STAFF labels and how to score detector output against balloon-up intervals.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

import common as C

# ---------------------------------------------------------------------------
# Territory (collapsed artery only — not prox/mid/distal)
# ---------------------------------------------------------------------------

ARTERY_LEADS: dict[str, tuple[str, ...]] = {
    "LAD": ("V1", "V2", "V3", "V4", "I", "aVL"),
    "RCA": ("II", "III", "aVF"),
    "LCX": ("I", "aVL", "V5", "V6"),
    "LM": ("I", "aVL", "V1", "V2", "V3", "V4", "V5", "V6", "aVR"),
}

# Substrings in diagnosis node ids / names → artery. Order: first match wins.
DIAGNOSIS_ARTERY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("anterior", "LAD"),
    ("lad", "LAD"),
    ("inferoposterior", "RCA"),
    ("inferior", "RCA"),
    ("rca", "RCA"),
    ("circumflex", "LCX"),
    ("lcx", "LCX"),
    ("_cx", "LCX"),
    ("left_main", "LM"),
    ("lm_", "LM"),
)


def default_positive_patterns() -> list[str]:
    """Default: any diagnosis whose id contains these (case-insensitive)."""
    return ["STEMI", "occlusion"]


@dataclass
class PositiveSpec:
    """Which graph outputs count as a detection.

    Edit in the notebook and pass into scoring. Patterns are case-insensitive
    substrings of diagnosis **or** rule node ids. Explicit names override.
    """

    diagnosis_names: list[str] | None = None
    rule_names: list[str] | None = None
    diagnosis_patterns: list[str] = field(default_factory=default_positive_patterns)
    rule_patterns: list[str] = field(default_factory=list)

    def matches_diagnosis(self, name: str) -> bool:
        if self.diagnosis_names is not None:
            return name in self.diagnosis_names
        return _matches_any(name, self.diagnosis_patterns)

    def matches_rule(self, name: str) -> bool:
        if self.rule_names is not None:
            return name in self.rule_names
        return _matches_any(name, self.rule_patterns)


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(pat.lower() in lowered for pat in patterns if pat)


def _feature_lead(attrs: dict[str, Any]) -> str | None:
    raw = attrs.get("ecg_feature")
    if isinstance(raw, dict):
        return raw.get("lead")
    if isinstance(raw, str) and raw.startswith("{"):
        import ast

        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return attrs.get("lead_scope")
        if isinstance(parsed, dict):
            return parsed.get("lead")
    return attrs.get("lead_scope")


def inventory_graph(graph) -> pd.DataFrame:
    """One row per Symptom / Rule / Diagnosis node."""
    rows = []
    for node, attrs in graph.nodes(data=True):
        rows.append(
            {
                "id": node,
                "type": attrs.get("type"),
                "description": (attrs.get("description") or "")[:160],
                "lead": _feature_lead(attrs),
            }
        )
    return pd.DataFrame(rows)


def listed_positives(graph, spec: PositiveSpec) -> dict[str, list[str]]:
    diagnoses = [
        n for n, a in graph.nodes(data=True) if a.get("type") == "Diagnosis" and spec.matches_diagnosis(n)
    ]
    rules = [n for n, a in graph.nodes(data=True) if a.get("type") == "Rule" and spec.matches_rule(n)]
    return {"diagnoses": sorted(diagnoses), "rules": sorted(rules)}


def artery_from_diagnosis(name: str) -> str | None:
    lowered = name.lower()
    for needle, artery in DIAGNOSIS_ARTERY_PATTERNS:
        if needle in lowered:
            return artery
    return None


def lead_from_symptom(name: str, trace: dict[str, Any] | None = None) -> str | None:
    if trace:
        lead = trace.get("lead")
        if isinstance(lead, str) and lead not in ("", "derived"):
            return lead
        feature = trace.get("ecg_feature") or {}
        if isinstance(feature, dict) and feature.get("lead"):
            return str(feature["lead"])
    match = re.search(r"lead[_\s]?([IV]+|aV[RLF]|V[1-6])", name, flags=re.I)
    if match:
        return C.canonicalize_lead_name(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Ground truth from .event marks (not labels.segments)
# ---------------------------------------------------------------------------


def balloon_up_intervals(
    labels: dict[str, Any],
    n_samples: int | None = None,
) -> list[tuple[int, int]]:
    """Half-open balloon-up sample ranges. Empty for baseline recordings."""
    n = int(n_samples if n_samples is not None else labels.get("n_samples") or 0)
    segs = C.inflation_segments_from_events(n, labels.get("events") or [])
    return [(int(s["start"]), int(s["end"])) for s in segs if s.get("name") == "inflation"]


def overlap_fraction(start: int, end: int, intervals: Sequence[tuple[int, int]]) -> float:
    length = max(0, int(end) - int(start))
    if length <= 0:
        return 0.0
    covered = 0
    for a, b in intervals:
        lo, hi = max(start, a), min(end, b)
        if hi > lo:
            covered += hi - lo
    return covered / length


def sample_in_intervals(sample: int, intervals: Sequence[tuple[int, int]]) -> bool:
    return any(a <= int(sample) < b for a, b in intervals)


def record_artery(labels: dict[str, Any]) -> str | None:
    if labels.get("occluded_artery"):
        return str(labels["occluded_artery"])
    for inf in labels.get("inflations") or []:
        artery = inf.get("occluded_artery")
        if artery:
            return str(artery)
    return None


# ---------------------------------------------------------------------------
# Load processed STAFF
# ---------------------------------------------------------------------------


def list_processed_records(processed_dir: Path | None = None) -> pd.DataFrame:
    proc = Path(processed_dir) if processed_dir is not None else C.PROCESSED_ROOT / "staff_iii"
    index = C.load_index(proc)
    if index.empty:
        ids = C.processed_record_ids(proc)
        index = pd.DataFrame({"record_id": ids})
    return index


def load_record(record_id: str, processed_dir: Path | None = None):
    proc = Path(processed_dir) if processed_dir is not None else C.PROCESSED_ROOT / "staff_iii"
    return C.load_processed_record(proc, record_id)


# ---------------------------------------------------------------------------
# Parse detector traces → per-beat detections
# ---------------------------------------------------------------------------


def _as_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, dict) and "values" in values:
        values = values.get("values")
    arr = np.atleast_1d(values)
    out: list[int] = []
    for item in arr.tolist() if hasattr(arr, "tolist") else list(arr):
        if item is None:
            continue
        try:
            if isinstance(item, (list, tuple)) and item:
                item = item[0]
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def detections_from_rules(
    rules: dict[str, Any] | None,
    spec: PositiveSpec,
) -> list[dict[str, Any]]:
    """Flatten evaluate_rules traces into detection events.

    One row per triggering symptom that contributed to a positive rule/diagnosis.
    """
    if not rules:
        return []
    positive_diagnoses = [d for d in rules.get("resulting_diagnoses") or [] if spec.matches_diagnosis(d)]
    positive_rules = [r for r in rules.get("satisfied_rules") or [] if spec.matches_rule(r)]
    # If the caller only configured diagnoses (the usual case), a rule is
    # positive when it produced a positive diagnosis.
    traced = (rules.get("trace") or {}).get("rules") or []
    events: list[dict[str, Any]] = []
    for block in traced:
        rule_name = block.get("rule_name")
        diagnoses = list(block.get("resulting_diagnoses") or [])
        rule_is_pos = spec.matches_rule(str(rule_name)) or any(spec.matches_diagnosis(d) for d in diagnoses)
        if not rule_is_pos and not positive_diagnoses and not positive_rules:
            continue
        if not rule_is_pos:
            continue
        for symptom in block.get("triggered_symptoms") or []:
            name = symptom.get("node_name") or ""
            beat_ids = _as_int_list(symptom.get("beat_ids"))
            indexes = _as_int_list(symptom.get("indexes"))
            lead = lead_from_symptom(name, symptom)
            for i, beat_id in enumerate(beat_ids or [None]):
                sample = indexes[i] if i < len(indexes) else (indexes[0] if indexes else None)
                if isinstance(sample, (list, tuple)):
                    sample = sample[0] if sample else None
                events.append(
                    {
                        "rule": rule_name,
                        "diagnoses": diagnoses,
                        "symptom": name,
                        "lead": lead,
                        "beat_id": beat_id,
                        "sample": int(sample) if sample is not None else None,
                        "arteries_pred": sorted(
                            {a for d in diagnoses if (a := artery_from_diagnosis(d))}
                        ),
                    }
                )
    if not events and (positive_diagnoses or positive_rules):
        events.append(
            {
                "rule": (positive_rules or [None])[0],
                "diagnoses": positive_diagnoses,
                "symptom": None,
                "lead": None,
                "beat_id": None,
                "sample": None,
                "arteries_pred": sorted({a for d in positive_diagnoses if (a := artery_from_diagnosis(d))}),
                "record_level_only": True,
            }
        )
    return events


def beat_sample(beat) -> int:
    leads = getattr(beat, "leads", {}) or {}
    for lead in ("II", "V2", "V5", "I"):
        feats = leads.get(lead) or {}
        if "R_peaks" in feats:
            return int(feats["R_peaks"])
    for feats in leads.values():
        if feats and "R_peaks" in feats:
            return int(feats["R_peaks"])
    boundary = beat.boundary
    return int(round((boundary.start + boundary.end) / 2))


def score_recording(
    *,
    record_id: str,
    labels: dict[str, Any],
    ir,
    rules: dict[str, Any] | None,
    spec: PositiveSpec,
    overlap_threshold: float = 0.5,
) -> dict[str, Any]:
    """Beat-level scores against balloon-up GT. Baselines have no positive intervals."""
    n_samples = int(labels.get("n_samples") or ir.signal_length)
    intervals = balloon_up_intervals(labels, n_samples)
    artery = record_artery(labels)
    phase = labels.get("phase")
    detections = detections_from_rules(rules, spec)
    detected_beats = {d["beat_id"] for d in detections if d.get("beat_id") is not None}
    detected_samples = [d["sample"] for d in detections if d.get("sample") is not None]
    record_level_hit = bool(detections)

    rows = []
    tp = fp = fn = tn = 0
    for beat in ir.beats:
        sample = beat_sample(beat)
        gt = sample_in_intervals(sample, intervals)
        pred = beat.beat_id in detected_beats
        if not pred and detected_beats == set() and record_level_hit:
            # No per-beat ids in the trace: fall back to sample overlap if we have indexes.
            pred = any(abs(int(s) - sample) < ir.fs * 0.4 for s in detected_samples) if detected_samples else False
        if gt and pred:
            tp += 1
            outcome = "tp"
        elif (not gt) and pred:
            fp += 1
            outcome = "fp"
        elif gt and not pred:
            fn += 1
            outcome = "fn"
        else:
            tn += 1
            outcome = "tn"
        rows.append(
            {
                "record_id": record_id,
                "beat_id": beat.beat_id,
                "sample": sample,
                "t_s": sample / float(ir.fs),
                "gt_positive": gt,
                "pred_positive": pred,
                "outcome": outcome,
                "phase": phase,
                "occluded_artery": artery,
            }
        )

    # Record-level (any detection vs any balloon-up in this file)
    gt_record = bool(intervals)
    pred_record = record_level_hit
    territory_hits = 0
    territory_n = 0
    if gt_record and pred_record and artery:
        territory_n = 1
        expected_leads = set(ARTERY_LEADS.get(artery, ()))
        fired_leads = {d["lead"] for d in detections if d.get("lead")}
        pred_arteries = {a for d in detections for a in (d.get("arteries_pred") or [])}
        in_leads = bool(fired_leads & expected_leads)
        in_dx = artery in pred_arteries
        territory_hits = int(in_leads or in_dx)

    latency_s: list[float | None] = []
    for start, _end in intervals:
        first = None
        for d in detections:
            sample = d.get("sample")
            if sample is not None and int(sample) >= start:
                first = (int(sample) - start) / float(ir.fs)
                break
        latency_s.append(first)

    return {
        "record_id": record_id,
        "phase": phase,
        "occluded_artery": artery,
        "n_beats": len(ir.beats),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_record": gt_record,
        "pred_record": pred_record,
        "territory_hit": territory_hits,
        "territory_n": territory_n,
        "latency_s": latency_s,
        "n_inflations": len(intervals),
        "detections": detections,
        "beat_table": pd.DataFrame(rows),
    }


def confusion_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    tn = sum(int(r["tn"]) for r in rows)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and (prec + rec)) else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Disk cache for expensive NeuroKit runs
# ---------------------------------------------------------------------------


def cache_path(cache_dir: Path, record_id: str, strategy: str = "NeurokitOnly") -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{record_id}__{strategy}.pkl"


def save_eval_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_eval_cache(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)
