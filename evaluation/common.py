"""Shared helpers for evaluation-dataset download, conversion, and plotting.

Processed signals match Graph_Visualizer: ``.npy`` of shape ``(n_leads, n_samples)``
plus a companion ``.pkl`` with ``{"fs": int, "channels": list[str]}``.
"""

from __future__ import annotations

import ast
import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wfdb
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DATA = REPO_ROOT / "data" / "evaluation"
RAW_ROOT = EVAL_DATA / "raw"
PROCESSED_ROOT = EVAL_DATA / "processed"

STANDARD_12_LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

LEAD_ALIASES = {
    "i": "I",
    "ii": "II",
    "iii": "III",
    "avr": "aVR",
    "avl": "aVL",
    "avf": "aVF",
    "AVR": "aVR",
    "AVL": "aVL",
    "AVF": "aVF",
    "dI": "I",
    "dII": "II",
    "dIII": "III",
    "DI": "I",
    "DII": "II",
    "DIII": "III",
}

LUDB_LEAD_ANN_EXTS = [
    "i",
    "ii",
    "iii",
    "avr",
    "avl",
    "avf",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
]

LUDB_DIAGNOSIS_FIELDS = [
    "Rhythm",
    "Electric axis of the heart",
    "Conduction abnormalities",
    "Extrasystolies",
    "Hypertrophies",
    "Cardiac pacing",
    "Ischemia",
    "Non-specific repolarization abnormalities",
    "Other states",
]


def dataset_dirs(name: str) -> tuple[Path, Path]:
    """Return ``(raw_dir, processed_dir)`` for a dataset slug."""
    raw = RAW_ROOT / name
    processed = PROCESSED_ROOT / name
    for path in (
        raw,
        processed / "signals",
        processed / "labels",
        processed / "figures",
    ):
        path.mkdir(parents=True, exist_ok=True)
    return raw, processed


# ---------------------------------------------------------------------------
# Lead geometry
# ---------------------------------------------------------------------------


def canonicalize_lead_name(name: str) -> str:
    key = str(name).strip()
    if key in LEAD_ALIASES:
        return LEAD_ALIASES[key]
    lowered = key.lower()
    if lowered in LEAD_ALIASES:
        return LEAD_ALIASES[lowered]
    if re.fullmatch(r"v[1-6]", lowered):
        return lowered.upper()
    return key


def to_12_lead(
    signal: np.ndarray,
    channel_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Reorder / derive a 12-lead matrix in ``STANDARD_12_LEADS`` order.

    Accepts 9-lead STAFF III (precordials + I/II/III) or already-complete 12-lead.
    """
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal, got shape {signal.shape}")

    if signal.shape[0] not in {9, 12} and signal.shape[1] in {9, 12}:
        signal = signal.T

    by_name: dict[str, np.ndarray] = {}
    for index, raw_name in enumerate(channel_names):
        by_name[canonicalize_lead_name(raw_name)] = np.asarray(signal[index], dtype=np.float64)

    if "I" in by_name and "II" in by_name:
        i_lead = by_name["I"]
        ii_lead = by_name["II"]
        by_name.setdefault("III", ii_lead - i_lead)
        by_name.setdefault("aVR", -(i_lead + ii_lead) / 2.0)
        by_name.setdefault("aVL", i_lead - ii_lead / 2.0)
        by_name.setdefault("aVF", ii_lead - i_lead / 2.0)

    missing = [name for name in STANDARD_12_LEADS if name not in by_name]
    if missing:
        raise ValueError(f"Cannot build 12-lead ECG; missing {missing} (have {sorted(by_name)})")

    stacked = np.vstack([by_name[name] for name in STANDARD_12_LEADS])
    return stacked, list(STANDARD_12_LEADS)


# ---------------------------------------------------------------------------
# Processed artefact I/O (Graph_Visualizer contract)
# ---------------------------------------------------------------------------


def save_signal_pair(
    dest_stem: Path,
    signal_12ch: np.ndarray,
    fs: int,
    channels: Sequence[str],
) -> tuple[Path, Path]:
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    npy_path = dest_stem.with_suffix(".npy")
    pkl_path = dest_stem.with_suffix(".pkl")
    np.save(npy_path, np.asarray(signal_12ch, dtype=np.float64))
    with open(pkl_path, "wb") as handle:
        pickle.dump({"fs": int(fs), "channels": list(channels)}, handle)
    return npy_path, pkl_path


def save_label_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def processed_exists(processed_dir: Path, record_id: str) -> bool:
    stem = processed_dir / "signals" / record_id
    labels = processed_dir / "labels" / f"{record_id}.json"
    return stem.with_suffix(".npy").is_file() and stem.with_suffix(".pkl").is_file() and labels.is_file()


def write_index(processed_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = processed_dir / "index.csv"
    frame = pd.DataFrame(rows)
    if path.exists() and not frame.empty:
        existing = pd.read_csv(path)
        combined = pd.concat([existing, frame], ignore_index=True)
        id_col = "record_id" if "record_id" in combined.columns else combined.columns[0]
        combined = combined.drop_duplicates(subset=[id_col], keep="last")
        combined.to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Download (skip files that already exist)
# ---------------------------------------------------------------------------


def physionet_file_url(db: str, version: str, relative: str) -> str:
    return f"https://physionet.org/files/{db}/{version}/{relative.replace(chr(92), '/')}"


def already_downloaded(dest: Path) -> bool:
    return dest.is_file() and dest.stat().st_size > 0


def download_files(
    db: str,
    files: Sequence[str],
    dest_root: Path,
    *,
    fallback_dirs: Sequence[Path] = (),
    version: str | None = None,
) -> dict[str, str]:
    """Copy or download PhysioNet files. Existing dest files are left untouched.

    ``files`` are paths relative to the PhysioNet project root, e.g. ``data/1.dat``.
    Returns a status map: ``downloaded`` / ``skipped`` / ``copied`` / ``failed``.
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    missing: list[str] = []

    for relative in files:
        rel = Path(relative.replace("\\", "/"))
        dest = dest_root / rel
        if already_downloaded(dest):
            status[relative] = "skipped"
            continue

        copied = False
        for fallback in fallback_dirs:
            source = fallback / rel
            if already_downloaded(source):
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                status[relative] = "copied"
                copied = True
                break
        if copied:
            continue
        missing.append(relative)

    if missing:
        # wfdb preserves the relative path under dest_root.
        try:
            wfdb.dl_files(db, str(dest_root), list(missing), keep_subdirs=True, overwrite=False)
            for relative in missing:
                dest = dest_root / Path(relative.replace("\\", "/"))
                status[relative] = "downloaded" if already_downloaded(dest) else "failed"
        except Exception:
            # Fallback: direct HTTPS, still skip files that appeared mid-loop.
            for relative in missing:
                dest = dest_root / Path(relative.replace("\\", "/"))
                if already_downloaded(dest):
                    status[relative] = "skipped"
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                if version is None:
                    status[relative] = "failed"
                    continue
                url = physionet_file_url(db, version, relative)
                try:
                    urlretrieve(url, dest)
                    status[relative] = "downloaded" if already_downloaded(dest) else "failed"
                except Exception:
                    status[relative] = "failed"

    return status


def download_single_file(url: str, dest: Path, *, force: bool = False) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if already_downloaded(dest) and not force:
        return "skipped"
    urlretrieve(url, dest)
    return "downloaded"


def summarize_status(status: dict[str, str]) -> str:
    counts = pd.Series(status).value_counts().to_dict() if status else {}
    parts = [f"{key}={value}" for key, value in sorted(counts.items())]
    return ", ".join(parts) if parts else "nothing to do"


# ---------------------------------------------------------------------------
# LUDB
# ---------------------------------------------------------------------------


def ludb_record_files(record_id: str) -> list[str]:
    """PhysioNet-relative files for one LUDB record (``data/1`` …)."""
    stem = f"data/{record_id}"
    files = [f"{stem}.dat", f"{stem}.hea"]
    files.extend(f"{stem}.{ext}" for ext in LUDB_LEAD_ANN_EXTS)
    return files


def parse_ludb_comments(comments: Sequence[str] | None) -> dict[str, list[str]]:
    parsed = {field: [] for field in LUDB_DIAGNOSIS_FIELDS}
    age = None
    sex = None
    if not comments:
        return {"age": age, "sex": sex, "diagnoses": parsed}  # type: ignore[return-value]

    for raw in comments:
        line = raw.strip().lstrip("#").strip()
        if line.lower().startswith("<age>"):
            age = line.split(":", 1)[-1].strip()
            continue
        if line.lower().startswith("<sex>"):
            sex = line.split(":", 1)[-1].strip()
            continue
        if line.lower().startswith("<diagnoses>"):
            continue
        for field in LUDB_DIAGNOSIS_FIELDS:
            if line.lower().startswith(field.lower() + ":"):
                value = line.split(":", 1)[1].strip().rstrip(".")
                if value:
                    parsed[field].append(value)
                break
        else:
            # Header lines like "Left ventricular hypertrophy." without a field prefix.
            if line and not line.lower().startswith("<"):
                lowered = line.lower()
                if "hypertrophy" in lowered or "overload" in lowered:
                    parsed["Hypertrophies"].append(line.rstrip("."))
                elif "block" in lowered or "hemiblock" in lowered:
                    parsed["Conduction abnormalities"].append(line.rstrip("."))
                elif "pacing" in lowered or "pacemaker" in lowered:
                    parsed["Cardiac pacing"].append(line.rstrip("."))
                elif "stemi" in lowered or "ischemia" in lowered or "scar" in lowered:
                    parsed["Ischemia"].append(line.rstrip("."))
                elif "repolarization" in lowered:
                    parsed["Non-specific repolarization abnormalities"].append(line.rstrip("."))
                elif "extrasystol" in lowered or "pvc" in lowered or "pac" in lowered:
                    parsed["Extrasystolies"].append(line.rstrip("."))
                elif "early repolarization" in lowered:
                    parsed["Other states"].append(line.rstrip("."))

    return {"age": age, "sex": sex, "diagnoses": parsed}


def parse_ludb_lead_annotation(ann: Any) -> dict[str, list[int]]:
    """Pair ``(``, peak, ``)`` triples into P / QRS / T onsets, peaks, offsets."""
    out = {
        "P_onsets": [],
        "P_peaks": [],
        "P_offsets": [],
        "R_onsets": [],
        "R_peaks": [],
        "R_offsets": [],
        "T_onsets": [],
        "T_peaks": [],
        "T_offsets": [],
    }
    samples = list(np.asarray(ann.sample).tolist())
    symbols = list(ann.symbol)
    peak_map = {
        "p": ("P_onsets", "P_peaks", "P_offsets"),
        "N": ("R_onsets", "R_peaks", "R_offsets"),
        "t": ("T_onsets", "T_peaks", "T_offsets"),
    }
    index = 0
    while index + 2 < len(symbols):
        if (
            symbols[index] == "("
            and symbols[index + 2] == ")"
            and symbols[index + 1] in peak_map
        ):
            onset_key, peak_key, offset_key = peak_map[symbols[index + 1]]
            out[onset_key].append(int(samples[index]))
            out[peak_key].append(int(samples[index + 1]))
            out[offset_key].append(int(samples[index + 2]))
            index += 3
        else:
            index += 1
    return out


def load_ludb_delineation(record_path: Path) -> dict[str, dict[str, list[int]]]:
    delineation: dict[str, dict[str, list[int]]] = {}
    for ext in LUDB_LEAD_ANN_EXTS:
        ann_file = record_path.with_suffix(f".{ext}")
        if not ann_file.is_file():
            continue
        try:
            ann = wfdb.rdann(str(record_path), ext)
        except Exception:
            continue
        delineation[canonicalize_lead_name(ext)] = parse_ludb_lead_annotation(ann)
    return delineation


# ---------------------------------------------------------------------------
# STAFF III
# ---------------------------------------------------------------------------

STAFF_HEADER_ROW = 9
STAFF_BI_SLOTS = ("BI1", "BI2", "BI3", "BI4", "BI5")
STAFF_PHASE_SLOTS = ("BR", "BC1", "BC2", "PC1", "PC2", "PR1", "PR2") + STAFF_BI_SLOTS


def staff_record_id(raw: Any) -> str | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    match = re.fullmatch(r"(\d+)([a-zA-Z])", text)
    if not match:
        return text.lower()
    return f"{int(match.group(1)):03d}{match.group(2).lower()}"


def normalize_staff_artery(text: str | None) -> str | None:
    if not text:
        return None
    lowered = str(text).lower()
    if "lad" in lowered:
        return "LAD"
    if "rca" in lowered:
        return "RCA"
    if "circ" in lowered or "lcx" in lowered:
        return "LCX"
    if "left main" in lowered or re.search(r"\blm\b", lowered):
        return "LM"
    return str(text).strip()


def _parse_d012(cell: Any) -> tuple[float | None, float | None, float | None]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None, None, None
    parts = [p.strip() for p in str(cell).split(";")]
    values: list[float | None] = []
    for part in parts[:3]:
        try:
            values.append(float(part))
        except ValueError:
            values.append(None)
    while len(values) < 3:
        values.append(None)
    return values[0], values[1], values[2]


def _parse_injection_times(cell: Any) -> list[float]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    times = []
    for part in str(cell).replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            times.append(float(part))
        except ValueError:
            continue
    return times


def parse_staff_annotations(xlsx_path: Path) -> pd.DataFrame:
    """One row per recording file mentioned in the official spreadsheet."""
    raw = pd.read_excel(xlsx_path, header=None)
    header = [str(value).strip() if pd.notna(value) else "" for value in raw.iloc[STAFF_HEADER_ROW].tolist()]
    body = raw.iloc[STAFF_HEADER_ROW + 1 :].copy()
    body.columns = header
    body = body.rename(columns={"#": "patient_id", "Age": "age", "Sex": "sex"})

    recordings: list[dict[str, Any]] = []
    for _, row in body.iterrows():
        if pd.isna(row.get("patient_id")):
            continue
        patient_id = int(row["patient_id"])
        age = row.get("age")
        sex = str(row.get("sex")).strip().upper() if pd.notna(row.get("sex")) else None

        for slot in STAFF_PHASE_SLOTS:
            record_id = staff_record_id(row.get(slot))
            if record_id is None:
                continue
            if slot.startswith("BI"):
                artery = row.get(f"{slot}:Occluded artery")
                d0, d1, d2 = _parse_d012(row.get(f"{slot}:D0;D1;D2"))
                injections = _parse_injection_times(row.get(f"{slot}:Injection time(s)"))
                phase = "inflation"
            elif slot.startswith("BR"):
                artery = None
                d0 = d1 = d2 = None
                injections = []
                phase = "baseline_room"
            elif slot.startswith("BC"):
                artery = None
                d0 = d1 = d2 = None
                injections = []
                phase = "baseline_cathlab"
            elif slot.startswith("PC"):
                artery = None
                d0 = d1 = d2 = None
                injections = []
                phase = "post_cathlab"
            else:
                artery = None
                d0 = d1 = d2 = None
                injections = []
                phase = "post_room"

            recordings.append(
                {
                    "patient_id": patient_id,
                    "age": age if pd.notna(age) else None,
                    "sex": sex,
                    "record_id": record_id,
                    "phase": phase,
                    "phase_slot": slot,
                    "occluded_artery_raw": None if pd.isna(artery) else str(artery),
                    "occluded_artery": normalize_staff_artery(None if pd.isna(artery) else str(artery)),
                    "d0_s": d0,
                    "d1_s": d1,
                    "d2_s": d2,
                    "injection_times_s": injections,
                    "location_flag": row.get("location"),
                }
            )
    return pd.DataFrame(recordings)


def staff_record_files(record_id: str, include_event: bool = True) -> list[str]:
    files = [f"data/{record_id}.dat", f"data/{record_id}.hea"]
    if include_event:
        files.append(f"data/{record_id}.event")
    return files


def load_staff_events(record_path: Path) -> list[dict[str, Any]]:
    event_path = record_path.with_suffix(".event")
    if not event_path.is_file():
        return []
    try:
        ann = wfdb.rdann(str(record_path), "event")
    except Exception:
        return []
    events = []
    aux = list(ann.aux_note or [])
    for i, sample in enumerate(ann.sample):
        label = aux[i].strip() if i < len(aux) and aux[i] else str(ann.symbol[i])
        events.append({"sample": int(sample), "label": label})
    return events


def staff_segments_from_events(
    n_samples: int,
    fs: int,
    events: list[dict[str, Any]],
    d0_s: float | None = None,
    d1_s: float | None = None,
) -> list[dict[str, Any]]:
    """Build pre / inflation / post windows from .event marks, falling back to D0/D1."""
    inflation_on = [e["sample"] for e in events if "inflation" in e["label"].lower() and "deflation" not in e["label"].lower()]
    deflation = [e["sample"] for e in events if "deflation" in e["label"].lower()]
    injections = [e["sample"] for e in events if "inject" in e["label"].lower()]

    segments: list[dict[str, Any]] = []
    if inflation_on:
        start = inflation_on[0]
        end = deflation[0] if deflation else n_samples
        if start > 0:
            segments.append({"name": "pre_inflation", "start": 0, "end": start})
        segments.append({"name": "inflation", "start": start, "end": min(end, n_samples)})
        if end < n_samples:
            segments.append({"name": "post_inflation", "start": end, "end": n_samples})
    elif d0_s is not None:
        start = int(round(d0_s * fs))
        duration = int(round((d1_s or 0) * fs))
        end = min(n_samples, start + duration) if duration else n_samples
        if start > 0:
            segments.append({"name": "pre_inflation", "start": 0, "end": min(start, n_samples)})
        segments.append({"name": "inflation", "start": min(start, n_samples), "end": end})
        if end < n_samples:
            segments.append({"name": "post_inflation", "start": end, "end": n_samples})
    else:
        segments.append({"name": "full_recording", "start": 0, "end": n_samples})

    return segments + [
        {"name": "contrast_injection", "start": sample, "end": sample, "kind": "instant"}
        for sample in injections
    ]


# ---------------------------------------------------------------------------
# PTB-XL
# ---------------------------------------------------------------------------


def parse_scp_codes(cell: Any) -> dict[str, float]:
    if isinstance(cell, dict):
        return {str(k): float(v) for k, v in cell.items()}
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return {}
    text = str(cell).strip()
    if not text:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            out[str(key)] = 0.0
    return out


def ptbxl_diagnostic_labels(
    scp_codes: dict[str, float],
    statements: pd.DataFrame,
) -> dict[str, Any]:
    """Map SCP codes to diagnostic super/subclasses using ``scp_statements.csv``."""
    table = statements.copy()
    if "Unnamed: 0" in table.columns:
        table = table.set_index("Unnamed: 0")
    superclasses: set[str] = set()
    subclasses: set[str] = set()
    diagnostic_codes: list[str] = []
    form_codes: list[str] = []
    rhythm_codes: list[str] = []

    for code in scp_codes:
        if code not in table.index:
            continue
        row = table.loc[code]
        if float(row.get("diagnostic") or 0) == 1.0:
            diagnostic_codes.append(code)
            klass = row.get("diagnostic_class")
            sub = row.get("diagnostic_subclass")
            if isinstance(klass, str) and klass:
                superclasses.add(klass)
            if isinstance(sub, str) and sub:
                subclasses.add(sub)
        if float(row.get("form") or 0) == 1.0:
            form_codes.append(code)
        if float(row.get("rhythm") or 0) == 1.0:
            rhythm_codes.append(code)

    is_norm = superclasses == {"NORM"} or (not superclasses and "NORM" in scp_codes)
    return {
        "scp_codes": scp_codes,
        "diagnostic_codes": diagnostic_codes,
        "form_codes": form_codes,
        "rhythm_codes": rhythm_codes,
        "diagnostic_superclasses": sorted(superclasses),
        "diagnostic_subclasses": sorted(subclasses),
        "is_norm": bool(is_norm),
    }


def ptbxl_record_files(filename_hr: str, include_100hz: bool = False) -> list[str]:
    files = [f"{filename_hr}.dat", f"{filename_hr}.hea"]
    if include_100hz:
        lr = filename_hr.replace("records500", "records100").replace("_hr", "_lr")
        files.extend([f"{lr}.dat", f"{lr}.hea"])
    return files


def select_ptbxl_subset(
    meta: pd.DataFrame,
    statements: pd.DataFrame,
    *,
    n_records: int,
    superclasses: Sequence[str] | None = None,
    strat_folds: Sequence[int] | None = None,
    sex: int | None = None,
    balanced_norm_abnormal: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    frame = meta.copy()
    parsed = frame["scp_codes"].map(parse_scp_codes)
    labels = parsed.map(lambda codes: ptbxl_diagnostic_labels(codes, statements))
    frame["diagnostic_superclasses"] = labels.map(lambda x: x["diagnostic_superclasses"])
    frame["is_norm"] = labels.map(lambda x: x["is_norm"])

    if strat_folds:
        frame = frame[frame["strat_fold"].isin(list(strat_folds))]
    if sex is not None:
        frame = frame[frame["sex"] == sex]
    if superclasses:
        wanted = set(superclasses)
        frame = frame[frame["diagnostic_superclasses"].map(lambda vals: bool(wanted & set(vals)))]

    rng = np.random.default_rng(seed)
    if balanced_norm_abnormal:
        half = max(1, n_records // 2)
        norm = frame[frame["is_norm"]]
        abn = frame[~frame["is_norm"]]
        n_norm = min(half, len(norm))
        n_abn = min(n_records - n_norm, len(abn))
        pick_norm = norm.sample(n=n_norm, random_state=int(rng.integers(0, 10**6))) if n_norm else norm
        pick_abn = abn.sample(n=n_abn, random_state=int(rng.integers(0, 10**6))) if n_abn else abn
        chosen = pd.concat([pick_norm, pick_abn], ignore_index=False)
    else:
        n_take = min(n_records, len(frame))
        chosen = frame.sample(n=n_take, random_state=int(rng.integers(0, 10**6))) if n_take else frame
    return chosen.sort_values("ecg_id")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _offset_leads(signal_12ch: np.ndarray, spacing: float | None = None) -> tuple[np.ndarray, float]:
    if spacing is None:
        spacing = float(np.nanpercentile(np.abs(signal_12ch), 95) * 2.8 + 0.5)
    offsets = np.arange(signal_12ch.shape[0])[::-1] * spacing
    return signal_12ch + offsets[:, None], spacing


def plot_12_lead_strip(
    signal_12ch: np.ndarray,
    fs: int,
    channels: Sequence[str],
    *,
    title: str,
    t_start: float = 0.0,
    t_end: float | None = None,
    ax=None,
):
    n_samples = signal_12ch.shape[1]
    t_end = (n_samples / fs) if t_end is None else t_end
    i0 = max(0, int(t_start * fs))
    i1 = min(n_samples, int(t_end * fs))
    time = np.arange(i0, i1) / fs
    stacked, spacing = _offset_leads(signal_12ch[:, i0:i1])
    created = False
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))
        created = True
    for row, name in enumerate(channels):
        ax.plot(time, stacked[row], color="black", lw=0.8)
        ax.text(time[0] if len(time) else 0.0, stacked[row, 0] + spacing * 0.15, name, fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlim(t_start, t_end)
    if created:
        return ax.figure, ax
    return ax


def plot_ludb_delineation(
    signal_12ch: np.ndarray,
    fs: int,
    channels: Sequence[str],
    delineation: dict[str, dict[str, list[int]]],
    *,
    lead: str = "II",
    title: str = "LUDB cardiologist marks",
):
    """Single-lead view with P / QRS / T spans from LUDB annotations."""
    if lead not in channels:
        lead = channels[1] if len(channels) > 1 else channels[0]
    idx = list(channels).index(lead)
    y = signal_12ch[idx]
    t = np.arange(len(y)) / fs
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, y, color="black", lw=0.9, label=lead)
    colors = {"P": "#1f77b4", "QRS": "#d62728", "T": "#ff7f0e"}
    marks = delineation.get(lead, {})
    for wave, (on_key, peak_key, off_key) in {
        "P": ("P_onsets", "P_peaks", "P_offsets"),
        "QRS": ("R_onsets", "R_peaks", "R_offsets"),
        "T": ("T_onsets", "T_peaks", "T_offsets"),
    }.items():
        ons = marks.get(on_key, [])
        peaks = marks.get(peak_key, [])
        offs = marks.get(off_key, [])
        for a, b in zip(ons, offs):
            ax.axvspan(a / fs, b / fs, color=colors[wave], alpha=0.18)
        ax.scatter([p / fs for p in peaks], [y[p] for p in peaks if p < len(y)],
                   color=colors[wave], s=18, zorder=3, label=wave)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    return fig, ax


STAFF_PHASE_COLORS = {
    "pre_inflation": "#9ecae1",
    "inflation": "#fc9272",
    "post_inflation": "#a1d99b",
    "full_recording": "#cccccc",
}
STAFF_PHASE_LABELS = {
    "pre_inflation": "pre inflation",
    "inflation": "inflation",
    "post_inflation": "post inflation",
    "full_recording": "full recording",
}


def load_processed_record(
    processed_dir: Path | str, record_id: str
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Load ``signals/{id}.npy`` + ``.pkl`` and ``labels/{id}.json``."""
    processed_dir = Path(processed_dir)
    npy_path = processed_dir / "signals" / f"{record_id}.npy"
    pkl_path = processed_dir / "signals" / f"{record_id}.pkl"
    label_path = processed_dir / "labels" / f"{record_id}.json"
    if not npy_path.is_file() or not pkl_path.is_file():
        raise FileNotFoundError(f"Missing processed signal for '{record_id}' in {processed_dir / 'signals'}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing labels for '{record_id}' in {processed_dir / 'labels'}")
    signal = np.load(npy_path)
    with open(pkl_path, "rb") as handle:
        meta = pickle.load(handle)
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    return signal, meta, labels


def format_staff_title(record_id: str, labels: dict[str, Any]) -> str:
    """Human-readable title; does not dump raw inflation dicts."""
    parts: list[str] = [f"STAFF III {record_id}"]
    patient = labels.get("patient_id")
    if patient is not None:
        parts.append(f"patient {patient}")
    phase = labels.get("phase")
    if phase:
        parts.append(str(phase).replace("_", " "))
    arteries: list[str] = []
    for inf in labels.get("inflations") or []:
        raw = inf.get("occluded_artery_raw") or inf.get("occluded_artery")
        if raw:
            text = str(raw)
            if text not in arteries:
                arteries.append(text)
    if arteries:
        parts.append(", ".join(arteries))
    duration = labels.get("duration_s")
    if duration is not None:
        parts.append(f"{float(duration):.0f} s")
    return " · ".join(parts)


def inflation_segments_from_events(
    n_samples: int, events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Paint every balloon-up / balloon-down pair, not only the first inflation."""
    ons = sorted(
        int(e["sample"])
        for e in events
        if "inflation" in str(e.get("label", "")).lower()
        and "deflation" not in str(e.get("label", "")).lower()
    )
    offs = sorted(
        int(e["sample"])
        for e in events
        if "deflation" in str(e.get("label", "")).lower()
    )
    if not ons:
        return []
    remaining = list(offs)
    pairs: list[tuple[int, int]] = []
    for start in ons:
        later = [t for t in remaining if t > start]
        end = later[0] if later else n_samples
        if later:
            remaining.remove(end)
        pairs.append((start, min(end, n_samples)))
    segs: list[dict[str, Any]] = []
    cursor = 0
    for start, end in pairs:
        if start > cursor:
            name = "pre_inflation" if cursor == 0 else "post_inflation"
            segs.append({"name": name, "start": cursor, "end": start})
        segs.append({"name": "inflation", "start": start, "end": end})
        cursor = end
    if cursor < n_samples:
        segs.append({"name": "post_inflation", "start": cursor, "end": n_samples})
    return segs


def plot_staff_12lead_phases(
    signal_12ch: np.ndarray,
    fs: int,
    channels: Sequence[str],
    segments: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    *,
    title: str = "STAFF III",
    display_hz: float = 100.0,
) -> go.Figure:
    """Interactive 12-lead ECG + phase strip. Zooming the x-axis updates every row."""
    events = events or []
    n_samples = signal_12ch.shape[1]
    event_segs = inflation_segments_from_events(n_samples, events)
    if event_segs:
        segments = event_segs
    n_leads = signal_12ch.shape[0]
    step = max(1, int(round(fs / float(display_hz))))
    t = np.arange(0, n_samples, step) / float(fs)
    duration_s = n_samples / float(fs)

    row_heights = [1.0] * n_leads + [0.45]
    fig = make_subplots(
        rows=n_leads + 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.008,
        row_heights=row_heights,
        subplot_titles=None,
    )

    for i, name in enumerate(channels):
        fig.add_trace(
            go.Scattergl(
                x=t,
                y=signal_12ch[i, ::step],
                mode="lines",
                line=dict(color="black", width=0.8),
                name=str(name),
                showlegend=False,
                hovertemplate=f"{name}: %{{y:.3f}} mV<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(title_text=str(name), title_standoff=0, row=i + 1, col=1, showticklabels=False, ticks="")

    seen_phases: set[str] = set()
    for seg in segments:
        if seg.get("kind") == "instant":
            continue
        key = str(seg.get("name", "full_recording"))
        t0 = float(seg["start"]) / float(fs)
        t1 = float(seg["end"]) / float(fs)
        label = STAFF_PHASE_LABELS.get(key, key.replace("_", " "))
        fig.add_trace(
            go.Bar(
                x=[t1 - t0],
                y=["phase"],
                base=[t0],
                orientation="h",
                marker=dict(color=STAFF_PHASE_COLORS.get(key, "#dddddd")),
                name=label,
                showlegend=key not in seen_phases,
                hovertemplate=f"{label}: {t0:.1f}–{t1:.1f} s<extra></extra>",
                offsetgroup=key,
            ),
            row=n_leads + 1,
            col=1,
        )
        seen_phases.add(key)

    fig.update_yaxes(
        title_text="phase",
        showticklabels=False,
        ticks="",
        row=n_leads + 1,
        col=1,
    )

    for event in events:
        x = float(event["sample"]) / float(fs)
        label = str(event.get("label", "event"))
        fig.add_vline(
            x=x,
            line_dash="dash",
            line_color="#54278f",
            line_width=1,
            annotation_text=label,
            annotation_position="top right",
            annotation_font_size=10,
            annotation_textangle=-90,
            row=1,
            col=1,
        )
        fig.add_vline(
            x=x,
            line_dash="dash",
            line_color="#54278f",
            line_width=1,
            row=n_leads + 1,
            col=1,
        )

    fig.update_xaxes(title_text="Time (s)", range=[0, duration_s], row=n_leads + 1, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=n_leads + 1, col=1)
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        barmode="overlay",
        height=70 * n_leads + 160,
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
        hovermode="x unified",
        bargap=0,
    )
    return fig


def plot_staff_record(
    record_id: str,
    processed_dir: Path | str | None = None,
    *,
    display_hz: float = 100.0,
    title: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Load a processed STAFF III record and show the interactive 12-lead + phase plot.

    Example::

        C.plot_staff_record("001c", PROC_DIR)
    """
    processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_ROOT / "staff_iii"
    signal, meta, labels = load_processed_record(processed_dir, record_id)
    fig = plot_staff_12lead_phases(
        signal,
        int(meta["fs"]),
        list(meta["channels"]),
        labels.get("segments") or [],
        events=labels.get("events") or [],
        title=title or format_staff_title(record_id, labels),
        display_hz=display_hz,
    )
    if show:
        fig.show()
    return fig


def plot_staff_timeline(
    signal_12ch: np.ndarray,
    fs: int,
    channels: Sequence[str],
    segments: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    lead: str = "V3",
    title: str = "STAFF III occlusion timeline",
):
    if lead not in channels:
        lead = "II" if "II" in channels else channels[0]
    idx = list(channels).index(lead)
    y = signal_12ch[idx]
    # Downsample for display of multi-minute records.
    step = max(1, int(fs / 50))
    t = np.arange(0, len(y), step) / fs
    y_ds = y[::step]
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    axes[0].plot(t, y_ds, color="black", lw=0.6)
    axes[0].set_ylabel(f"{lead} (mV)")
    axes[0].set_title(title)

    colors = {
        "pre_inflation": "#9ecae1",
        "inflation": "#fc9272",
        "post_inflation": "#a1d99b",
        "full_recording": "#cccccc",
    }
    for seg in segments:
        if seg.get("kind") == "instant":
            continue
        axes[1].axvspan(
            seg["start"] / fs,
            seg["end"] / fs,
            color=colors.get(seg["name"], "#dddddd"),
            alpha=0.9,
        )
    for event in events:
        axes[0].axvline(event["sample"] / fs, color="#54278f", ls="--", lw=1)
        axes[1].axvline(event["sample"] / fs, color="#54278f", ls="--", lw=1)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Time (s)")
    legend = [
        Patch(facecolor=color, label=name.replace("_", " "))
        for name, color in colors.items()
        if any(seg.get("name") == name for seg in segments)
    ]
    if legend:
        axes[1].legend(handles=legend, loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    return fig, axes


def plot_category_counts(
    counts: pd.Series,
    *,
    title: str,
    xlabel: str = "Records",
    color: str = "#3182bd",
):
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(counts))))
    ordered = counts.sort_values()
    ordered.plot.barh(ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig, ax


def plot_cooccurrence(binary: pd.DataFrame, *, title: str):
    cols = list(binary.columns)
    matrix = np.zeros((len(cols), len(cols)), dtype=int)
    values = binary.astype(int).to_numpy()
    for i, _ in enumerate(cols):
        for j, _ in enumerate(cols):
            matrix[i, j] = int(np.sum(values[:, i] & values[:, j]))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def save_figure(fig, processed_dir: Path, name: str) -> Path:
    dest = processed_dir / "figures" / f"{name}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    return dest
