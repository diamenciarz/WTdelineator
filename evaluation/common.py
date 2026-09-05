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
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DATA = REPO_ROOT / "data" / "evaluation"
RAW_ROOT = EVAL_DATA / "raw"
PROCESSED_ROOT = EVAL_DATA / "processed"
CATALOG_ROOT = EVAL_DATA / "catalogs"

# Local WFDB trees left over from earlier downloads. STAFF stays here as a
# shrinking cache (convert one record → delete that record's WFDB). LUDB/QT
# are converted in full by MergeLocal, then the trees are removed.
LOCAL_CACHES = {
    "ludb": REPO_ROOT / "data" / "LU_DB",
    "staff_iii": REPO_ROOT / "data" / "staff_III",
    "ptb_xl": REPO_ROOT / "data" / "ptb-xl",
    "qtdb": REPO_ROOT / "data" / "QT_DB",
}

PHYSIONET = {
    "ludb": ("ludb", "1.0.1"),
    "staff_iii": ("staffiii", "1.0.0"),
    "ptb_xl": ("ptb-xl", "1.0.3"),
    "qtdb": ("qtdb", "1.0.0"),
}

CATALOGUE_FALLBACK_N = {
    "ludb": 200,
    "qtdb": 105,
    "staff_iii": 521,
    "ptb_xl": 21799,
}

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


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size < 8:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_index(processed_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = processed_dir / "index.csv"
    incoming = pd.DataFrame(rows)
    existing = _read_csv_or_empty(path)
    if incoming.empty:
        return path
    if existing.empty:
        incoming.to_csv(path, index=False)
        return path
    combined = pd.concat([existing, incoming], ignore_index=True)
    id_col = "record_id" if "record_id" in combined.columns else combined.columns[0]
    combined = combined.drop_duplicates(subset=[id_col], keep="last")
    combined.to_csv(path, index=False)
    return path


def load_index(processed_dir: Path | str) -> pd.DataFrame:
    """Read ``index.csv``, or rebuild it from ``labels/*.json`` if the file is empty."""
    processed_dir = Path(processed_dir)
    index = _read_csv_or_empty(processed_dir / "index.csv")
    if not index.empty:
        return index
    rebuilt: list[dict[str, Any]] = []
    for label_path in sorted((processed_dir / "labels").glob("*.json")):
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {"record_id": payload.get("record_id", label_path.stem)}
        for key in (
            "patient_id",
            "phase",
            "fs",
            "n_samples",
            "duration_s",
            "age",
            "sex",
            "is_norm",
        ):
            if key in payload:
                row[key] = payload[key]
        inflations = payload.get("inflations") or []
        if inflations:
            row["occluded_artery"] = inflations[0].get("occluded_artery")
        rebuilt.append(row)
    index = pd.DataFrame(rebuilt)
    if not index.empty:
        index.to_csv(processed_dir / "index.csv", index=False)
    return index


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
# Plots (Plotly)
# ---------------------------------------------------------------------------

LUDB_WAVE_COLORS = {"P": "#1f77b4", "QRS": "#d62728", "T": "#ff7f0e"}
LUDB_WAVE_KEYS = {
    "P": ("P_onsets", "P_peaks", "P_offsets"),
    "QRS": ("R_onsets", "R_peaks", "R_offsets"),
    "T": ("T_onsets", "T_peaks", "T_offsets"),
}


def _time_window(
    n_samples: int, fs: int, t_start: float, t_end: float | None
) -> tuple[int, int, float]:
    end = (n_samples / float(fs)) if t_end is None else float(t_end)
    i0 = max(0, int(t_start * fs))
    i1 = min(n_samples, int(end * fs))
    return i0, i1, end


def _display_step(fs: int, display_hz: float) -> int:
    return max(1, int(round(float(fs) / float(display_hz))))


def _maybe_show(fig: go.Figure, show: bool) -> go.Figure:
    if show:
        fig.show()
    return fig


def plot_12_lead(
    signal_12ch: np.ndarray,
    fs: int,
    channels: Sequence[str],
    *,
    title: str = "",
    t_start: float = 0.0,
    t_end: float | None = None,
    display_hz: float = 100.0,
    show: bool = False,
) -> go.Figure:
    """Interactive 12-lead ECG. Zooming the x-axis updates every row."""
    n_samples = signal_12ch.shape[1]
    i0, i1, end = _time_window(n_samples, fs, t_start, t_end)
    step = _display_step(fs, display_hz)
    t = np.arange(i0, i1, step) / float(fs)
    n_leads = signal_12ch.shape[0]
    fig = make_subplots(
        rows=n_leads,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.008,
        row_heights=[1.0] * n_leads,
    )
    for i, name in enumerate(channels):
        fig.add_trace(
            go.Scattergl(
                x=t,
                y=signal_12ch[i, i0:i1:step],
                mode="lines",
                line=dict(color="black", width=0.8),
                name=str(name),
                showlegend=False,
                hovertemplate=f"{name}: %{{y:.3f}} mV<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(
            title_text=str(name),
            title_standoff=0,
            row=i + 1,
            col=1,
            showticklabels=False,
            ticks="",
        )
    fig.update_xaxes(title_text="Time (s)", range=[t_start, end], row=n_leads, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=n_leads, col=1)
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        height=70 * n_leads + 80,
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode="x unified",
    )
    return _maybe_show(fig, show)


def plot_ludb_delineation(
    signal_12ch: np.ndarray,
    fs: int,
    channels: Sequence[str],
    delineation: dict[str, dict[str, list[int]]],
    *,
    lead: str = "II",
    title: str = "LUDB cardiologist marks",
    display_hz: float = 100.0,
    show: bool = False,
) -> go.Figure:
    """Single-lead view with P / QRS / T spans from LUDB annotations."""
    if lead not in channels:
        lead = channels[1] if len(channels) > 1 else channels[0]
    idx = list(channels).index(lead)
    y = signal_12ch[idx]
    step = _display_step(fs, display_hz)
    t = np.arange(0, len(y), step) / float(fs)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=y[::step],
            mode="lines",
            line=dict(color="black", width=0.9),
            name=str(lead),
            hovertemplate=f"{lead}: %{{y:.3f}} mV<extra></extra>",
        )
    )
    marks = delineation.get(lead, {})
    for wave, (on_key, peak_key, off_key) in LUDB_WAVE_KEYS.items():
        color = LUDB_WAVE_COLORS[wave]
        ons = marks.get(on_key, [])
        peaks = marks.get(peak_key, [])
        offs = marks.get(off_key, [])
        for a, b in zip(ons, offs):
            if a is None or b is None:
                continue
            fig.add_vrect(
                x0=a / float(fs),
                x1=b / float(fs),
                fillcolor=color,
                opacity=0.18,
                line_width=0,
                layer="below",
            )
        peak_t = [p / float(fs) for p in peaks if p is not None and 0 <= p < len(y)]
        peak_y = [float(y[p]) for p in peaks if p is not None and 0 <= p < len(y)]
        fig.add_trace(
            go.Scatter(
                x=peak_t,
                y=peak_y,
                mode="markers",
                marker=dict(color=color, size=8),
                name=wave,
                hovertemplate=f"{wave} peak: %{{x:.3f}} s<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        xaxis_title="Time (s)",
        yaxis_title="mV",
        height=380,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.08))
    return _maybe_show(fig, show)


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


def format_ludb_title(record_id: str, labels: dict[str, Any]) -> str:
    parts: list[str] = [f"LUDB {record_id}"]
    sex = labels.get("sex")
    age = labels.get("age")
    demo = ", ".join(p for p in (sex, f"{age}y" if age not in (None, "") else None) if p)
    if demo:
        parts.append(demo)
    rhythm = (labels.get("diagnoses") or {}).get("Rhythm") or []
    if rhythm:
        parts.append("; ".join(str(item) for item in rhythm))
    return " · ".join(parts)


def format_ptbxl_title(record_id: str, labels: dict[str, Any]) -> str:
    parts: list[str] = [f"PTB-XL {record_id}"]
    sex = labels.get("sex_label")
    age = labels.get("age")
    age_text = None
    if age not in (None, ""):
        try:
            age_text = f"{int(float(age))}y"
        except (TypeError, ValueError):
            age_text = f"{age}y"
    demo = " ".join(str(p) for p in (sex, age_text) if p)
    if demo:
        parts.append(demo)
    supers = labels.get("diagnostic_superclasses") or []
    if supers:
        parts.append(", ".join(str(item) for item in supers))
    return " · ".join(parts)


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


def plot_ludb_record(
    record_id: str,
    processed_dir: Path | str | None = None,
    *,
    lead: str = "II",
    display_hz: float = 100.0,
    title: str | None = None,
    show: bool = True,
) -> tuple[go.Figure, go.Figure]:
    """Load a processed LUDB record and show 12-lead + cardiologist P/QRS/T marks.

    Example::

        C.plot_ludb_record("1", PROC_DIR)
    """
    processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_ROOT / "ludb"
    signal, meta, labels = load_processed_record(processed_dir, record_id)
    fs = int(meta["fs"])
    channels = list(meta["channels"])
    heading = title or format_ludb_title(record_id, labels)
    fig12 = plot_12_lead(
        signal, fs, channels, title=heading, display_hz=display_hz, show=False
    )
    fig_dx = plot_ludb_delineation(
        signal,
        fs,
        channels,
        labels.get("delineation") or {},
        lead=lead,
        title=f"{heading} — lead {lead} P / QRS / T",
        display_hz=display_hz,
        show=False,
    )
    if show:
        fig12.show()
        fig_dx.show()
    return fig12, fig_dx


def plot_ptbxl_record(
    record_id: str,
    processed_dir: Path | str | None = None,
    *,
    display_hz: float = 100.0,
    title: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Load a processed PTB-XL record and show the interactive 12-lead plot.

    Example::

        C.plot_ptbxl_record("00400_hr", PROC_DIR)
    """
    processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_ROOT / "ptb_xl"
    signal, meta, labels = load_processed_record(processed_dir, record_id)
    fig = plot_12_lead(
        signal,
        int(meta["fs"]),
        list(meta["channels"]),
        title=title or format_ptbxl_title(record_id, labels),
        display_hz=display_hz,
        show=False,
    )
    return _maybe_show(fig, show)


def plot_category_counts(
    counts: pd.Series | dict[Any, Any],
    *,
    title: str,
    xlabel: str = "Records",
    color: str = "#3182bd",
    show: bool = True,
) -> go.Figure:
    series = counts if isinstance(counts, pd.Series) else pd.Series(counts)
    ordered = series.sort_values()
    fig = go.Figure(
        go.Bar(
            x=ordered.to_numpy(),
            y=[str(idx) for idx in ordered.index],
            orientation="h",
            marker_color=color,
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        xaxis_title=xlabel,
        yaxis=dict(automargin=True),
        height=max(320, 28 * max(len(ordered), 1) + 80),
        margin=dict(l=20, r=20, t=50, b=40),
        showlegend=False,
    )
    return _maybe_show(fig, show)


def plot_count_panels(
    panels: Sequence[tuple[pd.Series, str] | tuple[pd.Series, str, str]],
    *,
    show: bool = True,
) -> go.Figure:
    """Side-by-side vertical bar charts, e.g. NORM vs sex in PTB-XL."""
    n = max(1, len(panels))
    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=[str(panel[1]) for panel in panels],
    )
    default_colors = ["#31a354", "#3182bd", "#e6550d", "#6a51a3"]
    for i, panel in enumerate(panels):
        series, _title = panel[0], panel[1]
        color = panel[2] if len(panel) > 2 else default_colors[i % len(default_colors)]
        fig.add_trace(
            go.Bar(
                x=[str(idx) for idx in series.index],
                y=series.to_numpy(),
                marker_color=color,
                showlegend=False,
                hovertemplate="%{x}: %{y}<extra></extra>",
            ),
            row=1,
            col=i + 1,
        )
    fig.update_layout(height=380, margin=dict(l=40, r=20, t=50, b=40))
    return _maybe_show(fig, show)


def plot_cooccurrence(binary: pd.DataFrame, *, title: str, show: bool = True) -> go.Figure:
    cols = list(binary.columns)
    values = binary.astype(int).to_numpy() if len(cols) else np.zeros((0, 0), dtype=int)
    matrix = values.T @ values if values.size else np.zeros((len(cols), len(cols)), dtype=int)
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=cols,
            y=cols,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="%{y} ∩ %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        height=max(420, 28 * max(len(cols), 1) + 120),
        margin=dict(l=20, r=20, t=50, b=80),
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
    )
    return _maybe_show(fig, show)


# ---------------------------------------------------------------------------
# JSON / skip / WFDB cleanup
# ---------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    """Turn NaN / numpy scalars into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def unprocessed_ids(
    processed_dir: Path | str,
    record_ids: Sequence[str],
    *,
    overwrite: bool = False,
) -> list[str]:
    processed_dir = Path(processed_dir)
    if overwrite:
        return [str(rid) for rid in record_ids]
    return [str(rid) for rid in record_ids if not processed_exists(processed_dir, str(rid))]


def processed_record_ids(processed_dir: Path | str) -> list[str]:
    processed_dir = Path(processed_dir)
    labels = processed_dir / "labels"
    if not labels.is_dir():
        return []
    return sorted(
        path.stem
        for path in labels.glob("*.json")
        if processed_exists(processed_dir, path.stem)
    )


def _npy_rel(processed_dir: Path, record_id: str) -> str:
    path = processed_dir / "signals" / f"{record_id}.npy"
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def delete_wfdb_stem(stem: Path, suffixes: Sequence[str]) -> list[Path]:
    """Unlink ``stem.suffix`` files that exist. ``suffixes`` include the dot (``.dat``)."""
    removed: list[Path] = []
    for suffix in suffixes:
        path = stem.with_suffix(suffix)
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


LUDB_WFDB_SUFFIXES = [".dat", ".hea"] + [f".{ext}" for ext in LUDB_LEAD_ANN_EXTS]
STAFF_WFDB_SUFFIXES = [".dat", ".hea", ".event"]
QTDB_WFDB_SUFFIXES = [".dat", ".hea", ".q1c", ".q2c"]
PTBXL_WFDB_SUFFIXES = [".dat", ".hea"]


# ---------------------------------------------------------------------------
# Catalogs (tiny metadata kept after raw waveforms are deleted)
# ---------------------------------------------------------------------------


def ensure_catalogs() -> dict[str, Path]:
    """Copy or download the small catalogue files into ``data/evaluation/catalogs/``."""
    CATALOG_ROOT.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    def _first_existing(dest: Path, sources: Sequence[Path]) -> Path | None:
        if dest.is_file() and dest.stat().st_size > 0:
            return dest
        for source in sources:
            if source.is_file() and source.stat().st_size > 0:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                return dest
        return dest if dest.is_file() else None

    ludb_csv = CATALOG_ROOT / "ludb.csv"
    found = _first_existing(
        ludb_csv,
        [RAW_ROOT / "ludb" / "ludb.csv", LOCAL_CACHES["ludb"] / "ludb.csv"],
    )
    if found is None:
        download_files("ludb", ["ludb.csv"], CATALOG_ROOT, version="1.0.1")
    out["ludb.csv"] = ludb_csv

    ludb_records = CATALOG_ROOT / "ludb_RECORDS"
    _first_existing(
        ludb_records,
        [RAW_ROOT / "ludb" / "RECORDS", LOCAL_CACHES["ludb"] / "RECORDS"],
    )
    if not ludb_records.is_file():
        download_files("ludb", ["RECORDS"], CATALOG_ROOT, version="1.0.1")
        downloaded = CATALOG_ROOT / "RECORDS"
        if downloaded.is_file() and not ludb_records.is_file():
            downloaded.replace(ludb_records)
    out["ludb_RECORDS"] = ludb_records

    staff_xlsx = CATALOG_ROOT / "STAFF-III-Database-Annotations.xlsx"
    found = _first_existing(
        staff_xlsx,
        [
            RAW_ROOT / "staff_iii" / "STAFF-III-Database-Annotations.xlsx",
            LOCAL_CACHES["staff_iii"] / "STAFF-III-Database-Annotations.xlsx",
        ],
    )
    if found is None:
        download_single_file(
            "https://physionet.org/files/staffiii/1.0.0/STAFF-III-Database-Annotations.xlsx",
            staff_xlsx,
        )
    out["staff_xlsx"] = staff_xlsx

    ptb_csv = CATALOG_ROOT / "ptbxl_database.csv"
    ptb_scp = CATALOG_ROOT / "scp_statements.csv"
    _first_existing(ptb_csv, [RAW_ROOT / "ptb_xl" / "ptbxl_database.csv"])
    _first_existing(ptb_scp, [RAW_ROOT / "ptb_xl" / "scp_statements.csv"])
    missing = [name for name, path in (("ptbxl_database.csv", ptb_csv), ("scp_statements.csv", ptb_scp)) if not path.is_file()]
    if missing:
        download_files("ptb-xl", missing, CATALOG_ROOT, version="1.0.3")
    out["ptbxl_database.csv"] = ptb_csv
    out["scp_statements.csv"] = ptb_scp

    qtdb_records = CATALOG_ROOT / "qtdb_RECORDS"
    found = _first_existing(qtdb_records, [LOCAL_CACHES["qtdb"] / "RECORDS"])
    if found is None:
        download_files("qtdb", ["RECORDS"], CATALOG_ROOT, version="1.0.0")
        downloaded = CATALOG_ROOT / "RECORDS"
        if downloaded.is_file() and not qtdb_records.is_file():
            downloaded.replace(qtdb_records)
    out["qtdb_RECORDS"] = qtdb_records
    return out


def load_ludb_catalogue() -> pd.DataFrame:
    ensure_catalogs()
    path = CATALOG_ROOT / "ludb.csv"
    frame = pd.read_csv(path)
    frame.columns = [c.strip() for c in frame.columns]
    frame["ID"] = frame["ID"].astype(str).str.strip()
    return frame


def load_staff_catalogue() -> pd.DataFrame:
    ensure_catalogs()
    return parse_staff_annotations(CATALOG_ROOT / "STAFF-III-Database-Annotations.xlsx")


def load_ptbxl_catalogue() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_catalogs()
    meta = pd.read_csv(CATALOG_ROOT / "ptbxl_database.csv")
    statements = pd.read_csv(CATALOG_ROOT / "scp_statements.csv", index_col=0)
    return meta, statements


def load_qtdb_record_ids() -> list[str]:
    ensure_catalogs()
    path = CATALOG_ROOT / "qtdb_RECORDS"
    if path.is_file():
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if ids:
            return ids
    cache = LOCAL_CACHES["qtdb"]
    return sorted(p.stem for p in cache.glob("*.hea"))


def catalogue_size(dataset: str) -> int:
    try:
        if dataset == "ludb":
            return len(load_ludb_catalogue())
        if dataset == "staff_iii":
            return int(load_staff_catalogue()["record_id"].nunique())
        if dataset == "ptb_xl":
            meta, _ = load_ptbxl_catalogue()
            return len(meta)
        if dataset == "qtdb":
            return len(load_qtdb_record_ids())
    except Exception:
        pass
    return CATALOGUE_FALLBACK_N.get(dataset, 0)


# ---------------------------------------------------------------------------
# Locate a WFDB stem (cache first, then evaluation/raw)
# ---------------------------------------------------------------------------


def list_cache_record_ids(dataset: str) -> list[str]:
    cache = LOCAL_CACHES[dataset]
    if dataset == "ludb":
        folder = cache / "data"
        return sorted(p.stem for p in folder.glob("*.hea")) if folder.is_dir() else []
    if dataset == "staff_iii":
        folder = cache / "data"
        return sorted(p.stem for p in folder.glob("*.hea")) if folder.is_dir() else []
    if dataset == "qtdb":
        return sorted(p.stem for p in cache.glob("*.hea")) if cache.is_dir() else []
    if dataset == "ptb_xl":
        ids = sorted(p.stem for p in cache.glob("*.hea")) if cache.is_dir() else []
        raw = RAW_ROOT / "ptb_xl"
        if raw.is_dir():
            ids.extend(p.stem for p in raw.rglob("*_hr.hea"))
        return sorted(set(ids))
    return []


def resolve_ludb_stem(record_id: str) -> tuple[Path | None, str | None]:
    cache = LOCAL_CACHES["ludb"] / "data" / record_id
    raw = RAW_ROOT / "ludb" / "data" / record_id
    if cache.with_suffix(".hea").is_file():
        return cache, "cache"
    if raw.with_suffix(".hea").is_file():
        return raw, "raw"
    return None, None


def resolve_staff_stem(record_id: str) -> tuple[Path | None, str | None]:
    cache = LOCAL_CACHES["staff_iii"] / "data" / record_id
    raw = RAW_ROOT / "staff_iii" / "data" / record_id
    if cache.with_suffix(".hea").is_file():
        return cache, "cache"
    if raw.with_suffix(".hea").is_file():
        return raw, "raw"
    return None, None


def resolve_qtdb_stem(record_id: str) -> tuple[Path | None, str | None]:
    cache = LOCAL_CACHES["qtdb"] / record_id
    raw = RAW_ROOT / "qtdb" / record_id
    if cache.with_suffix(".hea").is_file():
        return cache, "cache"
    if raw.with_suffix(".hea").is_file():
        return raw, "raw"
    return None, None


def resolve_ptbxl_stem(record_id: str, filename_hr: str | None = None) -> tuple[Path | None, str | None]:
    cache_flat = LOCAL_CACHES["ptb_xl"] / record_id
    if cache_flat.with_suffix(".hea").is_file():
        return cache_flat, "cache"
    if filename_hr:
        rel = Path(str(filename_hr).replace("\\", "/"))
        raw = RAW_ROOT / "ptb_xl" / rel
        if raw.with_suffix(".hea").is_file():
            return raw, "raw"
    raw_hits = list((RAW_ROOT / "ptb_xl").rglob(f"{record_id}.hea")) if (RAW_ROOT / "ptb_xl").is_dir() else []
    if raw_hits:
        return raw_hits[0].with_suffix(""), "raw"
    return None, None


# ---------------------------------------------------------------------------
# Convert one record → npy + pkl + json + index row
# ---------------------------------------------------------------------------


def convert_ludb_record(src_stem: Path, processed_dir: Path, record_id: str) -> dict[str, Any]:
    record = wfdb.rdrecord(str(src_stem))
    signal_12, channels = to_12_lead(record.p_signal.T, record.sig_name)
    save_signal_pair(processed_dir / "signals" / record_id, signal_12, record.fs, channels)
    parsed = parse_ludb_comments(record.comments)
    delineation = load_ludb_delineation(src_stem)
    n_qrs = sum(len(v.get("R_peaks", [])) for v in delineation.values())
    payload = {
        "dataset": "ludb",
        "record_id": record_id,
        "fs": int(record.fs),
        "n_samples": int(record.sig_len),
        "channels": channels,
        "age": parsed["age"],
        "sex": parsed["sex"],
        "diagnoses": parsed["diagnoses"],
        "delineation": delineation,
        "n_qrs_annotations_all_leads": n_qrs,
    }
    save_label_json(processed_dir / "labels" / f"{record_id}.json", payload)
    dx = parsed["diagnoses"]
    return {
        "record_id": record_id,
        "fs": int(record.fs),
        "n_samples": int(record.sig_len),
        "age": parsed["age"],
        "sex": parsed["sex"],
        "rhythm": "; ".join(dx["Rhythm"]),
        "axis": "; ".join(dx["Electric axis of the heart"]),
        "conduction": "; ".join(dx["Conduction abnormalities"]),
        "extrasystoles": "; ".join(dx["Extrasystolies"]),
        "hypertrophy": "; ".join(dx["Hypertrophies"]),
        "pacing": "; ".join(dx["Cardiac pacing"]),
        "ischemia": "; ".join(dx["Ischemia"]),
        "nonspecific_repol": "; ".join(dx["Non-specific repolarization abnormalities"]),
        "other": "; ".join(dx["Other states"]),
        "n_qrs_annotations_all_leads": n_qrs,
        "npy": _npy_rel(processed_dir, record_id),
    }


def convert_staff_record(
    src_stem: Path,
    processed_dir: Path,
    record_id: str,
    annotations: pd.DataFrame,
) -> dict[str, Any]:
    record = wfdb.rdrecord(str(src_stem))
    signal_12, channels = to_12_lead(record.p_signal.T, record.sig_name)
    save_signal_pair(processed_dir / "signals" / record_id, signal_12, record.fs, channels)
    rows = annotations[annotations["record_id"] == record_id]
    if rows.empty:
        raise ValueError(f"No STAFF III spreadsheet row for {record_id}")
    events = load_staff_events(src_stem)
    inflations = []
    for _, row in rows.iterrows():
        inflations.append(
            {
                "phase_slot": row["phase_slot"],
                "occluded_artery": jsonable(row["occluded_artery"]),
                "occluded_artery_raw": jsonable(row["occluded_artery_raw"]),
                "d0_s": jsonable(row["d0_s"]),
                "d1_s": jsonable(row["d1_s"]),
                "d2_s": jsonable(row["d2_s"]),
                "injection_times_s": jsonable(row["injection_times_s"]),
            }
        )
    first = rows.iloc[0]
    d0 = jsonable(first["d0_s"])
    d1 = jsonable(first["d1_s"])
    segments = staff_segments_from_events(record.sig_len, record.fs, events, d0, d1)
    payload = {
        "dataset": "staff_iii",
        "record_id": record_id,
        "patient_id": jsonable(first["patient_id"]),
        "age": jsonable(first["age"]),
        "sex": jsonable(first["sex"]),
        "fs": int(record.fs),
        "n_samples": int(record.sig_len),
        "duration_s": float(record.sig_len / record.fs),
        "channels": channels,
        "phase": first["phase"],
        "inflations": inflations,
        "events": events,
        "segments": segments,
        "header_comments": record.comments,
    }
    save_label_json(processed_dir / "labels" / f"{record_id}.json", payload)
    return {
        "record_id": record_id,
        "patient_id": jsonable(first["patient_id"]),
        "phase": first["phase"],
        "occluded_artery": jsonable(first["occluded_artery"]),
        "fs": int(record.fs),
        "n_samples": int(record.sig_len),
        "duration_s": round(record.sig_len / record.fs, 1),
        "n_events": len(events),
        "npy": _npy_rel(processed_dir, record_id),
    }


def convert_ptbxl_record(
    src_stem: Path,
    processed_dir: Path,
    record_id: str,
    row: pd.Series,
    statements: pd.DataFrame,
) -> dict[str, Any]:
    record = wfdb.rdrecord(str(src_stem))
    signal_12, channels = to_12_lead(record.p_signal.T, record.sig_name)
    save_signal_pair(processed_dir / "signals" / record_id, signal_12, record.fs, channels)
    scp = parse_scp_codes(row["scp_codes"])
    dx = ptbxl_diagnostic_labels(scp, statements)
    sex = jsonable(row.get("sex"))
    payload = {
        "dataset": "ptb_xl",
        "record_id": record_id,
        "ecg_id": int(row["ecg_id"]),
        "patient_id": int(row["patient_id"]),
        "fs": int(record.fs),
        "n_samples": int(record.sig_len),
        "channels": channels,
        "age": jsonable(row.get("age")),
        "sex": int(sex) if sex is not None else None,
        "sex_label": {0: "male", 1: "female"}.get(int(sex) if sex is not None else -1),
        "strat_fold": jsonable(row.get("strat_fold")),
        "report": None if pd.isna(row.get("report")) else str(row["report"]),
        "heart_axis": None if pd.isna(row.get("heart_axis")) else str(row["heart_axis"]),
        "infarction_stadium1": None if pd.isna(row.get("infarction_stadium1")) else str(row["infarction_stadium1"]),
        "validated_by_human": None if pd.isna(row.get("validated_by_human")) else bool(row["validated_by_human"]),
        **dx,
    }
    save_label_json(processed_dir / "labels" / f"{record_id}.json", payload)
    return {
        "record_id": record_id,
        "ecg_id": int(row["ecg_id"]),
        "patient_id": int(row["patient_id"]),
        "sex": payload["sex_label"],
        "age": payload["age"],
        "strat_fold": payload["strat_fold"],
        "is_norm": dx["is_norm"],
        "superclasses": ";".join(dx["diagnostic_superclasses"]),
        "subclasses": ";".join(dx["diagnostic_subclasses"]),
        "scp_codes": ";".join(f"{k}:{v}" for k, v in scp.items()),
        "npy": _npy_rel(processed_dir, record_id),
    }


def parse_qtdb_lead_annotation(ann: Any) -> dict[str, list[int | None]]:
    """Pair P/QRS/T marks. QTDB often stores T as ``t )`` with no opening ``(``."""
    out: dict[str, list[int | None]] = {
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
    while index < len(symbols):
        if (
            index + 2 < len(symbols)
            and symbols[index] == "("
            and symbols[index + 2] == ")"
            and symbols[index + 1] in peak_map
        ):
            onset_key, peak_key, offset_key = peak_map[symbols[index + 1]]
            out[onset_key].append(int(samples[index]))
            out[peak_key].append(int(samples[index + 1]))
            out[offset_key].append(int(samples[index + 2]))
            index += 3
        elif (
            index + 1 < len(symbols)
            and symbols[index] in peak_map
            and symbols[index + 1] == ")"
        ):
            onset_key, peak_key, offset_key = peak_map[symbols[index]]
            out[onset_key].append(None)
            out[peak_key].append(int(samples[index]))
            out[offset_key].append(int(samples[index + 1]))
            index += 2
        else:
            index += 1
    return out


def load_qtdb_delineation(record_path: Path, channel_names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """``{annotator: {lead: marks}}`` for ``q1c`` / ``q2c`` when those files exist."""
    by_annotator: dict[str, dict[str, Any]] = {}
    for ext in ("q1c", "q2c"):
        if not record_path.with_suffix(f".{ext}").is_file():
            continue
        try:
            ann = wfdb.rdann(str(record_path), ext)
        except Exception:
            continue
        per_lead: dict[str, dict[str, Any]] = {}
        chans = list(ann.chan) if ann.chan is not None else [0] * len(ann.sample)
        for chan_idx in sorted(set(int(c) for c in chans)):
            mask = [int(c) == chan_idx for c in chans]
            class _Ann:
                pass

            subset = _Ann()
            subset.sample = np.asarray(ann.sample)[np.asarray(mask)]
            subset.symbol = [sym for sym, keep in zip(ann.symbol, mask) if keep]
            lead = channel_names[chan_idx] if chan_idx < len(channel_names) else str(chan_idx)
            per_lead[lead] = parse_qtdb_lead_annotation(subset)
        by_annotator[ext] = per_lead
    return by_annotator


def qtdb_record_files(record_id: str, include_q2c: bool = True) -> list[str]:
    files = [f"{record_id}.dat", f"{record_id}.hea", f"{record_id}.q1c"]
    if include_q2c:
        files.append(f"{record_id}.q2c")
    return files


def convert_qtdb_record(src_stem: Path, processed_dir: Path, record_id: str) -> dict[str, Any]:
    record = wfdb.rdrecord(str(src_stem))
    signal = np.asarray(record.p_signal.T, dtype=np.float64)
    channels = [canonicalize_lead_name(name) for name in record.sig_name]
    save_signal_pair(processed_dir / "signals" / record_id, signal, record.fs, channels)
    delineation = load_qtdb_delineation(src_stem, channels)
    n_qrs = 0
    for per_lead in delineation.values():
        for marks in per_lead.values():
            n_qrs += len([p for p in marks.get("R_peaks", []) if p is not None])
    payload = {
        "dataset": "qtdb",
        "record_id": record_id,
        "fs": int(record.fs),
        "n_samples": int(record.sig_len),
        "n_leads": int(record.n_sig),
        "channels": channels,
        "comments": list(record.comments or []),
        "delineation": delineation,
        "n_qrs_annotations": n_qrs,
        "annotators": sorted(delineation),
    }
    save_label_json(processed_dir / "labels" / f"{record_id}.json", payload)
    return {
        "record_id": record_id,
        "fs": int(record.fs),
        "n_samples": int(record.sig_len),
        "n_leads": int(record.n_sig),
        "channels": ";".join(channels),
        "annotators": ";".join(sorted(delineation)),
        "n_qrs_annotations": n_qrs,
        "npy": _npy_rel(processed_dir, record_id),
    }


# ---------------------------------------------------------------------------
# Acquire (cache / raw / download) → convert → optionally consume WFDB
# ---------------------------------------------------------------------------


def _consume_if(flag: bool, stem: Path | None, suffixes: Sequence[str]) -> list[Path]:
    if not flag or stem is None:
        return []
    return delete_wfdb_stem(stem, suffixes)


def acquire_and_convert_ludb(
    record_id: str,
    *,
    overwrite: bool = False,
    consume: bool = True,
) -> dict[str, Any]:
    raw_dir, proc_dir = dataset_dirs("ludb")
    if processed_exists(proc_dir, record_id) and not overwrite:
        cache_stem, _ = resolve_ludb_stem(record_id)
        removed = _consume_if(consume, cache_stem, LUDB_WFDB_SUFFIXES)
        return {"status": "skipped", "record_id": record_id, "removed": [str(p) for p in removed]}

    stem, origin = resolve_ludb_stem(record_id)
    if stem is None:
        download_files("ludb", ludb_record_files(record_id), raw_dir, version="1.0.1")
        stem = raw_dir / "data" / record_id
        origin = "download"
    if not stem.with_suffix(".hea").is_file():
        return {"status": "failed", "record_id": record_id, "reason": "missing WFDB"}

    index_row = convert_ludb_record(stem, proc_dir, record_id)
    write_index(proc_dir, [index_row])
    removed = _consume_if(consume, stem, LUDB_WFDB_SUFFIXES)
    if consume and origin == "cache":
        removed.extend(delete_wfdb_stem(raw_dir / "data" / record_id, LUDB_WFDB_SUFFIXES))
    return {
        "status": "converted",
        "origin": origin,
        "record_id": record_id,
        "index_row": index_row,
        "removed": [str(p) for p in removed],
    }


def acquire_and_convert_staff(
    record_id: str,
    annotations: pd.DataFrame,
    *,
    overwrite: bool = False,
    consume: bool = True,
) -> dict[str, Any]:
    raw_dir, proc_dir = dataset_dirs("staff_iii")
    cache_stem = LOCAL_CACHES["staff_iii"] / "data" / record_id
    raw_stem = raw_dir / "data" / record_id
    if processed_exists(proc_dir, record_id) and not overwrite:
        removed = []
        removed.extend(_consume_if(consume, cache_stem, STAFF_WFDB_SUFFIXES))
        removed.extend(_consume_if(consume, raw_stem, STAFF_WFDB_SUFFIXES))
        return {"status": "skipped", "record_id": record_id, "removed": [str(p) for p in removed]}

    stem, origin = resolve_staff_stem(record_id)
    if stem is None:
        download_files(
            "staffiii",
            staff_record_files(record_id, include_event=True),
            raw_dir,
            version="1.0.0",
        )
        stem = raw_stem
        origin = "download"
    if not stem.with_suffix(".hea").is_file():
        return {"status": "failed", "record_id": record_id, "reason": "missing WFDB"}

    index_row = convert_staff_record(stem, proc_dir, record_id, annotations)
    write_index(proc_dir, [index_row])
    removed = _consume_if(consume, stem, STAFF_WFDB_SUFFIXES)
    if consume and origin == "cache":
        removed.extend(delete_wfdb_stem(raw_stem, STAFF_WFDB_SUFFIXES))
    return {
        "status": "converted",
        "origin": origin,
        "record_id": record_id,
        "index_row": index_row,
        "removed": [str(p) for p in removed],
    }


def acquire_and_convert_ptbxl(
    record_id: str,
    row: pd.Series,
    statements: pd.DataFrame,
    *,
    filename_hr: str | None = None,
    overwrite: bool = False,
    consume: bool = True,
) -> dict[str, Any]:
    raw_dir, proc_dir = dataset_dirs("ptb_xl")
    if processed_exists(proc_dir, record_id) and not overwrite:
        stem, _ = resolve_ptbxl_stem(record_id, filename_hr)
        removed = _consume_if(consume, stem, PTBXL_WFDB_SUFFIXES)
        return {"status": "skipped", "record_id": record_id, "removed": [str(p) for p in removed]}

    stem, origin = resolve_ptbxl_stem(record_id, filename_hr)
    if stem is None:
        rel = filename_hr or f"records500/{int(record_id.split('_')[0]) // 1000 * 1000:05d}/{record_id}"
        download_files("ptb-xl", [f"{rel}.dat", f"{rel}.hea"], raw_dir, version="1.0.3")
        stem, origin = resolve_ptbxl_stem(record_id, rel)
        origin = origin or "download"
    if stem is None or not stem.with_suffix(".hea").is_file():
        return {"status": "failed", "record_id": record_id, "reason": "missing WFDB"}

    index_row = convert_ptbxl_record(stem, proc_dir, record_id, row, statements)
    write_index(proc_dir, [index_row])
    removed = _consume_if(consume, stem, PTBXL_WFDB_SUFFIXES)
    return {
        "status": "converted",
        "origin": origin,
        "record_id": record_id,
        "index_row": index_row,
        "removed": [str(p) for p in removed],
    }


def acquire_and_convert_qtdb(
    record_id: str,
    *,
    overwrite: bool = False,
    consume: bool = True,
) -> dict[str, Any]:
    raw_dir, proc_dir = dataset_dirs("qtdb")
    if processed_exists(proc_dir, record_id) and not overwrite:
        stem, _ = resolve_qtdb_stem(record_id)
        removed = _consume_if(consume, stem, QTDB_WFDB_SUFFIXES)
        return {"status": "skipped", "record_id": record_id, "removed": [str(p) for p in removed]}

    stem, origin = resolve_qtdb_stem(record_id)
    if stem is None:
        download_files("qtdb", qtdb_record_files(record_id), raw_dir, version="1.0.0")
        stem = raw_dir / record_id
        origin = "download"
    if not stem.with_suffix(".hea").is_file():
        return {"status": "failed", "record_id": record_id, "reason": "missing WFDB"}

    index_row = convert_qtdb_record(stem, proc_dir, record_id)
    write_index(proc_dir, [index_row])
    removed = _consume_if(consume, stem, QTDB_WFDB_SUFFIXES)
    if consume and origin == "cache":
        removed.extend(delete_wfdb_stem(raw_dir / record_id, QTDB_WFDB_SUFFIXES))
    return {
        "status": "converted",
        "origin": origin,
        "record_id": record_id,
        "index_row": index_row,
        "removed": [str(p) for p in removed],
    }


def summarize_acquire(results: Sequence[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for item in results:
        counts[item.get("status", "unknown")] = counts.get(item.get("status", "unknown"), 0) + 1
    parts = [f"{key}={value}" for key, value in sorted(counts.items())]
    return ", ".join(parts) if parts else "nothing to do"


# ---------------------------------------------------------------------------
# Inventory (DisplayDownloaded)
# ---------------------------------------------------------------------------


def processed_bytes(processed_dir: Path) -> int:
    total = 0
    signals = processed_dir / "signals"
    if not signals.is_dir():
        return 0
    for path in signals.glob("*"):
        if path.is_file():
            total += path.stat().st_size
    labels = processed_dir / "labels"
    if labels.is_dir():
        for path in labels.glob("*.json"):
            total += path.stat().st_size
    return total


def dataset_inventory() -> pd.DataFrame:
    """One row per dataset: processed vs catalogue, plus leftover STAFF/LUDB/QT cache."""
    ensure_catalogs()
    rows = []
    for slug, title in (
        ("ludb", "LUDB"),
        ("staff_iii", "STAFF III"),
        ("ptb_xl", "PTB-XL"),
        ("qtdb", "QT Database"),
    ):
        proc = PROCESSED_ROOT / slug
        ids = processed_record_ids(proc)
        n_processed = len(ids)
        total = catalogue_size(slug)
        cache_ids = list_cache_record_ids(slug)
        nbytes = processed_bytes(proc)
        # Mean size of records already converted × catalogue length.
        # STAFF durations vary; this is a linear estimate, not a guarantee.
        if n_processed and nbytes:
            projected = nbytes / n_processed * total
        else:
            projected = float("nan")
        rows.append(
            {
                "dataset": slug,
                "title": title,
                "processed": n_processed,
                "catalogue": total,
                "fraction": (n_processed / total) if total else 0.0,
                "cache_wfdb": len(cache_ids),
                "processed_bytes": nbytes,
                "projected_processed_bytes": projected,
            }
        )
    return pd.DataFrame(rows)


def _format_coverage_pct(percent: float) -> str:
    if percent >= 99.95:
        return "100%"
    if percent >= 10:
        return f"{percent:.1f}%"
    if percent > 0:
        return f"{percent:.2f}%"
    return "0%"


def plot_download_fractions(inventory: pd.DataFrame | None = None, *, show: bool = True) -> go.Figure:
    frame = dataset_inventory() if inventory is None else inventory
    totals = frame["catalogue"].astype(float)
    pct = np.where(totals > 0, 100.0 * frame["processed"].astype(float) / totals, 0.0)
    remaining_pct = np.clip(100.0 - pct, 0.0, 100.0)
    x_labels = [
        f"{title} ({int(total)})"
        for title, total in zip(frame["title"], frame["catalogue"])
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="processed",
            x=x_labels,
            y=pct,
            marker_color="#31a354",
            hovertemplate="%{x}<br>%{y:.2f}% processed<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="not processed",
            x=x_labels,
            y=remaining_pct,
            marker_color="#d9d9d9",
            hovertemplate="%{x}<br>%{y:.2f}% remaining<extra></extra>",
        )
    )
    for label, value in zip(x_labels, pct):
        fig.add_annotation(
            x=label,
            y=101,
            text=_format_coverage_pct(float(value)),
            showarrow=False,
            font=dict(size=13),
            yanchor="bottom",
        )
    fig.update_layout(
        barmode="stack",
        title=dict(text="Processed records vs each dataset catalogue", x=0.01, xanchor="left"),
        yaxis=dict(title="Percent of catalogue", range=[0, 118], ticksuffix="%"),
        height=420,
        margin=dict(l=50, r=20, t=50, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    )
    return _maybe_show(fig, show)


def plot_qtdb_record(
    record_id: str,
    processed_dir: Path | str | None = None,
    *,
    annotator: str = "q1c",
    lead: str | None = None,
    display_hz: float = 100.0,
    title: str | None = None,
    show: bool = True,
) -> tuple[go.Figure, go.Figure]:
    """Native-lead plot plus cardiologist P/QRS/T marks from ``q1c`` / ``q2c``."""
    processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_ROOT / "qtdb"
    signal, meta, labels = load_processed_record(processed_dir, record_id)
    fs = int(meta["fs"])
    channels = list(meta["channels"])
    heading = title or f"QTDB {record_id} · {annotator}"
    fig_leads = plot_12_lead(signal, fs, channels, title=heading, display_hz=display_hz, show=False)
    per_annot = (labels.get("delineation") or {}).get(annotator) or {}
    if lead is None:
        lead = next(iter(per_annot), channels[0] if channels else "0")
    fig_dx = plot_ludb_delineation(
        signal,
        fs,
        channels,
        per_annot,
        lead=str(lead),
        title=f"{heading} — lead {lead} P / QRS / T",
        display_hz=display_hz,
        show=False,
    )
    if show:
        fig_leads.show()
        fig_dx.show()
    return fig_leads, fig_dx
