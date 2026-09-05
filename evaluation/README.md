# Evaluation datasets

Notebooks that download a **chosen subset** of LUDB, STAFF III, PTB-XL, and the QT Database, convert each record to the Graph_Visualizer signal pair (`.npy` + `.pkl`), and write dataset-native labels.

Run them from the `WTdelineator` repo root (or from this folder). **Already-processed** `.npy` / `.pkl` / `.json` triples are skipped — the notebooks do not re-fetch those waveforms.

```
data/evaluation/
  catalogs/               # tiny metadata (kept)
  processed/<dataset>/
    signals/*.npy + *.pkl # (n_leads, n_samples), fs + channel names
    labels/*.json         # labels in that dataset's own scheme
    figures/
    index.csv
  raw/<dataset>/          # ephemeral WFDB; deleted after a successful convert
```

STAFF III recordings that you have not converted yet stay compressed in `data/staff_III`. When `Download_STAFF_III.ipynb` converts a record from that cache, it **deletes that record's WFDB** so the cache shrinks. If the file is not in the cache, it is downloaded temporarily, converted, and discarded.

| Notebook | Why it is in the thesis | Labels |
|---|---|---|
| `Download_LUDB.ipynb` | Gold-standard P/QRS/T on every lead + diagnoses | Multi-label diagnoses + per-lead bounds |
| `Download_STAFF_III.ipynb` | Balloon occlusion in LAD/RCA/LCX/LM | Phase, artery, D0/D1/D2, `.event` marks |
| `Download_PTB_XL.ipynb` | ~21k records, NORM vs disease, sex, folds | SCP-ECG codes, super/subclasses |
| `Download_QTDB.ipynb` | Classic delineation benchmark (2-lead, 250 Hz) | `q1c` / `q2c` P/QRS/T bounds. **Not padded to 12 leads.** |
| `MergeLocal.ipynb` | Convert leftovers already on disk | Same as the matching downloader |
| `DisplayDownloaded.ipynb` | Coverage vs each catalogue | — |
| `Evaluate_STAFF_III.ipynb` | Graph Visualizer rules vs balloon-up / artery | `staff_eval.py` + `evaluate_recording()` |
| `Cutting.ipynb` | Cut a processed STAFF window for Graph_Visualizer | — |

STAFF III spreadsheet parsing needs `openpyxl` (`pip install openpyxl`).

Plots are Plotly. One-liners after convert:

```python
C.plot_ludb_record("1", PROC_DIR)
C.plot_staff_record("039c", PROC_DIR)
C.plot_ptbxl_record("00400_hr", PROC_DIR)
C.plot_qtdb_record("sel100", PROC_DIR)
```
