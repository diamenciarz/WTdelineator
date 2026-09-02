# Evaluation datasets

Notebooks that download a **chosen subset** of LUDB, STAFF III, and PTB-XL, convert each record to the Graph_Visualizer signal pair (`.npy` + `.pkl`), and write dataset-native labels.

Run them from the `WTdelineator` repo root (or from this folder). Already-downloaded PhysioNet files and already-processed `.npy`/`.pkl`/`.json` triples are skipped.

```
data/evaluation/
  raw/<dataset>/          # WFDB / csv as published
  processed/<dataset>/
    signals/*.npy + *.pkl # (12, n_samples), fs + channel names
    labels/*.json         # labels in that dataset's own scheme
    figures/              # plots produced by the notebooks
    index.csv
```

| Notebook | Why it is in the thesis | What the labels are |
|---|---|---|
| `Download_LUDB.ipynb` | Widest labelled abnormality set + gold-standard wave bounds | Multi-label diagnoses + per-lead P/QRS/T onsets, peaks, offsets |
| `Download_STAFF_III.ipynb` | Long recordings of balloon occlusion in LAD/RCA/LCX/LM | Phase, artery, D0/D1/D2, `.event` inflation/deflation/injection times |
| `Download_PTB_XL.ipynb` | ~21k records, ~50/50 sex, ~9.5k NORM | SCP-ECG codes, diagnostic super/subclasses, fold, demographics |

STAFF III (~3.2 GB) and PTB-XL (~3 GB at 500 Hz) should always be subsetted. LUDB is ~24 MB and can be downloaded in full (`N_RECORDS = None`).

Existing copies under `data/LU_DB`, `data/staff_III`, and `data/ptb-xl` are used as fallbacks so the notebooks do not re-fetch files you already have.

STAFF III spreadsheet parsing needs `openpyxl` (`pip install openpyxl`).
