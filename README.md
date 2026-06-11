# CSE 144 Final Project — Transfer Learning Image Classifier

Fine-tunes two ImageNet-pretrained backbones — **ResNet-152** and **ViT-B/16** — on a
100-class image-classification dataset, and produces Kaggle submission files. See
[the project report](CSE%20144%20Final%20Project%20Report.pdf) for the experimental write-up
(backbone, pipeline, hyperparameters, results, and future work).

Kaggle competition: <https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project>

## Results

| Model      | Backbone params | Fine-tune | Best val. accuracy | Submission file              |
|------------|----------------:|-----------|-------------------:|------------------------------|
| ResNet-152 |          ~58.3M | Full      |          **66.2%** | `submission_resnet152.csv`   |
| ViT-B/16   |          ~85.8M | Full      |          **63.0%** | `submission_vit_b16.csv`     |

Both exceed the project's 60% baseline on the held-out validation split (20% of the labeled
training data). Validation accuracy is the best epoch's score; the corresponding weights are
saved automatically (see below). The ResNet figure is post-augmentation (rotation +
MixUp/CutMix); the ViT figure predates those changes and should be regenerated.

Four additional backbones (ResNet-50, ConvNeXt-Small, Swin-S, EfficientNet-B3) and a
softmax-averaging **ensemble** are available via `run_model.py` / `ensemble.py` — see
[the project report](CSE%20144%20Final%20Project%20Report.pdf) §10. They are trained, and the
ensemble is the best submission so far:

| Submission | Kaggle public score |
|------------|--------------------:|
| Swin-S | 0.709 |
| EfficientNet-B3 | 0.736 |
| ResNet-50 | 0.755 |
| ConvNeXt-Small | 0.755 |
| **Ensemble** | **0.764** |

Two inference-time / regularization options are also available: **head dropout** (`--dropout p`,
training) and **test-time augmentation** (`--tta` / `--tta-five-crop`, inference, no retraining
required) — see [the project report](CSE%20144%20Final%20Project%20Report.pdf) §7 and §10.4.

> The Kaggle test set is unlabeled, so per-epoch curves use a held-out **validation** split as
> the proxy for generalization. The public leaderboard reports only ~10% of the test set and
> should be used to verify submission format, not to estimate final accuracy.

## Repository structure

```
Kaggle Transfer Learning/
├── transfer_lib.py                    # SHARED pipeline: data, augmentation, train loop, inference
├── run_model.py                       # thin CLI: train/eval/infer ONE backbone via transfer_lib
├── ensemble.py                        # average softmax probs across trained models
├── resnet.ipynb                       # ResNet-152 notebook (same pipeline, cell-by-cell)
├── vit.ipynb                          # ViT-B/16 notebook   (same pipeline, cell-by-cell)
├── requirements.txt                   # Python dependencies (see PyTorch install note)
├── README.md                          # this file
├── CSE 144 Final Project Report.pdf   # project report (methods, results, future work)
├── .gitignore                         # excludes weights, dataset, caches (see below)
│
│   # generated artifacts (committed — small, document the results):
├── submission_*.csv                   # per-model + ensemble predictions   (inference / ensemble.py)
├── *_training_curves.png              # per-model loss/accuracy curves      (training)
│
│   # NOT in the repo (gitignored — see .gitignore):
├── <model>_finetuned.pth              # trained weights — too large for GitHub; on Google Drive (above)
└── ucsc-cse-144-spring-2026-final-project/   # Kaggle competition dataset — download from Kaggle
    ├── train/                         # 100 class folders "0".."99", ~10 images each (1079 total)
    ├── test/                          # 1036 unlabeled .jpg images (IDs 0..1035)
    └── sample_submission.csv          # template: the 1000 IDs (0..999) Kaggle scores
```

`<model>` is one of: `resnet152`, `resnet50`, `vit_b16`, `convnext_small`, `swin_s`,
`efficientnet_b3`. The `.pth` weights and the dataset directory are **gitignored** (too large /
redistributed from Kaggle); the trained weights are on Google Drive — see
[Trained model weights](#trained-model-weights) below.

> **Filename note.** `run_model.py` names its submissions `submission_<model>.csv` (e.g.
> `submission_resnet152.csv`). The two original notebooks instead write `submission_resnet.csv`
> and `submission_vit.csv`. Both produce identical predictions for the same trained checkpoint —
> only the output filename differs. Checkpoint names (`resnet152_finetuned.pth`,
> `vit_b16_finetuned.pth`) match in both paths, so `ensemble.py` finds them either way.

## Setup

Requires Python 3.10+ (developed on 3.14) and, for fast training, an NVIDIA GPU with a
CUDA-capable PyTorch build.

```bash
# 1) (recommended) create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/CMD)
# source .venv/bin/activate     # Linux / macOS

# 2) install PyTorch FIRST from the index matching your hardware:
#    GPU (CUDA 12.6 — what this project used):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#    or CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3) install the remaining dependencies
pip install -r requirements.txt
```

Verify the GPU is visible (optional but recommended):

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

The notebooks **auto-detect** CUDA and fall back to CPU automatically. CPU works but is
~10–50× slower per image.

### Data

The dataset directory `ucsc-cse-144-spring-2026-final-project/` (with `train/`, `test/`, and
`sample_submission.csv`) must sit next to the notebooks, exactly as shown above. It is the
data provided by the Kaggle competition.

## How to run

Each notebook is **self-contained** and runs top to bottom. Open it in Jupyter / VS Code and
**Run All**, or execute from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace resnet.ipynb
jupyter nbconvert --to notebook --execute --inplace vit.ipynb
```

### Or use the CLI (any of the six backbones)

`run_model.py` runs the same pipeline for any registered backbone — train, plot curves, reload
the best checkpoint, and write the submission:

```bash
python run_model.py resnet152            # primary CNN
python run_model.py vit_b16              # primary transformer
python run_model.py convnext_small       # modern CNN
python run_model.py swin_s               # hierarchical transformer
python run_model.py resnet50             # smaller/faster CNN
python run_model.py efficientnet_b3 --batch-size 8   # 300px input → use a smaller batch on 4 GB

python run_model.py convnext_small --infer-only      # skip training; reuse saved checkpoint
python run_model.py swin_s --epochs 15 --no-mix      # shorter run / disable MixUp/CutMix (ablation)
python run_model.py resnet50 --dropout 0.3           # regularize the head with dropout
python run_model.py convnext_small --infer-only --tta            # TTA: avg original + h-flip
python run_model.py convnext_small --infer-only --tta-five-crop  # TTA: 5-crop × flip (10 views)
```

Each run writes `<model>_finetuned.pth`, `submission_<model>.csv`, and
`<model>_training_curves.png`.

`--dropout p` (training) inserts `Dropout(p)` before the classifier head; the value is saved in
the checkpoint, so `--infer-only` and `ensemble.py` rebuild the matching head automatically.
`--tta` / `--tta-five-crop` (inference) average softmax probabilities over augmented views and
need no retraining — they work on any existing checkpoint.

### Ensemble

Once two or more models are trained, average their softmax probabilities into one submission:

```bash
python ensemble.py                                   # every model with a saved checkpoint
python ensemble.py convnext_small swin_s resnet152   # a specific (diverse) subset
python ensemble.py --weight-by-val-acc               # weight members by their val accuracy
python ensemble.py --tta                             # run every member under TTA, then average
```

This writes `submission_ensemble.csv`. Only include members that individually clear ~60% —
a weak member drags the average down — and prefer architecturally diverse members (a CNN + a
transformer beats two similar CNNs).

### What each notebook does, in order

1. **Config** — sets the random seed (`SEED = 42`), device, and all hyperparameters.
2. **Data** — loads `train/` with a **numeric** class mapping (folder `"k"` → label `k`;
   see the warning below), applies augmentation, and makes an 80/20 train/validation split.
3. **Model** — loads the pretrained backbone, sets `FINE_TUNE_MODE`, and swaps in a 100-way head.
4. **Train / evaluate** — fine-tunes for `EPOCHS`, printing per-epoch train/val loss & accuracy,
   and saves the **best** validation checkpoint to `*_finetuned.pth`.
5. **Training curves** — plots and saves `*_training_curves.png`.
6. **Inference** — loads the best checkpoint and writes predictions for every `test/` image to
   `submission_<model>.csv`.

### Training only

Run cells 1–4 (config → train). The best checkpoint is written automatically whenever
validation accuracy improves, so the saved `*_finetuned.pth` is always the best epoch.

### Inference only (reuse saved weights)

If `*_finetuned.pth` already exists, run cells 1–3 (config → model build) and then the final
inference cell — it reloads the checkpoint and regenerates the submission CSV without retraining.

## Reproducibility

- Fixed seed (`SEED = 42`) controls the train/val split and weight init of the new head.
- Hyperparameters are centralized in the config cell; see [the project report](CSE%20144%20Final%20Project%20Report.pdf) for the table.
- Exact dependency versions are pinned in `requirements.txt`.
- The best checkpoint stores `class_names` and `val_acc` alongside the weights, so inference
  is independent of how the training session ordered things.

## Key correctness notes

- **Label ordering (critical).** Class folders are named `"0"`…`"99"`. The default
  `torchvision.ImageFolder` sorts them *lexicographically* (`"0","1","10","11",…`), which maps
  class `"10"` to label 2 and scrambles every prediction — the spec warns this yields *no
  better than random guessing*. Both notebooks override this with a **numeric** mapping so
  folder `"k"` is always label `k`.
- **Submission length.** `test/` holds 1036 images (IDs 0–1035), but `sample_submission.csv`
  lists only the **1000** IDs Kaggle scores (0–999). The notebooks predict all 1036 by default
  (`PREDICT_ALL = True`) and verify that all 1000 scored IDs are covered; set `PREDICT_ALL =
  False` to emit exactly the template's 1000 rows.

## Hardware notes (Windows + single display GPU)

This project was developed on a GTX 960 (4 GB) that also drives the display. Two
platform-specific gotchas, both already handled in the notebooks:

- **`NUM_WORKERS = 0`** — multiprocessing DataLoader workers are unreliable on Windows
  (they re-import the notebook and can crash/hang). The dataset is tiny, so single-process
  loading is plenty fast.
- **TDR watchdog** — Windows kills any GPU kernel running longer than ~2 s on the display GPU
  (`cudaErrorLaunchTimeout`). If training crashes with that error, either raise
  `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\TdrDelay` (DWORD, seconds; e.g. 60) and
  reboot, or lower `BATCH_SIZE`.

## Trained model weights

The `.pth` checkpoints are too large for the repo (~1.1 GB for the six models), so they are
hosted on Google Drive:

**➡️ [Trained model weights (Google Drive)](https://drive.google.com/drive/folders/1F_UiFPEATg1A_n9l0p1VIcxEc76CsQVq?usp=sharing)** 

The bundle contains the six canonical checkpoints behind the submitted results:

| File | Model | Backbone params |
|------|-------|----------------:|
| `resnet152_finetuned.pth`      | ResNet-152      | ~58.3M |
| `resnet50_finetuned.pth`       | ResNet-50       | ~23.7M |
| `vit_b16_finetuned.pth`        | ViT-B/16        | ~85.8M |
| `convnext_small_finetuned.pth` | ConvNeXt-Small  | ~49.5M |
| `swin_s_finetuned.pth`         | Swin-S          | ~48.9M |
| `efficientnet_b3_finetuned.pth`| EfficientNet-B3 | ~10.8M |

To use them: download into this directory (next to `run_model.py`), then regenerate any
submission without retraining, e.g. `python run_model.py resnet50 --infer-only`, or rebuild the
ensemble with `python ensemble.py`. Each checkpoint stores its `class_names` and best `val_acc`,
so inference is self-describing.
