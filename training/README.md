# Perception training data

Builds two datasets from public sources, with no manual labeling:

1. **Detector** (`data/detector`): one class, `litter`. Every litter box from every
   source is merged into it. Results from detect-waste show 1-class litter
   detection is far more reliable than per-type detection (TACO: 61 vs 19 AP@0.5).
2. **Classifier** (`data/classifier`): crops labeled `bottle / can / cup / paper / other`.
   It runs on detector boxes and is only needed for sorting and per-class size checks.

How each source's categories map to these targets is in [`datasets.yaml`](datasets.yaml).

## Setup

```bash
cd training
uv sync --group dev
uv run pytest          # all tests run offline
```

## 1. Download the data

| Source | Used for | How to get it |
|---|---|---|
| TACO (official + unofficial, ~5.3k images) | detector + classifier | `uv run download-taco` (640px thumbnails; `--full-res` for originals) |
| Indoor COCO negatives (no litter-like objects) | detector backgrounds | `uv run download-coco-negatives` |
| TrashNet | classifier | `uv run download-trashnet` |
| [Indoor Waste Image Dataset](https://data.mendeley.com/datasets/dy7smxr93r/1) (indoor floors, made for sweeping robots) | detector + classifier (paper) | Manual: download from Mendeley and unzip into `data/raw/indoor_waste/`. **Check the license and then check the `indoor_waste` entry in `datasets.yaml`** (format and class names are unconfirmed). |
| Drinking Waste Classification (Kaggle, CC0) | detector + classifier (cans, bottles) | Manual: download from Kaggle and unzip into `data/raw/drinking_waste/`. Check the folder name and class order in `datasets.yaml`. |

Downloads resume where they left off, so re-running is safe. A source that hasn't been
downloaded is skipped with a warning, so you can build with whatever you have.

## 2. Build the datasets

```bash
uv run build-detector-dataset      # -> data/detector/data.yaml (Ultralytics format)
uv run build-classifier-dataset    # -> data/classifier/{train,val,test}/<class>/
```

Each build prints a per-source table and writes `report.json`. Check it:

- **missing**: images listed in annotations but not downloaded. Re-run the download.
- **mismatch**: the image file's orientation disagrees with its annotations. Those images are dropped so they don't add wrong labels.
- **empty**: images with no boxes (negatives, or images where every box was `ignore`).

Splits are assigned by a hash of each image's name. They stay the same across rebuilds, and all crops from one image stay in that image's split.

## 3. Train on a GPU (Ubuntu + NVIDIA)

Training runs natively with uv. On Linux, the PyTorch wheels from PyPI already include
CUDA, so you don't need Docker or a separate CUDA toolkit, just the NVIDIA driver.

```bash
nvidia-smi                                   # confirms the driver sees the GPU
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv isn't installed

cd training
uv sync --group dev --extra train
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints `False`, update the NVIDIA driver; the CUDA runtime inside the PyTorch wheel needs a recent one.

**Build the datasets on the GPU machine** (steps 1 and 2 above). The built datasets symlink to
absolute paths under `data/raw/`, so a dataset built on one machine doesn't work when copied to another.

```bash
# Train (~150 epochs max with early stopping; settings in configs/*.yaml)
uv run yolo cfg=configs/detector.yaml   project=$PWD/runs
uv run yolo cfg=configs/classifier.yaml project=$PWD/runs

# Evaluate on the held-out test split
uv run yolo val model=runs/detector/weights/best.pt   data=data/detector/data.yaml split=test
uv run yolo val model=runs/classifier/weights/best.pt data=data/classifier split=test

# Export INT8 OpenVINO models for the Stretch NUC's CPU
uv run yolo export model=runs/detector/weights/best.pt   format=openvino int8=True data=data/detector/data.yaml
uv run yolo export model=runs/classifier/weights/best.pt format=openvino int8=True data=data/classifier
```

Override any setting on the command line, e.g. `epochs=50 batch=32 name=detector_v2`.
After adding a dataset (Indoor Waste, Drinking Waste), rebuild and retrain with a new `name=`
so you can compare runs.

No NVIDIA GPU? On an Apple Silicon Mac add `device=mps batch=16`. It works, just slower.
