# Trained perception models

| File | What | Use on |
|---|---|---|
| `detector.pt` | 1-class "litter" detector, YOLO26n at 640 px | laptops, Macs (PyTorch) |
| `detector_int8_openvino_model/` | same detector, INT8 OpenVINO | the robot's Intel NUC (launch default) |
| `classifier.pt` | crop classifier: bottle / can / cup / paper / other, YOLO26n-cls at 224 px | laptops, Macs |
| `classifier_int8_openvino_model/` | same classifier, INT8 OpenVINO | the robot (launch default) |

These are trained on TACO, Indoor Waste, COCO negatives and TrashNet. They were trained on a
Kaggle T4 with [`training/kaggle/train.py`](../../../training/kaggle/train.py). The detector is
the `detector_all` run, the classifier is the all-sources run, and both were trained on 2026-11-10.
Test scores at confidence 0.35:

- **Detector:** TACO AP50 0.51, recall 0.43. On indoor floors (Indoor Waste) AP50 0.98, recall 0.96.
- **Classifier:** top-1 accuracy 0.77.

The full comparison is in [`training/README.md`](../../../training/README.md#results).

To replace them, train new ones (see `training/README.md`) and copy `best.pt` and its
`best_int8_openvino_model/` here under these names. Keep the `_openvino_model` ending, because
Ultralytics needs it to load the folder.

Ultralytics models are licensed under AGPL-3.0. The training data licenses are listed in
[`training/datasets.yaml`](../../../training/datasets.yaml) and the training README.
