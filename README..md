
# 🛠️ Construction Site Safety Object Detection

This repository contains three Jupyter Notebooks implementing object detection models using **YOLOv8** and **Faster R-CNN** for safety gear and hazard detection at construction sites.

## 📂 Repository Contents

| Notebook                   | Model             | Description                                                                 |
|---------------------------|-------------------|-----------------------------------------------------------------------------|
| `yolo-training.ipynb`     | YOLOv8            | Trains YOLOv8 on a construction safety dataset using transfer learning.    |
| `pretrained-yolo.ipynb`   | YOLOv8 (Pretrained) | Evaluates a pretrained YOLOv8 model on the same dataset and saves metrics. |
| `fast-rcnn.ipynb`         | Faster R-CNN      | Applies a pretrained Faster R-CNN model for inference and evaluation.      |

---

## 📊 Dataset

**Construction Site Safety Dataset** from Roboflow:
- Classes:
  - `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `Machinery`, `Vehicle`
- Directory Structure:
  ```
  /css-data/
    ├── train/
    ├── valid/
    └── test/
  ```

---

## 🧠 Models & Training

### 1. `yolo-training.ipynb`

**Objective**: Fine-tune a YOLOv8 model on construction safety data.

- Uses Ultralytics YOLOv8 framework.
- Creates a custom `YAML` config for dataset structure.
- Trains the model by freezing initial layers.
- Outputs validation metrics including:
  - Precision
  - Recall
  - mAP@0.5
  - mAP@0.5:0.95
- Saves results as CSV.

---

### 2. `pretrained-yolo.ipynb`

**Objective**: Evaluate a pretrained YOLOv8 model.

- Loads pretrained model weights (`last.pt`).
- Validates on `valid/` split.
- Extracts and saves evaluation metrics in CSV.
- Ideal for benchmarking vs custom-trained YOLO model.

---

### 3. `fast-rcnn.ipynb`

**Objective**: Use a pretrained Faster R-CNN model for inference and evaluation.

- Loads `fasterrcnn_resnet50_fpn` from PyTorch.
- Replaces classification head to fit custom classes.
- Uses a custom `PyTorch Dataset` for validation images.
- Performs inference and stores:
  - Bounding boxes
  - Predicted classes
  - Confidence scores
- Saves evaluation results and displays bounding boxes visually.

---

## 📦 Dependencies

Install the required packages:

```bash
pip install ultralytics torch torchvision pandas pyyaml tqdm matplotlib
```

---

## 📈 Outputs

- `validation_results.csv`: Contains evaluation metrics for each model.
- Visualizations (Faster R-CNN) are displayed using PIL and matplotlib.
- YOLO model summary is printed using `model.info()`.

---

## 📝 License

MIT License.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
- [Roboflow Dataset](https://universe.roboflow.com/)

---

## 📬 Contact

For questions or collaborations, please reach out via GitHub Issues or email.
