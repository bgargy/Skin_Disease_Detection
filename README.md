
# 🧠 Skin Cancer Detection Using EfficientNetB0

This project builds a deep learning model to classify skin lesions into different disease categories using dermatoscopic images from the HAM10000 dataset. It leverages transfer learning with EfficientNetB0 to achieve accurate classification results.

---

## 📂 Dataset

- **HAM10000 ("Human Against Machine with 10000 training images")**
  - Total: 10,015 dermatoscopic images
  - Includes metadata like age, sex, and lesion location
  - Dataset split across two folders:
    - `HAM10000_images_part_1`
    - `HAM10000_images_part_2`
  - CSV metadata: `HAM10000_metadata.csv`

---

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/skin-cancer-detection.git
   cd skin-cancer-detection
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Requirements:
   - Python ≥ 3.7
   - TensorFlow ≥ 2.8
   - scikit-learn
   - Pandas, NumPy, Seaborn, Matplotlib

---

## 📊 Exploratory Data Analysis

- Class distribution (`dx`)
- Gender distribution
- Age histogram

Visualizations help understand class imbalance and demographic distributions.

---

## 🏗️ Model Architecture

- **Base Model**: `EfficientNetB0` (pre-trained on ImageNet)
- Layers added:
  - `GlobalAveragePooling2D`
  - `Dense` layers (128 or 256 units)
  - `Dropout`
  - `Softmax` output layer for multi-class classification

---

## ⚙️ Training Details

- Image size: 224×224
- Batch size: 32
- Loss: Sparse Categorical Crossentropy
- Optimizer: Adam (learning rate = 0.0001)
- Data Augmentation: Rotation, zoom, flips, shifts
- Callbacks:
  - EarlyStopping
  - ModelCheckpoint
  - ReduceLROnPlateau

---

## 📈 Evaluation Metrics

- **Accuracy** on validation/test set
- **Loss curves** plotted over epochs
- **Class weights** used to handle class imbalance

---

## 🧪 Results

| Metric       | Value                  |
|--------------|------------------------|
| Test Accuracy| ~85–90% (varies based on tuning) |
| Best Model   | Saved as `.keras` and `.h5` format |

---

## 📌 Future Work

- Try other architectures: ResNet, Inception
- Include lesion segmentation as preprocessing
- Add explainability (e.g., Grad-CAM)

---

## 📄 License

This project is for academic and research purposes. If you use it, please cite the HAM10000 dataset and acknowledge the author(s).
