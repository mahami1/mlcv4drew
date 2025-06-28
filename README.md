# mlcv4drew

# Machine learning computer vision for robotic disassembly of e-waste

# Screw Detector CNN

This project contains Python code and an associated dataset for detecting screws in images using a CNN-based object detector. It includes tools for dataset handling, model training, evaluation, and simulation.

## 📂 Directory Structure

```
<base_directory>/
├── README.md             # This file
├── python/               # Python source files
├── docs/                 # Documentation
├── repo/                 # Dataset repository
│   ├── scenes/           # Labelled source scenes
│   │   ├── Images/       # Source scene images
│   │   └── Annotations/  # Scene labels
│   └── cnn/              # CNN resources
│       ├── weights/      # Saved CNN weights
│       └── input/        # ROI datasets
│           ├── train/    # Training ROIs (screw, hole, other)
│           ├── validate/ # Validation ROIs
│           └── test/     # Test ROIs
└── output/               # Annotated output scenes
```

## ⚙️ Environment Setup

1. **Install Dependencies**

   * Python `3.11.6`
   * OpenCV `4.7.0` (`pip install opencv-python`)

2. **Setup Directory**

   * Create a base directory (e.g., `C:\UNSW\comp4953`)
   * Extract this repository and dataset into the base directory following the structure above.

---

## 🚀 Usage Instructions

### 1. ✅ Test the Environment (Proxy Simulation)

Run:

```bash
python python/proxy.py -count 100 -ep 0.7 -dp 0.9 -learn -seed 42
```

Expected output sample:

```
BaseM: precision=1.00(0.98), recall=0.67(0.65), f1score=0.80(0.76), offset=1.00(1.66)
Model: precision=1.00(0.95), recall=1.00(0.76), f1score=1.00(0.82), offset=1.17(1.78)
Done
```

---

### 2. 🔍 Test the Hough Detector (Optional)

Run:

```bash
python python/extractor.py -scenes repo/scenes -output output
```

Sample output:

```
BaseM: precision=0.45(0.45), recall=0.36(0.36), f1score=0.40(0.40), offset=2.60(2.60)
...
Done
```

---

### 3. 🏗️ Train the Screw Detector CNN

> ⚠️ Note: Training can take \~45 minutes on a standard laptop with Intel i7 and 8GB RAM. The weights file will be \~350MB.

Run:

```bash
python python/detector.py -roi repo/cnn/input -weights repo/cnn/weights/Xception-final.h5
```

Sample output snippet:

```
Epoch 15/15
318/318 [==============================] - loss: 0.7806 - accuracy: 0.7879 - val_loss: 0.3831 - val_accuracy: 0.8623
```

---

### 4. 🧪 Test the Screw Detector CNN (Optional)

Run:

```bash
python python/detector.py -roi repo/cnn/input/test -weights repo/cnn/weights/Xception-final.h5
```

Sample evaluation output:

```
              precision    recall  f1-score   support
Hole              0.75      0.97      0.85       100
Other             1.00      0.77      0.87       100
Screw             0.93      0.91      0.92       201

accuracy                               0.89       401
macro avg         0.90      0.88      0.88       401
weighted avg      0.90      0.89      0.89       401

Confusion matrix:
[[ 97   0   3]
 [ 13  77  10]
 [ 19   0 182]]
Done
```

---

### 5. 🚫 Run Non-Learning Proposer Test

Run:

```bash
python python/main.py -scenes repo/scenes -output output -weights repo/cnn/weights/Xception-final.h5 -generate 10
```

Sample output:

```
BaseM: precision=1.00(0.90), recall=0.29(0.24), f1score=0.44(0.38), offset=2.00(2.45)
Model: precision=1.00(0.90), recall=0.57(0.52), f1score=0.73(0.62), offset=2.62(2.78)
Done
```

---

### 6. 📈 Run Learning Proposer Test

Run:

```bash
python python/main.py -scenes repo/scenes -output output -weights repo/cnn/weights/Xception-final.h5 -generate 20 -learn
```

Sample output:

```
BaseM: precision=1.00(0.95), recall=0.27(0.23), f1score=0.43(0.37), offset=4.00(3.00)
Model: precision=1.00(0.95), recall=0.36(0.28), f1score=0.53(0.42), offset=4.00(2.98)
Done
```

---

## 💾 Notes

* Clear the `output/` directory before each run to avoid cluttered results.
* CNN weights are saved incrementally during training in `repo/cnn/weights/`.
* Test and validation ROI images are categorized into `screw`, `hole`, and `other`.

---

