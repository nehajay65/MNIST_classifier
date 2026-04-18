# MNIST Digit Classifier

A convolutional neural network that recognises handwritten digits (0–9) from the classic MNIST dataset, built with **PyTorch**.

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~99% |
| Architecture | CNN (2 conv layers + FC) |
| Epochs | 10 |
| Optimizer | Adam |

---

## Project Structure

```
mnist_project/
├── train.py  <- Train the model
├── predict.py <- Run inference
├── requirements.txt <- Python dependencies
├── notebooks/
│   └── exploration.ipynb <- EDA + visualisations
├── data/   <- MNIST downloads here (auto, git-ignored)
└── models/  <- Saved weights (git-ignored)
```

---

## Setup

### 1 · Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/mnist-classifier.git
cd mnist-classifier
```

### 2 · Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3 · Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Train
```bash
python train.py
```
The MNIST dataset downloads automatically on the first run.  
The best model is saved to `models/best_model.pth`.

### Predict on the test set
```bash
python predict.py
```

### Predict a single image
```bash
python predict.py path/to/your/digit.png
```

### Explore in Jupyter
```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## Model Architecture

```
Input (1×28×28)
  └─ Conv2d(1→32, k=3) + ReLU + MaxPool  ->  32×14×14
  └─ Conv2d(32→64, k=3) + ReLU + MaxPool -> 64×7×7
  └─ Flatten → Linear(3136→256) + ReLU + Dropout(0.5)
  └─ Linear(256→10)
Output: logits for digits 0–9
```

---

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — 70,000 greyscale 28×28 images of handwritten digits.  
Downloaded automatically by `torchvision.datasets.MNIST` — no manual step needed.

---

## License

MIT- feel free to use and modify.
