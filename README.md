# 🎬 IMDb Sentiment Analyzer

> A step-by-step journey from classical Machine Learning to Deep Learning, classifying 50,000 IMDb movie reviews as **Positive** or **Negative**.

---

## 📌 Project Overview

This project was built as a **learning portfolio** while studying ML/AI engineering from scratch. The idea is simple: use the same dataset and task across four progressively complex models, so you can clearly see how each approach works and compares.

Each stage must **beat the previous baseline**. Every model, every concept, learned hands-on.

---

## 🗺️ Roadmap

| Stage | Model | Status | Accuracy |
|-------|-------|--------|----------|
| 1 | Logistic Regression + TF-IDF | ✅ Complete | **89.51%** |
| 2 | Feedforward Neural Network (PyTorch) | ✅ Complete | **89.34%** |
| 3 | LSTM (Bidirectional) | ✅ Complete | **88.81%** |
| 4 | DistilBERT (Transformer) | ✅ Complete | **86.77%** |

> **Baseline:** 89.51% (Stage 1 Logistic Regression). See the [full comparison table](#-all-stages-comparison) for a deeper look at why the more complex models did not simply win.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [IMDb Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| Total Samples | 50,000 |
| Positive Reviews | 25,000 (50%) |
| Negative Reviews | 25,000 (50%) |
| Train / Test Split | 80% / 20% → 40,000 train · 10,000 test |
| Features | Raw review text |
| Target | Sentiment label (positive = 1, negative = 0) |
| Missing Values | None |

The dataset is perfectly balanced — no class imbalance to worry about.

---

## 🛠️ Environment & Setup

```
OS:            Windows 10
IDE:           VS Code + Jupyter Notebook
Terminal:      Git Bash (MINGW64)
GPU:           NVIDIA GeForce GTX 1060 6GB
CUDA Driver:   581.29 (CUDA 13.0 runtime)
PyTorch Build: cu121 (CUDA 12.1)
Python:        3.11.9 (virtual environment)
```

**Activate the virtual environment (Git Bash):**
```bash
source venv/Scripts/activate
```

**Key Libraries:**
```
torch (cu121)    — PyTorch deep learning framework
transformers     — Hugging Face (DistilBERT tokenizer & model)
scikit-learn     — TF-IDF vectorisation, Logistic Regression, metrics
pandas / numpy   — Data loading and array manipulation
matplotlib / seaborn — Plots (training curves, confusion matrix, ROC)
```

---

## 📁 Project Structure

```
sentiment-analyzer/
├── data/
│   └── IMDB Dataset.csv
├── stage1_logistic_regression.ipynb
├── stage2_feedforward_nn.ipynb
├── stage3_lstm.ipynb
├── stage4_distilbert.ipynb
└── README.md
```

---

## ⚡ Stage 1 — Logistic Regression + TF-IDF

### What is TF-IDF?

TF-IDF (Term Frequency – Inverse Document Frequency) converts raw text into numbers a model can understand. It rewards words that appear often in one specific review but rarely across all reviews — so words like *"brilliant"* or *"terrible"* score high, while *"the"* or *"and"* score low automatically.

### What is Logistic Regression?

Despite its name, Logistic Regression is a **classification** algorithm. It learns a weight for each TF-IDF feature and combines them into a single probability between 0 and 1. Reviews scoring ≥ 0.5 are classified as Positive.

### Pipeline

1. Load CSV → 50,000 rows, 2 columns (`review`, `sentiment`)
2. Encode labels → `positive = 1`, `negative = 0`
3. 80/20 train-test split (`random_state=42` for reproducibility)
4. TF-IDF vectorisation → `max_features=5000` (top 5,000 words kept)
5. Train `LogisticRegression(max_iter=10000)`
6. Evaluate → accuracy, classification report, confusion matrix

### Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **89.51%** |
| Precision | 89.7% |
| Recall | 89.5% |
| F1 Score | 89.5% |

**Confusion Matrix:**

|  | Predicted NEG | Predicted POS |
|--|:---:|:---:|
| **Actual NEG** | 4,380 ✅ | 581 ❌ |
| **Actual POS** | 463 ❌ | 4,576 ✅ |

### Key Insight

A simple linear model achieves nearly 90% accuracy with **no deep learning at all**. This is our baseline. Every future stage must beat **89.51%**.

---

## 🧠 Stage 2 — Feedforward Neural Network (PyTorch)

### What changed from Stage 1?

Instead of one linear layer, we now stack **three fully-connected (Linear) layers**. Between each layer we add ReLU activations and Dropout to help the network generalise. The final layer outputs a single number that Sigmoid squashes into a probability (0 → 1).

### Architecture

```
Input (5,000 TF-IDF features)
    ↓
Linear(5000 → 512) → ReLU → Dropout(0.3)
    ↓
Linear(512 → 256)  → ReLU → Dropout(0.3)
    ↓
Linear(256 → 1)    → Sigmoid
    ↓
Output (probability of being Positive)
```

**Total parameters:** ~2.8 million

### Training Setup

| Component | Setting |
|-----------|---------|
| Loss Function | BCELoss (Binary Cross-Entropy) |
| Optimiser | AdamW (lr=0.001, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early Stopping | patience=5, min_delta=1e-4 → restores best weights |
| Batch Size | 64 (625 batches per epoch) |
| Max Epochs | 30 (actual: 6 — early stopping triggered) |

### Training Log

| Epoch | Train Loss | Val Loss | Val Acc | F1 | Note |
|:---:|:---:|:---:|:---:|:---:|---|
| **1** | **0.3112** | **0.2544** | **89.34%** | **0.8941** | **Best checkpoint ⭐** |
| 2 | 0.2224 | 0.2609 | 89.22% | 0.8936 | Overfit begins |
| 3 | 0.1720 | 0.2849 | 89.17% | 0.8925 | |
| 4 | 0.0931 | 0.3702 | 88.39% | 0.8833 | LR halved by scheduler |
| 5 | 0.0323 | 0.4879 | 88.25% | 0.8820 | |
| 6 | 0.0097 | 0.7037 | 88.22% | 0.8861 | Early stop → restore epoch 1 |

### Results (restored best checkpoint)

| Metric | Score |
|--------|-------|
| **Accuracy** | **89.34%** |
| Precision | 89.52% |
| Recall | 89.30% |
| F1 Score | 89.41% |
| **AUC** | **0.9607** |

**Confusion Matrix:**

|  | Predicted NEG | Predicted POS |
|--|:---:|:---:|
| **Actual NEG** | 4,434 ✅ | 527 ❌ |
| **Actual POS** | 539 ❌ | 4,500 ✅ |

### Key Insight

The FNN achieves a very similar accuracy to Logistic Regression despite being 560× larger in parameter count. TF-IDF + LR is a very strong baseline for bag-of-words sentiment tasks. The real gains require models that understand **word order** and **context**.

---

## 🔁 Stage 3 — Bidirectional LSTM

### What is an LSTM and why is it different?

The FNN in Stage 2 treated each review as a **bag of words** — it had no sense of order. The sentence *"not good"* and *"good not"* would look identical to it.

An **LSTM** (Long Short-Term Memory) is a type of Recurrent Neural Network. It reads text **word by word**, left to right, like a human. As it reads, it carries a **memory** (called the hidden state) that accumulates context from all previous words. This lets it understand that *"not"* changes the meaning of *"good"* that follows.

**Bidirectional** means it reads both **forwards AND backwards**, then combines both views for richer context.

### Key New Concepts

| Concept | Simple Explanation |
|---------|-------------------|
| **Embedding layer** | A lookup table that converts each word's integer ID into a dense vector of 128 numbers. The model learns these vectors — similar words end up with similar vectors. |
| **Vocabulary** | Instead of TF-IDF, we give each unique word an integer ID. We keep the top 20,000 most frequent words. |
| **`<PAD>` token** | Used to make all reviews the same length (200 tokens). Padded positions are ignored by the LSTM. |
| **`<UNK>` token** | Represents any word not in our vocabulary (rare words, typos). |

### Architecture

```
Input (200 integer token IDs per review)
    ↓
Embedding(20,002 → 128)
    ↓
Bidirectional LSTM(128 → 256 per direction, 2 layers, dropout=0.3)
    ↓
Take final hidden state → concat forward + backward → (512,)
    ↓
Dropout(0.3) → Linear(512 → 1) → Sigmoid
    ↓
Output (probability of being Positive)
```

**Total trainable parameters:** 4,928,257

### Vocabulary & Tokenisation

| Setting | Value |
|---------|-------|
| Vocabulary size | 20,000 most frequent words + `<PAD>` + `<UNK>` = 20,002 |
| Max sequence length | 200 tokens |
| Total unique words in corpus | 99,420 |
| Text preprocessing | Remove HTML tags · lowercase · keep letters only |

### Training Setup

| Component | Setting |
|-----------|---------|
| Loss Function | BCELoss |
| Optimiser | AdamW (lr=0.001, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Early Stopping | patience=5, min_delta=1e-4 |
| Batch Size | 64 (625 batches per epoch) |
| Max Epochs | 20 (actual: 13 — early stopping triggered) |

### Training Log

| Epoch | Train Loss | Val Loss | Val Acc | F1 | Note |
|:---:|:---:|:---:|:---:|:---:|---|
| 1 | 0.6517 | 0.6249 | 63.51% | 0.5048 | Still learning |
| 2 | 0.5674 | 0.4667 | 80.27% | 0.8232 | Big jump |
| 3 | 0.3950 | 0.3278 | 85.39% | 0.8571 | |
| 4 | 0.3195 | 0.2993 | 87.16% | 0.8769 | |
| 5 | 0.2730 | 0.2855 | 88.24% | 0.8870 | |
| 6 | 0.2384 | 0.2960 | 87.82% | 0.8736 | |
| 7 | 0.2089 | 0.3038 | 88.51% | 0.8908 | |
| **8** | **0.1805** | **0.2849** | **88.81%** | **0.8913** | **Best checkpoint ⭐** |
| 9 | 0.1519 | 0.3080 | 89.24% | 0.8956 | Val loss rising |
| 10 | 0.1298 | 0.3199 | 89.27% | 0.8946 | |
| 11 | 0.1109 | 0.3490 | 88.64% | 0.8906 | |
| 12 | 0.0801 | 0.3745 | 88.94% | 0.8932 | |
| 13 | 0.0691 | 0.4005 | 88.79% | 0.8909 | Early stop → restore epoch 8 |

### Results (restored best checkpoint — epoch 8)

| Metric | Score |
|--------|-------|
| **Accuracy** | **88.81%** |
| Precision | 87.30% |
| Recall | 91.03% |
| F1 Score | 89.13% |
| **AUC** | **0.9554** |
| Optimal Threshold | 0.531 |

**Confusion Matrix:**

|  | Predicted NEG | Predicted POS |
|--|:---:|:---:|
| **Actual NEG** | 4,294 ✅ | 667 ❌ |
| **Actual POS** | 452 ❌ | 4,587 ✅ |

### Key Insight

The LSTM has the **highest recall (91%)** of the first three models — it almost never misses a positive review — but at the cost of precision (more false positives). Despite being architecturally more sophisticated, it did not beat Stage 1's baseline, showing that sequential models need careful tuning and that bag-of-words + LR is extremely competitive for this task.

---

## 🤖 Stage 4 — DistilBERT (Transformer)

### What is DistilBERT?

In Stages 1–3 we built models from scratch. **DistilBERT is different** — it is a pre-trained model created by Hugging Face that has already read **billions of words** from the internet and books. It already understands English deeply before seeing a single movie review.

What we do is called **fine-tuning**: we take this already-smart model and do a small amount of extra training on our IMDb data so it learns the specific task of sentiment classification. Like hiring an expert who already knows how to read — you just train them on your specific job for a few days.

**Why DistilBERT and not full BERT?** BERT is very large and slow. DistilBERT keeps ~97% of BERT's performance at 40% fewer parameters — much more practical for a 6 GB GPU.

### Key New Concepts

| Concept | Simple Explanation |
|---------|-------------------|
| **WordPiece Tokenizer** | DistilBERT's own tokenizer (vocab = 30,522 tokens). We do not build a vocabulary ourselves — we use the pre-trained one. |
| **`[CLS]` token** | Every input starts with this special token. After processing the whole sequence, the `[CLS]` vector summarizes the entire text — we feed it into our classifier. |
| **Attention mask** | Tells the model which tokens are real words vs. padding zeros. |
| **Fine-tuning** | Training a pre-trained model on a new, smaller dataset with a very small learning rate to avoid forgetting pre-learned knowledge. |
| **Linear warmup** | LR starts at 0, rises to 2e-5 over the first 10% of training, then stays constant — prevents a destabilizing first update. |

### Architecture

```
Input (raw review text)
    ↓
DistilBERT Tokenizer → input_ids + attention_mask (max_len=128)
    ↓
DistilBERT Backbone (66M pre-trained parameters)
    ↓
Take [CLS] token output → (768,)
    ↓
Dropout(0.3) → Linear(768 → 1) → Sigmoid
    ↓
Output (probability of being Positive)
```

**Total parameters:** 66,363,649 (all trainable during fine-tuning)

### Training Setup

| Component | Setting |
|-----------|---------|
| Loss Function | BCELoss |
| Optimiser | AdamW (lr=2e-5, weight_decay=1e-2) |
| LR Scheduler | Linear warmup (1,000 warmup steps / 10,000 total) |
| Early Stopping | patience=2, min_delta=1e-4 |
| Batch Size | 16 (2,500 batches per epoch) |
| Max Epochs | 4 (actual: 3 — early stopping triggered) |
| Max Token Length | 128 |

> ⚠️ Each epoch took ~20–40 minutes on a GTX 1060 6GB. Batch size of 16 (vs 64 in previous stages) was required due to the model's memory footprint.

### Training Log

| Epoch | Train Loss | Val Loss | Val Acc | F1 | Note |
|:---:|:---:|:---:|:---:|:---:|---|
| **1** | **0.3746** | **0.3028** | **86.77%** | **0.8771** | **Best checkpoint ⭐** |
| 2 | 0.2349 | 0.3368 | 88.45% | 0.8886 | Val loss rising |
| 3 | 0.1446 | 0.4229 | 88.23% | 0.8867 | Early stop → restore epoch 1 |

### Results (restored best checkpoint — epoch 1)

| Metric | Score |
|--------|-------|
| **Accuracy** | **86.77%** |
| Precision | 82.43% |
| Recall | **93.73%** |
| F1 Score | 87.71% |
| **AUC** | **0.9518** |
| Optimal Threshold | 0.699 |

**Confusion Matrix:**

|  | Predicted NEG | Predicted POS |
|--|:---:|:---:|
| **Actual NEG** | 3,954 ✅ | 1,007 ❌ |
| **Actual POS** | 316 ❌ | 4,723 ✅ |

### Key Insight

DistilBERT achieves the **highest recall of all four models (93.7%)** — it almost never misses a positive review — but also the lowest precision, meaning it over-predicts positives. The model converges extremely fast (best result at epoch 1) and overfits rapidly on limited hardware.

The lower-than-expected overall accuracy is primarily a **hardware constraint**: a GTX 1060 forces `batch_size=16` and `max_len=128`. With a larger GPU (batch_size=32+, max_len=256+, more epochs), DistilBERT would be expected to reach 92–93%+ and comfortably beat the Stage 1 baseline.

---

## 📈 All Stages — Comparison

| Criterion | Stage 1 · LR | Stage 2 · FNN | Stage 3 · LSTM | Stage 4 · DistilBERT |
|-----------|:---:|:---:|:---:|:---:|
| **Accuracy** | **89.51%** | 89.34% | 88.81% | 86.77% |
| **AUC** | ~0.96 | 0.9607 | 0.9554 | 0.9518 |
| **F1 Score** | 89.5% | 89.41% | 89.13% | 87.71% |
| **Recall** | 89.5% | 89.3% | 91.0% | **93.7%** |
| **Precision** | **89.7%** | 89.5% | 87.3% | 82.4% |
| Parameters | 5,000 | 2.8M | 4.9M | 66.4M |
| Training Time | Seconds | ~2 min | ~30 min | ~2 hours |
| GPU Required | No | Yes | Yes | Yes |
| Word Order Awareness | ❌ | ❌ | ✅ | ✅ |
| Pre-trained Knowledge | ❌ | ❌ | ❌ | ✅ |
| Beats Stage 1 Baseline? | — | ❌ −0.17pp | ❌ −0.70pp | ❌ −2.74pp |

### Why did the more complex models not win?

This is one of the most important lessons of this project:

- **TF-IDF + Logistic Regression is exceptionally strong** for balanced binary sentiment classification. It is fast, stable, and does not overfit.
- **The FNN, LSTM, and DistilBERT all overfit** — their training loss drops far lower than validation loss, meaning they memorise training data rather than generalising.
- **Hardware constraints** limited DistilBERT the most: a small batch size and short sequence length significantly reduce its effectiveness.
- **More complexity ≠ better results** without sufficient compute, tuning time, and data augmentation. This is a real-world lesson that applies far beyond this project.

---

## 👤 Author

**Manos Sanoudakis**
Statistics & Insurance Science Graduate · University of Piraeus
Transitioning into ML/AI Engineering


