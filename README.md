# IMDb Sentiment Analyzer

Binary sentiment classification on 50,000 IMDb movie reviews, built as a learning portfolio. Four models, same dataset, increasing complexity — the goal was to understand what actually drives performance, not just chase accuracy.

---

## Results

| Stage | Model | Accuracy | F1 | AUC |
|-------|-------|----------|----|-----|
| 1 | Logistic Regression + TF-IDF | **89.51%** | 89.5% | ~0.96 |
| 2 | Feedforward Neural Network | 89.34% | 89.4% | 0.9607 |
| 3 | Bidirectional LSTM | 88.81% | 89.1% | 0.9554 |
| 4 | DistilBERT (fine-tuned) | 86.77% | 87.7% | 0.9518 |

The simplest model won. All three deep learning models overfit — the FNN and LSTM on bag-of-words inputs that discard word order, DistilBERT on hardware constraints (GTX 1060, batch size 16, max length 128). With a larger GPU, DistilBERT would be expected to clear 92%.

---

## Dataset

[IMDb Movie Reviews — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). 50,000 reviews, perfectly balanced (50/50), 80/20 train-test split.

---

## Setup

```
Python 3.11.9  |  PyTorch cu121  |  CUDA 12.1  |  GTX 1060 6GB
```

```bash
source venv/Scripts/activate

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Everything else
pip install -r requirements.txt
```

---

## Structure

```
sentiment-analyzer/
├── data/
│   └── IMDB Dataset.csv
├── stage1_logistic_regression.ipynb
├── stage2_feedforward_nn.ipynb
├── stage3_lstm.ipynb
└── stage4_distilbert.ipynb
```