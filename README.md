# IMDb Sentiment Analyzer

I wanted to know what actually matters in NLP, so I ran the same dataset through four models, from logistic regression to DistilBERT, and paid attention. Same 50k reviews, increasing complexity, one question: does harder always mean better?

## Results

| Stage | Model | Accuracy | F1 | AUC |
|-------|-------|----------|----|-----|
| 1 | Logistic Regression + TF-IDF | **89.51%** | 89.5% | ~0.96 |
| 2 | Feedforward Neural Network | 89.34% | 89.4% | 0.9607 |
| 3 | Bidirectional LSTM | 88.81% | 89.1% | 0.9554 |
| 4 | DistilBERT (fine-tuned) | 87.62% | 87.9% | 0.9503 |

The simplest model won. All three deep learning models overfit. The FNN and LSTM were working on bag-of-words inputs that throw away word order anyway, and DistilBERT was bottlenecked by the hardware I had (GTX 1060, batch size 16, max length 128). That last one stings a bit. Given a proper GPU, DistilBERT should comfortably hit 92%+. Model choice without the hardware to match isn't really a fair fight.

## Dataset

[IMDb Movie Reviews - Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). 50,000 reviews, perfectly balanced (50/50), 80/20 train-test split.

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
