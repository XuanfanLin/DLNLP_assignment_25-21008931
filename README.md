# MarianMT English-Chinese Translation - Fine-tuning Pipeline

This project fine-tunes the [Helsinki-NLP/opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) MarianMT model on a subset of the `news-commentary-v14` English-Chinese parallel dataset. The pipeline is fully implemented in `train_pipeline.py` and evaluated using BLEU score.

## 📁 Project Structure

```
DLNLP_assignment_25-21008931/
│
├── Datasets/
│   └── news-commentary-v14.en-zh.tsv        # Parallel 
│
├── result/
│   ├── final_model/        
│   ├── training_loss.png                    # Loss per epoch plot
│   ├── bleu_score.png                      # BLEU score per epoch plot
│   └── sentence_length_distribution.png     # EDA visualization
│
├── train_pipeline.py                    # Full training and evaluation pipeline
└── README.md                                # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Clone the repository or download the files.

```bash
git clone <your-repo-url>
cd DLNLP_assignment_25-21008931
```

### 2. Install dependencies

Make sure you're using Python 3.10+. Create a virtual environment and install required packages:

```bash
pip install torch==2.0.1
pip install transformers==4.41.2
pip install datasets evaluate sacrebleu sentencepiece matplotlib sacremoses
```

> ⚠️ If you're replicating the environment from Kaggle, ensure `peft` is removed:
```bash
pip uninstall -y peft
```

### 3. Prepare data

Download the dataset manually and place it into the `Datasets/` folder:

- File name: `news-commentary-v14.en-zh.tsv`
- Source: [statmt.org/news-commentary](https://data.statmt.org/news-commentary/v14/training/)


### 4. Run training pipeline

```bash
python train_pipeline.py
```

This will:

- Load and visualize sentence length distribution
- Tokenize and split the dataset
- Fine-tune MarianMT on the EN-ZH pairs
- Save training loss / BLEU plots to `result/`
- Save the best model to `result/final_model/`
- Output BLEU score and sample translations

---

## 📊 Output Results

| File                             | Description                                  |
|----------------------------------|----------------------------------------------|
| `result/sentence_length_distribution.png` | Distribution of EN and ZH sentence lengths |
| `result/training_loss.png`       | Loss curve over epochs                      |
| `result/bleu_score.png`          | BLEU score evolution                        |
| `result/final_model/`            | Saved model, tokenizer, and generation config |

---

## 🧪 Evaluation

- **Metric**: SacreBLEU
- **Final BLEU Score** (on 10% held-out test set): ~7.88
- **Sample translations** printed in console

---

## 📎 Kaggle Notebook Reference

The original online implementation can be found at:

👉 [Kaggle Notebook - Xuanfan Lin](https://www.kaggle.com/code/xuanfanlin/nlp-task)

---

## 📌 Notes

- Training limited to first 25,000 sentence pairs to balance accuracy and runtime
- EarlyStopping with patience of 2 is applied
- Tokenization uses `max_length=128`, `num_beams=5` for generation

---
