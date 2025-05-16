# run_training.py

import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    MarianTokenizer, MarianMTModel,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
    GenerationConfig
)
import evaluate

# === Create necessary output directories ===
os.makedirs("result", exist_ok=True)
os.makedirs("result/logs", exist_ok=True)

# === Set random seeds for reproducibility ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# === Load and preprocess raw data ===
df = pd.read_csv("Datasets/news-commentary-v14.en-zh.tsv", sep="\t", names=["en", "zh"], on_bad_lines='skip')
df.dropna(inplace=True)
df = df.head(25000)  # Limit dataset size for faster training

# === Load tokenizer and perform EDA ===
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
df['zh_len'] = df['zh'].apply(lambda x: len(tokenizer.encode(x, truncation=True)))
df['en_len'] = df['en'].apply(lambda x: len(tokenizer.encode(x, truncation=True)))

# Plot sentence length distributions
plt.figure(figsize=(12, 6))
plt.hist(df['en_len'], bins=50, alpha=0.6, label='English')
plt.hist(df['zh_len'], bins=50, alpha=0.6, label='Chinese')
plt.title("Sentence Length Distribution")
plt.xlabel("Length")
plt.ylabel("Count")
plt.xlim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig("result/sentence_length_distribution.png")
plt.close()

# === Convert to HuggingFace Dataset and tokenize ===
dataset = Dataset.from_pandas(df[['en', 'zh']])
dataset = dataset.train_test_split(test_size=0.1, seed=SEED)

def preprocess(example):
    model_inputs = tokenizer(example['en'], truncation=True, padding='max_length', max_length=128)
    labels = tokenizer(text_target=example['zh'], truncation=True, padding='max_length', max_length=128)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# === Load pretrained model and data collator ===
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# === Define training arguments ===
training_args = Seq2SeqTrainingArguments(
    output_dir="result",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=12,
    learning_rate=2e-5,
    evaluation_strategy="epoch",  # runs evaluation at the end of each epoch
    save_strategy="epoch",        # saves the model at the end of each epoch
    logging_dir="result/logs",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=5,
    report_to="none",  # disable WandB or TensorBoard reporting
    seed=SEED
)

# === Define BLEU metric for evaluation ===
bleu = evaluate.load("sacrebleu")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    return {"bleu": result["score"]}

# === Initialize trainer with early stopping ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# === Begin training ===
trainer.train()

# === Save model, tokenizer, and generation config ===
model.save_pretrained("result/final_model")
tokenizer.save_pretrained("result/final_model")
gen_config = GenerationConfig(
    max_length=128,
    num_beams=5,
    forced_eos_token_id=tokenizer.eos_token_id,
    bad_words_ids=[[tokenizer.pad_token_id]]
)
gen_config.save_pretrained("result/final_model")

# === Plot training loss over epochs ===
loss_logs = [x for x in trainer.state.log_history if "loss" in x and "epoch" in x]
if loss_logs:
    epochs = [log["epoch"] for log in loss_logs]
    losses = [log["loss"] for log in loss_logs]
    plt.plot(epochs, losses, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("result/training_loss.png")
    plt.close()

# === Plot BLEU score over epochs ===
metrics = trainer.state.log_history
bleu_scores = [(m["epoch"], m["eval_bleu"]) for m in metrics if "eval_bleu" in m]
if bleu_scores:
    epochs, scores = zip(*bleu_scores)
    plt.plot(epochs, scores, marker='o')
    plt.title("BLEU Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU Score")
    plt.grid(True)
    plt.savefig("result/bleu_score.png")
    plt.close()

# === Final evaluation on test set ===
eval_result = trainer.evaluate()
print(f"✅ Final BLEU from best model: {eval_result['eval_bleu']:.2f}")

# === Show a few sample translations ===
for i in range(5):
    print(f"EN: {df['en'].iloc[i]}")
    print(f"GT: {df['zh'].iloc[i]}")
    inputs = tokenizer(df['en'].iloc[i], return_tensors="pt", truncation=True, padding=True).to(model.device)
    translated = model.generate(**inputs)
    print(f"MT: {tokenizer.decode(translated[0], skip_special_tokens=True)}\n")
