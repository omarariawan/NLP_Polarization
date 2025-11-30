import os
import random
import numpy as np
import pandas as pd
import datasets
from datasets import Dataset
import evaluate
f1_metric = evaluate.load("f1")
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from torch.nn import CrossEntropyLoss

# ------------------------ CONFIG ------------------------
TRAIN_FILE = "/content/sample_data/eng.csv"        # labeled train file (your ground truth)
DEV_FILE = "/content/sample_data/dev_eng.csv"      # unlabeled dev file (for final predictions)
OUTPUT_PRED_CSV = "predictions_for_evaluation.csv"

# Model choice (you chose option A earlier)
MODEL_NAME = "xlm-roberta-large"


# Tokenization / batching / memory
MAX_LENGTH = 256              # reduces memory compared to 512; usually enough for social posts
PER_DEVICE_BATCH = 8          # try 8; if OOM drop to 4
EVAL_BATCH = 4
SEED = 42

# Manual hyperparameter grid to try (small grid, faster than full optuna search)
LR_GRID = [2e-5, 1e-5, 5e-6]
EPOCH_GRID = [3, 4]
WARMUP_RATIO = 0.06           # fraction of total steps used for warmup
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 2

# Save & logging
OUTPUT_DIR = "./results_xlmr_large"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------- REPRODUCIBILITY ---------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ---------------------- LOAD DATA --------------------------
df = pd.read_csv(TRAIN_FILE)
print(f"Loaded TRAIN shape: {df.shape}")
# make sure label column is integer 0/1 and columns are named 'id','text','polarization'
assert "text" in df.columns, "No 'text' column found in train file."
assert "polarization" in df.columns, "No 'polarization' column found in train file."

df = df.dropna(subset=["text", "polarization"]).reset_index(drop=True)
df["polarization"] = df["polarization"].astype(int)

# Split train/val for hyperparameter search (stratified)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["polarization"])
print("Train / Val sizes:", train_df.shape, val_df.shape)

# Rename 'polarization' to 'labels' for HuggingFace Trainer compatibility
train_df = train_df.rename(columns={"polarization": "labels"})
val_df = val_df.rename(columns={"polarization": "labels"})

# Create HuggingFace datasets
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

# Also prepare a 'full' dataset for final training on all labeled data
full_train_ds = Dataset.from_pandas(df.rename(columns={"polarization": "labels"}).reset_index(drop=True))

# Prepare test (unlabeled) dataset for final predictions
if os.path.exists(DEV_FILE):
    dev_df = pd.read_csv(DEV_FILE)
    dev_df = dev_df.rename(columns={"text": "text"})  # ensure correct name
    test_ds = Dataset.from_pandas(dev_df.reset_index(drop=True))
else:
    test_ds = None
    print("Warning: dev file not found; final prediction step will be skipped.")

# -------------------- TOKENIZER -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=256   # Reduced for memory
    )

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        gradient_checkpointing=True
    )


print("Tokenizing datasets (this may take a moment)...")

def tokenize_fn(x):
    return tokenizer(
        x["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

cols_to_remove = [c for c in train_ds.column_names if c not in ["labels"]]

tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
tokenized_full = full_train_ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)

tokenized_test = (
    test_ds.map(tokenize_fn, batched=True, remove_columns=[c for c in test_ds.column_names if c != "text"])
    if test_ds is not None
    else None
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------- CLASS WEIGHTS (to counter imbalance) -----------
# compute weights inverse to class frequency on the training set
label_counts = train_df["labels"].value_counts().to_dict()
total = sum(label_counts.values())
class_weights = []
for lbl in [0, 1]:
    freq = label_counts.get(lbl, 0)
    if freq == 0:
        class_weights.append(1.0)
    else:
        class_weights.append(total / (2.0 * freq))  # balanced inverse freq normalized by 2 classes
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

print("Class counts (train split):", label_counts)
print("Class weights used (0,1):", class_weights)

# ---------------- Custom Trainer (to use class weights) ------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # Remove labels so model doesn't try to use them automatically
        inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = CrossEntropyLoss(weight=class_weights_tensor.to(model.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ---------------- Metrics ------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return f1_metric.compute(predictions=predictions, references=labels, average="macro")

# ---------------- Manual grid search -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
best_score = -999
best_config = None
best_checkpoint = None

# small safety: reduce batch if no GPU
if device == "cpu":
    PER_DEVICE_BATCH = 4
    EVAL_BATCH = 16

for lr in LR_GRID:
    for epochs in EPOCH_GRID:
        print(f"\n--- Training trial: lr={lr}, epochs={epochs} ---")

        torch.cuda.empty_cache()

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir="./trial",
    learning_rate=lr,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    num_train_epochs=epochs,
    # remove evaluation_strategy/save_strategy/load_best_model_at_end if unsupported
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
)


print("\n=== Grid search complete ===")
print("Best validation macro_f1:", best_score)
print("Best hyperparameters:", best_config)
print("Best model saved at:", os.path.join(OUTPUT_DIR, "best_trial_model"))

# ---------------- Train FINAL on FULL TRAINING SET -----------------
print("\n--- Training FINAL model on 100% of labeled data with best config ---")
final_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.gradient_checkpointing_enable()



final_training_args = TrainingArguments(
    output_dir="./trial",
    learning_rate=lr,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    num_train_epochs=epochs,
    # remove evaluation_strategy/save_strategy/load_best_model_at_end if unsupported
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

final_trainer = WeightedTrainer(
    model=final_model,
    args=final_training_args,
    train_dataset=tokenized_full,
    eval_dataset=tokenized_val,   # keep a small validation to watch overfitting during final training
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


final_trainer.train()
final_trainer.save_model(os.path.join(OUTPUT_DIR, "final_trained_model"))
print("Final model trained and saved.")

# ------------------ Predict on dev / unlabeled test ------------------
if tokenized_test is not None:
    print("\n--- Running predictions on dev / unlabeled test set ---")
    preds_output = final_trainer.predict(tokenized_test)
    y_pred = np.argmax(preds_output.predictions, axis=1)

    # retrieve IDs from the original dev dataframe
    if "id" in dev_df.columns:
        ids = dev_df["id"].tolist()
    else:
        ids = list(range(len(y_pred)))

    out_df = pd.DataFrame({"id": ids, "polarization": y_pred})
    out_df.to_csv(OUTPUT_PRED_CSV, index=False)
    print(f"Predictions written to {OUTPUT_PRED_CSV} (rows: {len(out_df)})")
else:
    print("No test/dev dataset found; skipping prediction step.")