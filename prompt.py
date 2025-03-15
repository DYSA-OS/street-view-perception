import warnings
import pandas as pd
import numpy as np
import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import random
import os

warnings.simplefilter(action="ignore", category=FutureWarning)

# Seed setting
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def create_tokenizer(tokenizer, input_texts, max_length=600):
    data_tokenizer = tokenizer.batch_encode_plus(
        input_texts,
        max_length=max_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )

    return data_tokenizer


def training(class_name, result_dir):
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased", do_lower_case=True)

    df_train = pd.read_csv("./data/llava/train.csv")
    df_val = pd.read_csv("./data/llava/val.csv")
    df_test = pd.read_csv("./data/llava/test.csv")

    train_input_texts = df_train["prompt"].tolist()
    val_input_texts = df_val["prompt"].tolist()
    test_input_texts = df_test["prompt"].tolist()

    # Tokenize data
    train_encoded_inputs = create_tokenizer(tokenizer, train_input_texts)
    val_encoded_inputs = create_tokenizer(tokenizer, val_input_texts)
    test_encoded_inputs = create_tokenizer(tokenizer, test_input_texts)

    # Data preparation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_input_ids = train_encoded_inputs["input_ids"].to(device)
    train_attention_mask = train_encoded_inputs["attention_mask"].to(device)
    train_labels = torch.tensor(df_train[class_name].tolist()).to(device)

    val_input_ids = val_encoded_inputs["input_ids"].to(device)
    val_attention_mask = val_encoded_inputs["attention_mask"].to(device)
    val_labels = torch.tensor(df_val[class_name].tolist()).to(device)

    test_input_ids = test_encoded_inputs["input_ids"].to(device)
    test_attention_mask = test_encoded_inputs["attention_mask"].to(device)
    test_labels = torch.tensor(df_test[class_name].tolist()).to(device)

    # Model and data loaders
    num_labels = len(df_train[class_name].unique())
    model = XLNetForSequenceClassification.from_pretrained(
        "xlnet-base-cased", num_labels=num_labels
    ).to(device)

    batch_size = 64

    train_data = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = TensorDataset(val_input_ids, val_attention_mask, val_labels)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    test_data = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3)
    num_epochs = 20
    patience = 5
    best_val_loss = float("inf")
    early_stopping_counter = 0

    # Training and validation
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        with open(os.path.join(result_dir, f"{class_name}.txt"), "a") as f:
            f.write(
                f"Epoch {epoch + 1} | Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}\n"
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(), os.path.join(result_dir, f"{class_name}_model.pt")
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping")
                break

    # Load the best model
    model.load_state_dict(
        torch.load(os.path.join(result_dir, f"{class_name}_model.pt"))
    )

    # Test evaluation
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"{class_name}, acc={acc:.4f}, f1={f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(cm)


def main():
    result_dir = "./results/prompt"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    classes = ["beautiful", "clean"]

    for class_name in classes:
        training(class_name, result_dir)


if __name__ == "__main__":
    main()
