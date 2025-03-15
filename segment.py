import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import os


def training(class_name):
    df_train = pd.read_csv("./data/hrnet/train.csv")
    df_val = pd.read_csv("./data/hrnet/val.csv")
    df_test = pd.read_csv("./data/hrnet/test.csv")

    X_train = df_train.copy()[["tree", "road", "sky", "grass", "person", "sidewalk"]]
    y_train = df_train[class_name]

    X_val = df_val.copy()[["tree", "road", "sky", "grass", "person", "sidewalk"]]
    y_val = df_val[class_name]

    X_test = df_test.copy()[["tree", "road", "sky", "grass", "person", "sidewalk"]]
    y_test = df_test[class_name]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)

    val_preds = xgb.predict(X_val)

    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="weighted")
    print(f"[val] acc={val_acc:.4f}, f1={val_f1:.4f}")

    y_pred = xgb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"{class_name}, acc={acc:.4f}, f1={f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(cm)


def main():
    classes = ["beautiful", "clean"]

    for class_name in classes:
        training(class_name)


if __name__ == "__main__":
    main()
