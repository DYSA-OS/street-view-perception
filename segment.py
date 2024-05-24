import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os


def training(class_name, result_dir):
    df_train = pd.read_csv('./data/hrnet_train.csv')
    df_val = pd.read_csv('./data/hrnet_val.csv')
    df_test = pd.read_csv('./data/hrnet_test.csv')

    X_train = df_train.copy()[['tree', 'road', 'sky', 'grass', 'person', 'sidewalk']]
    y_train = df_train[class_name]

    X_val = df_val.copy()[['tree', 'road', 'sky', 'grass', 'person', 'sidewalk']]
    y_val = df_val[class_name]

    X_test = df_test.copy()[['tree', 'road', 'sky', 'grass', 'person', 'sidewalk']]
    y_test = df_test[class_name]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)

    val_preds = xgb.predict(X_val)

    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average='weighted')

    y_pred = xgb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    with open('./results/test.txt', 'a') as f:
        f.write(f'segment, {class_name}, {acc:.6f}, {f1:.6f}\n')

    print(f'CLASS => {class_name}')
    print(f'Val Accuracy: {val_acc:.2f}')
    print(f'Val F1 Score: {val_f1:.2f}\n')
    print(f'Test Accuracy: {acc:.2f}')
    print(f'Test F1 Score: {f1:.2f}\n')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(result_dir, f'{class_name}.png'))


def main():
    result_dir = './results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    segment_dir = './results/segment'
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)

    classes = ['beautiful', 'clean']

    for class_name in classes:
        training(class_name, segment_dir)


if __name__ == '__main__':
    main()