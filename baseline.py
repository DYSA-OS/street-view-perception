import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os


def create_data_generator(
    df, class_name, batch_size=64, target_size=(224, 224), shuffle=False
):
    df["img_path"] = df["filename"].apply(
        lambda val: os.path.join("./place-pulse", val)
    )
    df[class_name] = df[class_name].astype(str)

    datagen = ImageDataGenerator()
    generator = datagen.flow_from_dataframe(
        df,
        x_col="img_path",
        y_col=class_name,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=shuffle,
    )
    return generator


def training(class_name, result_dir):
    train_df = pd.read_csv("./data/origin/train.csv")
    val_df = pd.read_csv("./data/origin/val.csv")
    test_df = pd.read_csv("./data/origin/test.csv")

    train_generator = create_data_generator(train_df, class_name)
    val_generator = create_data_generator(val_df, class_name)
    test_generator = create_data_generator(test_df, class_name)

    base_model = DenseNet121(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(3, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    csv_logger = CSVLogger(os.path.join(result_dir, f"{class_name}.csv"), append=True)

    model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=[early_stopping, csv_logger],
    )
    model.save(os.path.join(result_dir, f"{class_name}_model.h5"))

    test_generator.reset()

    predictions = model.predict(test_generator)
    y_true = test_generator.classes
    y_pred = np.argmax(predictions, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"{class_name}, acc={acc:.4f}, f1={f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(cm)


def main():
    result_dir = "./results/baseline"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    classes = ["beautiful", "clean"]

    for class_name in classes:
        training(class_name, result_dir)


if __name__ == "__main__":
    main()
