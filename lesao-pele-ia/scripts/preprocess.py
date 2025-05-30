import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

IMG_SIZE = (224, 224)

def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['dx'].notna()]
    return df

def preprocess_images(df, img_dir):
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, f"{row['image_id']}.jpg")
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=IMG_SIZE)
            img = img_to_array(img) / 255.0  # Normalização
            images.append(img)
            labels.append(row['dx'])

    return np.array(images), np.array(labels)

def encode_labels(labels):
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    return encoded, encoder

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def augment_data():
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

if __name__ == "__main__":
    CSV_PATH = "./data/HAM10000_metadata.csv"
    IMG_DIR = "./data/ham10000_images/"

    print("📄 Carregando metadados...")
    df = load_metadata(CSV_PATH)

    print("🖼️ Carregando imagens...")
    X, y = preprocess_images(df, IMG_DIR)

    print("🔤 Codificando classes...")
    y_encoded, encoder = encode_labels(y)

    print("📊 Dividindo conjunto de dados...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_encoded)

    print("✅ Pré-processamento finalizado.")
