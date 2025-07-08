# src/training/train.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def treinar_modelo():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1
    )

    train_data = datagen.flow_from_directory(
        'data/processed',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        'data/processed',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(train_data.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, validation_data=val_data, epochs=10)

    # Salvar o modelo
    model.save("models/modelo_cnn.h5")

    # Obter a acurácia final
    final_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    # Salvar em arquivo .txt
    with open("models/acuracia_final.txt", "w") as f:
        f.write(f"Acurácia (treinamento): {final_acc:.4f}\n")
        f.write(f"Acurácia (validação): {final_val_acc:.4f}\n")

    # Exibir no terminal
    print(f"Modelo treinado com acurácia de treino: {final_acc:.4f}")
    print(f"Acurácia de validação: {final_val_acc:.4f}")
