import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os

train_dir = "D:/Websites/Example1/LIDC-IDRI-Ppp/"
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, rotation_range=20, zoom_range=0.2, horizontal_flip=True, validation_split=0.2
)
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode="binary", subset="training"
)
val_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode="binary", subset="validation"
)
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
if train_generator.samples == 0:
    raise ValueError("No training images found!")

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers:
    layer.trainable = False
model = Sequential([
    base_model, Flatten(), Dense(256, activation="relu"), Dropout(0.5), Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
history = model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[early_stopping])
model.save("D:/Websites/Example1/lung_cancer_model_corrected.h5")
print("✅ Model trained and saved as 'lung_cancer_model_corrected.h5'")