import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- Paths ---
dataset_train = os.path.join('dataset', 'train')   # banana/dataset/train

if not os.path.exists(dataset_train):
    raise FileNotFoundError(f"Dataset train folder not found at: {dataset_train}")

# --- Data augmentation & generators ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,       # 20% of train used as validation
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

IMG_SIZE = (128, 128)
BATCH = 32

train_generator = train_datagen.flow_from_directory(
    dataset_train,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',   # multi-class
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    dataset_train,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = train_generator.num_classes
print(f"Detected classes ({num_classes}): {train_generator.class_indices}")

# --- Model (slightly deeper) ---
model = Sequential([
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')   # multi-class output
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train ---
EPOCHS = 30
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# --- Save model ---
model.save('banana_leaf_model.h5')
print("Model saved as banana_leaf_model.h5")
print("Training accuracy (last):", history.history['accuracy'][-1])
print("Validation accuracy (last):", history.history['val_accuracy'][-1])
