# -*- coding: utf-8 -*-
"""Brain Tumor Classification Using MRI Scans.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1w9Ejqh9q62eerb26Ziak7QABkflzjrzS

# **Libraries**
"""

import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import regularizers
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import MobileNetV2
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.utils import plot_model
from IPython.display import Image

import warnings
warnings.filterwarnings("ignore")

"""# **Loading Data**"""

def load_and_split_data(root_dir, categories):
    image_paths = []
    labels = []

    for category in categories:
        category_path = os.path.join(root_dir, category)
        for img_name in os.listdir(category_path):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(category_path, img_name))
                labels.append(category)

    df = pd.DataFrame({'Image': image_paths, 'Class': labels})

    train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Class'])

    valid_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=20, stratify=test_val_df['Class'])

    return train_df, valid_df, test_df

def print_data_info(train_df, valid_df, test_df):
    print("Data Split Information:")

    print(f"Total Training Images: {len(train_df)}")
    print(f"Total Validation Images: {len(valid_df)}")
    print(f"Total Testing Images: {len(test_df)}")

    print("\nCategory-wise Distribution:")
    for category in train_df['Class'].unique():
        print(f"Category: {category}")
        print(f"  Train: {len(train_df[train_df['Class'] == category])} images")
        print(f"  Validation: {len(valid_df[valid_df['Class'] == category])} images")
        print(f"  Test: {len(test_df[test_df['Class'] == category])} images")

root_directory = '/kaggle/input/brain-tumor-mri'
categories = ['glioma', 'healthy', 'meningioma', 'pituitary']
train_df, valid_df, test_df = load_and_split_data(root_directory, categories)
print_data_info(train_df, valid_df, test_df)

train_df

test_df

valid_df

"""# **Data Augmentation**"""

batch_size = 32
image_size = (299, 299)
image_shape = (299, 299, 3)

datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Image',
    y_col='Class',
    target_size=image_size,
    batch_size=batch_size,
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col='Image',
    y_col='Class',
    target_size=image_size,
    batch_size=batch_size,
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='Image',
    y_col='Class',
    target_size=image_size,
    batch_size=16,
    shuffle=False
)

"""# Visualization

### 1- Confusion Matrix
"""

def plot_confusion_matrix(model, test_generator):
    y_true = test_generator.classes
    y_pred = model.predict(test_generator, verbose=2)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

"""### 2- Classification Report"""

def print_classification_report(model, test_generator):
    y_true = test_generator.classes
    y_pred = model.predict(test_generator, verbose=2)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

"""### 3- Accuracy and Loss Curves"""

def plot_accuracy_and_loss(hist):
    # Accuracy plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label='Train Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def save_and_display_model(model,filename):
    plot_model(
    model,
    to_file=filename,
    show_shapes=True,
    show_layer_names=True,
    dpi=70,
)
    Image(filename=filename)

"""# **Model Building & Evaluation**

### 1- ResNet50
"""

resnet50 = ResNet50(include_top= False, weights= "imagenet",
                            input_shape= image_shape, pooling= 'max')

resnet50.trainable = True

model_1 = Sequential([
    resnet50,
    Flatten(),
    Dropout(rate= 0.3),
    Dense(128, activation= 'relu'),
    Dropout(rate= 0.25),
    Dense(4, activation= 'softmax')
])

model_1.compile(Adamax(learning_rate= 0.0001),
              loss= 'categorical_crossentropy',
              metrics= ['accuracy'])

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-6)
]

save_and_display_model(model_1, "ResNet50.png")

hist = model_1.fit(train_generator,
                 epochs=10,
                 validation_data=validation_generator,
                 shuffle= False,
                 callbacks=callbacks)

score_1 = model_1.evaluate(test_generator, verbose=2)
print(f"Test Loss: {score_1[0]:.2f}%")
print(f'Test accuracy: {score_1[1] * 100:.2f}%')

plot_confusion_matrix(model_1, test_generator)

print_classification_report(model_1, test_generator)

plot_accuracy_and_loss(hist)

"""### 2- InceptionV3"""

inception = InceptionV3(weights='imagenet', include_top=False, input_shape=image_shape)
inception.trainable = True


model_2 = Sequential([
    inception,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(rate= 0.4),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(rate= 0.2),
    Dense(4, activation= 'softmax')
])

model_2.compile(Adamax(learning_rate= 0.0001),
              loss= 'categorical_crossentropy',
              metrics= ['accuracy'])

save_and_display_model(model_2, "InceptionV3.png")

hist = model_2.fit(train_generator,
                 epochs=10,
                 validation_data=validation_generator,
                 shuffle= False,
                 callbacks=callbacks)

score_2 = model_2.evaluate(test_generator, verbose=2)
print(f"Test Loss: {score_2[0]:.2f}%")
print(f'Test accuracy: {score_2[1] * 100:.2f}%')

plot_confusion_matrix(model_2, test_generator)

print_classification_report(model_2, test_generator)

plot_accuracy_and_loss(hist)

"""### 3- VGG19"""

from tensorflow.keras.applications import VGG19
vgg19 = VGG19(include_top = False, weights = 'imagenet', input_shape = image_shape)

for layer in vgg19.layers:
    layer.trainable = False
vgg19.layers[-2].trainable = True
vgg19.layers[-3].trainable = True
vgg19.layers[-4].trainable = True

model_3 = Sequential([
    vgg19,
    Flatten(),
    Dropout(rate= 0.3),
    Dense(128, activation= 'relu'),
    Dropout(rate= 0.2),
    Dense(4, activation= 'softmax')
])
model_3.compile(Adamax(learning_rate= 0.0001),
              loss= 'categorical_crossentropy',
              metrics= ['accuracy'])

save_and_display_model(model_3, "VGG19.png")

hist = model_3.fit(train_generator,
                 epochs=10,
                 validation_data=validation_generator,
                 shuffle= False,
                 callbacks=callbacks)

score_3 = model_3.evaluate(test_generator, verbose=2)
print(f"Test Loss: {score_3[0]:.2f}%")
print(f'Test accuracy: {score_3[1] * 100:.2f}%')

plot_confusion_matrix(model_3, test_generator)

print_classification_report(model_3, test_generator)

plot_accuracy_and_loss(hist)

"""### 4- VGG16"""

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=image_shape, pooling='max')
vgg16.trainable = True

model_4 = Sequential([
    vgg16,
    Flatten(),
    Dropout(rate=0.3),
    Dense(128, activation='relu'),
    Dropout(rate=0.25),
    Dense(4, activation='softmax')
])

model_4.compile(
    optimizer=Adamax(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

save_and_display_model(model_4, "VGG16.png")

hist = model_4.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    shuffle=False,
    callbacks=callbacks
)

score_4 = model_4.evaluate(test_generator, verbose=2)
print(f"Test Loss: {score_4[0]:.2f}%")
print(f'Test accuracy: {score_4[1] * 100:.2f}%')

plot_confusion_matrix(model_4, test_generator)

print_classification_report(model_4, test_generator)

plot_accuracy_and_loss(hist)

"""### 5- MobileNetV2 with GridSearch"""

def create_model(learning_rate=0.0001, dropout_rate=0.5):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling='max')
    base_model.trainable = True

    model = Sequential([
        base_model,
        Flatten(),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer=Adamax(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

from sklearn.model_selection import ParameterGrid

def train_model_with_params(learning_rate, dropout_rate, epochs, batch_size):
    model = create_model(learning_rate=learning_rate, dropout_rate=dropout_rate)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        batch_size=batch_size,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc}")

    return test_acc

param_grid = {
    'learning_rate': [0.0001, 0.001],
    'dropout_rate': [0.3, 0.5],
    'epochs': [5, 10],
    'batch_size': [16, 32]
}

grid = ParameterGrid(param_grid)

best_accuracy = 0
best_params = {}

for params in grid:
    print(f"Training with parameters: {params}")
    accuracy = train_model_with_params(
        learning_rate=params['learning_rate'],
        dropout_rate=params['dropout_rate'],
        epochs=params['epochs'],
        batch_size=params['batch_size']
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

print(f"Best Accuracy: {best_accuracy}")
print(f"Best Parameters: {best_params}")

model_5 = create_model(learning_rate=0.001, dropout_rate=0.5)
hist = model_5.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    shuffle=False,
    callbacks=callbacks
)

save_and_display_model(model_5, "MobileNetV2.png")

score_5 = model_5.evaluate(test_generator, verbose=2)
print(f"Test Loss: {score_5[0]:.2f}%")
print(f'Test accuracy: {score_5[1] * 100:.2f}%')

plot_confusion_matrix(model_5,test_generator)

print_classification_report(model_5, test_generator)

plot_accuracy_and_loss(hist)

"""# **Comparing Accuracies**"""

scores=[score_1[1],score_2[1],score_3[1],score_4[1],score_5[1]]
models = ['ResNet50','InceptionV3','VGG19','VGG16','MobileNetV2']
scores_with_names = list(zip(models, scores))
sorted_scores_with_names = sorted(scores_with_names, key=lambda x: x[1], reverse=True)
print("Scores from highest to smallest accuracy:")
for rank, (name, score) in enumerate(sorted_scores_with_names, start=1):
    print(f"{rank}- {name}: {score *100 :.2f}%")

def plot_comparison_barplot(models, scores):

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, [score * 100 for score in scores], color='skyblue', edgecolor='black', linewidth=1)

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{score * 100:.1f}%", ha='center', va='bottom', fontsize=11)

    max_index = scores.index(max(scores))
    bars[max_index].set_color('dodgerblue')
    bars[max_index].set_edgecolor('navy')
    plt.text(bars[max_index].get_x() + bars[max_index].get_width() / 2,
             bars[max_index].get_height() + 10,
             "Best Model", ha='center', va='bottom', fontsize=12, color='darkred', fontweight='bold')

    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Comparison of Model Accuracies', fontsize=16, fontweight='bold')
    plt.ylim(0, 130)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


plot_comparison_barplot(models, scores)