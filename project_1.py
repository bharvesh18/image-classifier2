import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers,models
import joblib
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Validation data shape:", x_val.shape)
print("Validation labels shape:", y_val.shape)
print("Testing data shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
from sklearn.metrics import classification_report
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
print("Classification Report Of Model:")
print(report)
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest')
augmented_model = model
augmented_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
augmented_history = augmented_model.fit(datagen.flow(x_train, y_train, batch_size=64),epochs=10,validation_data=(x_val, y_val))
augmented_test_loss, augmented_test_acc = augmented_model.evaluate(x_test, y_test)
print("Test accuracy (augmented model):", augmented_test_acc)
model.save('models/image.h5')
augmented_model.save('models/image_classifier.h5')