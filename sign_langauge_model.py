import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.applications.resnet50 import preprocess_input
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

def split_features_labels(data):
    X = data.drop(columns=['label'])
    Y = data['label']
    return X, Y

X_train , Y_train = split_features_labels(train)
X_test , Y_test = split_features_labels(test)

def preprocess_images(data):

    data = data.to_numpy()
    images = np.array([img.reshape(28,28,1) for img in data])
    images_resize = tf.image.resize(images, (32,32))
    images_tensor = tf.convert_to_tensor(images_resize)
    images_rgb = tf.image.grayscale_to_rgb(images_tensor).numpy()
    images_preprocessed = preprocess_input(images_rgb)

    return images_preprocessed

# x_train_data_processed = preprocess_images(X_train)
# x_test_data_processed = preprocess_images(X_test)

# #saving the processed data
# np.save('x_train_data_processed.npy', x_train_data_processed)
# np.save('x_test_data_processed.npy', x_test_data_processed)

x_train_data_processed = np.load('x_train_data_processed.npy')
x_test_data_processed = np.load('x_test_data_processed.npy')


OHE = OneHotEncoder()
Y_train = OHE.fit_transform(Y_train.values.reshape(-1, 1)).toarray()
Y_test = OHE.transform(Y_test.values.reshape(-1, 1)).toarray()
x_train, x_val, y_train, y_val = train_test_split(x_train_data_processed, Y_train, train_size=0.8, random_state=42)

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_data, y_data, batch_size=256):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
    def __len__(self):
        return int(np.ceil(len(self.x_data) / self.batch_size))

    def augment_batch(self, batch):
        batch = tf.image.random_flip_left_right(batch)
        batch = tf.image.random_flip_up_down(batch)
        batch = tf.image.random_brightness(batch, max_delta=0.1)
        batch = tf.image.random_contrast(batch, lower=0.9, upper=1.1)

        return batch
    def __getitem__(self, idx):
        batch_x = self.x_data[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y_data[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.augment_batch(batch_x)

        return batch_x, batch_y


train_generator = CustomDataGenerator(x_train, y_train)
val_generator = CustomDataGenerator(x_val, y_val)

def rate_decay(epoch):
    initial_learning_rate = 1e-7
    decay_rate = 1
    learning_rate_decay = initial_learning_rate / (1 + decay_rate * epoch)
    return learning_rate_decay

Lr_Scheduler = LearningRateScheduler(rate_decay)

def Resnet50_Model():
    base_model = tf.keras.applications.ResNet50(
        input_shape=(32, 32, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    set_trainable = False

    for layer in base_model.layers:
        if layer.name == 'conv5_block3_3_conv':
            set_trainable = True
        if set_trainable:
            layer.trainable = True

    X = base_model.output
    X = GlobalMaxPooling2D()(X)
    X = Dense(1024, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.1))(X)
    X = Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.git p1))(X)
    X = Dense(24, activation='softmax')(X)

    return Model(inputs=base_model.input, outputs=X)

model = Resnet50_Model()
print(model.summary())
model.load_weights('ASL_PreTrained_Model_weights.h5')

model.compile(optimizer=Adam(learning_rate=1e-7), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[Lr_Scheduler])
model.save('ASL_PreTrained_Model.h5')
model.save_weights('ASL_PreTrained_Model_weights.h5')

loss, accuracy = model.evaluate(x_test_data_processed, Y_test)
print(f'Accuracy: {accuracy}  loss: {loss}')


# Plot: Training Accuracy & Loss
def train_vs_loss(h):
    epochs = range(len(h.history['accuracy']))
    plt.figure(figsize=(8, 7))
    plt.plot(epochs, h.history['accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, h.history['loss'], 'r*-', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training Accuracy vs Loss')
    plt.legend()
    plt.show()

# Plot: Training vs Validation Accuracy
def train_vs_validation(h):
    epochs = range(len(h.history['accuracy']))
    plt.figure(figsize=(8, 7))
    plt.plot(epochs, h.history['accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, h.history['val_accuracy'], 'r*-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.show()

train_vs_loss(history)
train_vs_validation(history)

def inference():

    while True:

        image_path = input('Enter the path of the image: ')
        image = cv2.imread(image_path)

        if image is None:
            print('Image is not found')
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image / 255.0
        image = image.reshape(1, 32, 32, 3)


        pred = model.predict(image)
        predicted_class = np.argmax(pred)
        asl_dictionary = {i: chr(65 + i) for i in range(26) if i != 9}  # Excludes 9 (for 'I')
        predicted_sign = asl_dictionary.get(predicted_class, 'Unknown Class')


        print(f'Predicted ASL sign class (index): {predicted_class}')
        print(f'Predicted ASL sign (letter): {predicted_sign}')

        repeat = input('Enter "x" to break loop: ')
        if repeat == 'x':
            break

inference()


