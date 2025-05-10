import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, GlobalMaxPooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import warnings 
warnings.filterwarnings('ignore')


train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

X_train = train.drop(columns=['label'])
Y_train = train['label']

X_test = test.drop(columns=['label'])
Y_test = test['label']

print('X_train shape:', X_train.shape)
print('Y_train shape: ',Y_train.shape)
print('X_test shape: ', X_test.shape)
print('Y_test shape', Y_test.shape)

m = len(X_train)
fig, axes = plt.subplots(6,6, figsize=(10,8))
fig.tight_layout(pad=0.1)

for ax in axes.flat:
    random_idx = np.random.randint(m)
    random_img = X_train.iloc[random_idx].values.reshape(28,28)
    ax.imshow(random_img, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Image {Y_train[random_idx]}', fontsize=8)
# plt.show()


def create_model():
    model = Sequential()
    model.add(Input(shape=(784,)))
    model.add(Dense(units=1024, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.6))

    model.add(Dense(units=24, activation='softmax'))

    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = build_model()

model.load_weights('Model_Weights.keras')
model = load_model('Model.h5')

history = model.fit(X_train, Y_train, epochs=200, batch_size=64, validation_split=0.2, callbacks=[lr_scheduler])

model.save_weights('Model_Weights.keras')
model.save('Model.h5')


# Evaluation
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Loss: {loss} , Accuracy: {accuracy}')


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

train_vs_loss(history)

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
    
train_vs_validation(history)

def inference():
    # image_path = input('Enter the path of the image: ')
    image_path = r'C:\Users\Hacx\Desktop\maxresdefault.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print('Image is not found')
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image / 255.0
    image = image.reshape(1, 224, 224, 3)


    pred = model.predict(image)
    predicted_class = np.argmax(pred)
    asl_dictionary = {i: chr(65 + i) for i in range(26) if i != 9}  # Excludes 9 (for 'I')
    predicted_sign = asl_dictionary.get(predicted_class, 'Unknown Class')


    print(f'Predicted ASL sign class (index): {predicted_class}')
    print(f'Predicted ASL sign (letter): {predicted_sign}')

inference()


