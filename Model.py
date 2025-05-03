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


def build_model():
    X_input = Input(shape=(28, 28, 1), name='Input')

    #Layer 1
    X = Conv2D(16, (3, 3), kernel_initializer='he_normal', kernel_regularizer=l2(0.02))(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.3)(X)

    #Layer 2
    X = Conv2D(32, (3, 3), kernel_initializer='he_normal', kernel_regularizer=l2(0.02))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.3)(X)

    #Layer 3
    X = Conv2D(64, (3, 3), kernel_initializer='he_normal', kernel_regularizer=l2(0.02))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.4)(X)

    X = GlobalMaxPooling2D()(X)

    # FC Layer
    X = Dense(units=32, kernel_initializer='he_normal', kernel_regularizer=l2(0.02))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    
    output = Dense(units=24, activation='softmax')(X)
    model = Model(inputs=X_input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.002), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
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


