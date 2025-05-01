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

