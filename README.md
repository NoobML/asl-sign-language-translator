# ASL (American Sign Language) Recognition using Transfer Learning

## Project Overview

This project implements a deep learning solution for recognizing American Sign Language (ASL) hand gestures using computer vision and transfer learning techniques. The model achieves high accuracy by leveraging a pre-trained ResNet50 architecture fine-tuned specifically for ASL classification.

## 🎯 Results

- **Training Accuracy**: 99%
- **Validation Accuracy**: 99%
- **Test Accuracy**: 95%
- **Total Trainable Parameters**: 3.69M
  - Custom Dense layers: ~2.63M parameters
  - Unfrozen ResNet block: ~1.06M parameters

## 📊 Dataset

The project uses the Sign Language MNIST dataset, which contains:
- **Training data**: Hand gesture images representing ASL letters
- **Test data**: Separate test set for model evaluation
- **Image format**: 28x28 grayscale images
- **Classes**: 24 ASL letters (excluding J and Z due to motion requirements)

## Data Source

The dataset used in this project was taken from Kaggle:  
[American Sign Language Dataset](https://www.kaggle.com/datasets/deeppythonist/american-sign-language-dataset)



## 🏗️ Architecture

### Transfer Learning Approach

The model employs **ResNet50** as the backbone architecture with the following modifications:

1. **Base Model**: Pre-trained ResNet50 (ImageNet weights)
2. **Input Adaptation**: Images resized from 28x28 to 32x32 and converted to RGB
3. **Selective Fine-tuning**: Only the final convolutional block (`conv5_block3_3_conv` onwards) is unfrozen
4. **Custom Classification Head**:
   - Global Max Pooling layer
   - Dense layer (1024 neurons, ReLU activation, L2 regularization)
   - Dense layer (512 neurons, ReLU activation, L2 regularization)
   - Output layer (24 neurons, Softmax activation)

### Key Technical Decisions

- **Image Preprocessing**: ResNet50's `preprocess_input` function for proper normalization
- **Regularization**: L2 regularization (0.001) to prevent overfitting
- **Optimizer**: Adam optimizer with low learning rate (1e-5)
- **Learning Rate Scheduling**: Decay scheduler for stable convergence

## 🔄 Data Pipeline

### Preprocessing Steps

1. **Reshape**: Convert flattened pixel arrays back to 28x28 images
2. **Resize**: Scale images to 32x32 for ResNet50 compatibility
3. **Color Conversion**: Transform grayscale to RGB format
4. **Normalization**: Apply ResNet50 preprocessing

### Preprocessing Optimization

**Efficient Data Caching Strategy:**
```python
# One-time preprocessing and caching
x_train_data_processed = preprocess_images(X_train)
x_test_data_processed = preprocess_images(X_test)

# Save preprocessed data to avoid repeated computation
np.save('x_train_data_processed.npy', x_train_data_processed)
np.save('x_test_data_processed.npy', x_test_data_processed)

# Load preprocessed data for training
x_train_data_processed = np.load('x_train_data_processed.npy')
x_test_data_processed = np.load('x_test_data_processed.npy')
```

## 📝 Notes

- **Preprocessing Optimization**: Images are preprocessed once and cached as `.npy` files to avoid redundant computation during training epochs
- Model weights are saved for future use without retraining
- **Efficient Data Pipeline**: Preprocessed data loading eliminates bottlenecks and speeds up training
- Interactive inference allows testing with custom images
- Comprehensive visualization aids in model analysis
- **Performance Benefits**: Caching reduces training time and computational overhead significantly

This implementation demonstrates effective use of transfer learning, proper data preprocessing, and practical deployment considerations for computer vision applications in ASL recognition.

**Benefits of Preprocessing Caching:**
- **Training Efficiency**: Eliminates repetitive preprocessing operations during each epoch
- **Memory Optimization**: Preprocessed data is loaded once and reused throughout training
- **Time Savings**: Significantly reduces training time by avoiding redundant computations
- **Consistency**: Ensures identical preprocessing across all training runs
- **Resource Management**: Reduces CPU load during training, allowing focus on GPU computation

### Data Augmentation

Custom data generator implements real-time augmentation:
- Random horizontal flipping
- Random vertical flipping
- Random brightness adjustment (±10%)
- Random contrast variation (90%-110%)

### Data Splitting

- **Training Set**: 80% of original training data
- **Validation Set**: 20% of original training data
- **Test Set**: Separate test dataset
- **Encoding**: One-hot encoding for multi-class classification

## 🚀 Training Process

### Training Configuration

```python
# Model compilation
optimizer = Adam(learning_rate=1e-5)
loss = 'categorical_crossentropy'
metrics = ['accuracy']

# Training parameters
epochs = 50
batch_size = 256
```

### Learning Rate Strategy

Implements exponential decay scheduling:
```python
learning_rate = initial_lr / (1 + decay_rate × epoch)
```

### Callbacks

- **LearningRateScheduler**: Automatically adjusts learning rate during training

## 🔍 Model Performance

### Training Metrics Visualization

The project includes comprehensive visualization functions:

1. **Training Accuracy vs Loss**: Monitors model convergence
2. **Training vs Validation Accuracy**: Detects overfitting/underfitting

### Performance Analysis

- **High Training/Validation Accuracy**: Indicates effective learning
- **Good Generalization**: 95% test accuracy shows robust performance
- **Minimal Overfitting**: Close training and validation accuracies

## 🎮 Inference System

### Real-time Prediction

Interactive inference function that:
1. Accepts image path input from user
2. Preprocesses the image (resize, normalize, reshape)
3. Generates prediction using trained model
4. Maps prediction to corresponding ASL letter
5. Provides continuous prediction loop with exit option

### ASL Mapping

```python
asl_dictionary = {i: chr(65 + i) for i in range(26) if i != 9}
# Maps class indices to letters A-Z (excluding J)


```
## 📁 Project Structure

```
├── .gitattributes                # LFS tracking config for large files
├── README.md                    # Project documentation
├── sign_mnist_train.csv         # Training dataset
├── sign_mnist_test.csv          # Test dataset
├── ASL_PreTrained_Model.h5      # Saved complete model
├── ASL_PreTrained_Model_weights.h5  # Model weights
└── x_train_data_processed.npy   # Preprocessed training data
├── x_test_data_processed.npy    # Preprocessed test data
├── sign_language_model.py       # Sign language model script
```


## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib
- **Transfer Learning**: ResNet50 (ImageNet pre-trained)

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
```

### Usage

1. **Training**: Run the main script to train the model
2. **Inference**: Use the interactive inference function for real-time predictions
3. **Evaluation**: Model automatically evaluates on test set

### Running Inference

```python
# The inference function provides an interactive loop
inference()
# Enter image path when prompted
# View prediction results
# Enter 'x' to exit
```

## 🔬 Technical Highlights

### Transfer Learning Strategy

- **Frozen Layers**: Lower layers retain ImageNet features
- **Fine-tuned Layers**: Final block adapts to ASL-specific features
- **Efficient Training**: Reduces computation while maintaining accuracy

### Data Efficiency

- **Preprocessing Caching**: One-time preprocessing with `.npy` file storage eliminates redundant computations during training
- **Augmentation**: Increases dataset diversity without additional data
- **Batch Processing**: Custom generator for memory-efficient training
- **Smart Caching**: Preprocessed data loaded once and reused across all epochs

### Model Optimization

- **Regularization**: Prevents overfitting with L2 penalties
- **Learning Rate Scheduling**: Ensures stable convergence
- **Architecture Design**: Balanced complexity for ASL classification

## 📈 Future Improvements

- Implement real-time webcam inference
- Add support for dynamic gestures (J, Z)
- Explore other pre-trained architectures
- Deploy as web application
- Add confidence scoring for predictions

