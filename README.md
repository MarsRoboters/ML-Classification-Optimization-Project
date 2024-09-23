# ML-Optimization-Project

In this notebook there are:
- Digit MNIST dataset
- Tools to visualize the samples
- Model: fully connected neural network with two hidden layer
- A train/validation function
- A random search for the hyperparameters (best model selection)
- Testing

Note: for our objectives the focus is on the model and on the function of the train. The testing part is important for us only for the output graph (what we need to get). Our dataset of interest is only the one related to the train and not validation or testing datasets: our goal is to minimize the loss function not to see the generalization of the model. So, in the following code there are some parts that we should NOT implement.



**Overview:**
  - This project involved developing a Feed Forward Neural Network (FFNN) to classify handwritten digits from the MNIST dataset. The model was trained to recognize 10 different digits, achieving an accuracy of 98.13% on the validation set.

**Introduction:**
  - Image classification is a fundamental task in computer vision with applications in various fields such as digit recognition, autonomous driving, and more. The MNIST dataset is a widely used benchmark for evaluating image classification algorithms.

**Problem Statement:**
  - The objective of this project was to develop a Feed Forward Neural Network (FFNN) to classify images from the MNIST dataset. The MNIST dataset consists of 70,000 28x28 grayscale images of handwritten digits in 10 different classes. The goal was to achieve high accuracy in recognizing these digits.

**Dataset:**
  - Dataset Used: MNIST
  - Number of Classes: 10
  - Classes: Digits 0 to 9
  - Training Samples: 60,000
  - Testing Samples: 10,000
  - Key Attributes: Pixel values and class labels

**Methodology:**
  - Data Preprocessing
    - Normalization: Pixel values were normalized to be between 0 and 1.
    - Transformation: Images were transformed into tensors for compatibility with PyTorch.
  - Model Architecture
    - The FFNN model was built using PyTorch with the following architecture:
      - Input Layer: 784 neurons (28x28 pixels).
      - Hidden Layers: Two hidden layers with configurable neurons.
      - Activation Function: Configurable (ReLU or Sigmoid).
      - Dropout Layers: Applied after each hidden layer to prevent overfitting.
      - Output Layer: 10 neurons (one for each digit).

**Training:**
  - Loss Function: CrossEntropyLoss.
  - Optimizers: Adam and AdaGrad optimizers were implemented.
  - Training Process: The model was trained for 50 epochs with a batch size of 50. The training process included data augmentation and callbacks for learning rate scheduling and early stopping.
  
**Techniques Implemented (for Generalization)**
  - Dropout: Applied dropout layers to prevent overfitting.
  - Batch Normalization: Applied after each hidden layer to stabilize and accelerate training.

**Key Results:**
  - The FFNN model achieved high accuracy on the test set. This performance indicates the model’s effectiveness in classifying handwritten digits into the correct categories.
    - Validation Loss: 0.0626
    - Accuracy on Validation Set: 98.13%

**Visualization:**
  - The training and validation accuracy and loss were plotted to visualize the model’s performance over epochs.

**Future Work:**
  - Experiment with Different Architectures: Try deeper or more complex models.
  - Hyperparameter Tuning: Optimize batch size, learning rate, and other parameters.
  - Transfer Learning: Utilize pre-trained models and fine-tune them on the MNIST dataset.
  - Ensemble Methods: Combine predictions from multiple models to improve accuracy.
