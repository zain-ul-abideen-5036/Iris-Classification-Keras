# Iris Flower Species Classification with Keras

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-brightgreen)
A neural network model to classify iris flowers into 3 species using Keras.
---

## Overview
This project uses the **Iris dataset** to train a neural network model for species classification. The model achieves **~90% validation accuracy** and **73.9% test accuracy**. The repository includes data preprocessing, model training, and evaluation steps.
---

## Key Features
- **Data Splitting**: 70% training, 15% validation, 15% test.
- **Model Architecture**:  
  - 1 hidden layer with ReLU activation.
  - Output layer with Softmax activation for multi-class classification.
- **Training**: 10 epochs, batch size of 8, Adam optimizer.
- **Evaluation**: Test accuracy and loss metrics.
---

## Dataset
The Iris dataset contains 150 samples with 4 features:  
- Sepal length, sepal width, petal length, petal width.  
- Target classes: Setosa, Versicolor, Virginica.
---

##  Model Architecture
```python
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(3, activation='softmax')
])
```
- Loss Function: ```sparse_categorical_crossentropy```
- Optimizer: Adam
- Metrics: Accuracy
---

## Results
- Validation Accuracy: 90.9% (Epoch 10)
- Test Accuracy: 73.9%
---

## Repository Structure
```
Iris-Classification-Keras/
├── notebooks/
│   └── Iris_Classification.ipynb  # Jupyter notebook file
├── README.md
└── requirements.txt  # List of dependencies
```
---

## Installation
1. Clone the repository:
```
git clone https://github.com/zain-ul-abideen-5036/Iris-Classification-Keras.git
cd Iris-Classification-Keras
```
2. Install dependencies:
```
pip install -r requirements.txt
```
---

## Usage
1. Run the Jupyter notebook:
```
jupyter notebook notebooks/Iris_Classification.ipynb
``` 
2. Follow the code cells to preprocess data, train the model, and evaluate performance.
---

## Contact
For questions or feedback, email: abideen5036@gmail.com
---

