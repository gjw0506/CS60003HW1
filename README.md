# CS60003HW1
## EuroSAT Image Classification with MLP (Manual Backpropagation)

This repository implements a Multi-Layer Perceptron (MLP) from scratch (without deep learning frameworks like PyTorch/TensorFlow) for classifying the EuroSAT_RGB satellite image dataset. The model uses manual backpropagation, stochastic gradient descent (SGD) with learning rate decay, L2 regularization, and includes comprehensive training/evaluation pipelines.

## Environment Setup

### 1. Create a Virtual Environment
```bash
# Create venv (Python 3.8+)
python -m venv venv

# Activate venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install numpy Pillow matplotlib scikit-learn seaborn
```



### 3. Prepare the Dataset
1. Download the EuroSAT_RGB dataset:  
2. Extract the zip file and place the `EuroSAT_RGB` folder in the root directory of the project.  
   Project structure should look like:
   ```
   eurosat-mlp-classification/
   ├── EuroSAT_RGB/
   │   ├── AnnualCrop/
   │   ├── Forest/
   │   ├── ... (other classes)
   ├── hw1.py
   ├── requirements.txt
   └── README.md
   ```


## Full Training Pipeline
- Data loading & train/val/test split
- Hyperparameter grid search (lr, hidden size, weight decay)
- Full training with best hyperparameters
- Training curve visualization
- Test set evaluation (accuracy + confusion matrix)
- Weight visualization & error analysis

Run the script:
```bash
python hw1.py
```

## Key Features
- ✅ Manual implementation of MLP (forward/backward pass without DL frameworks)
- ✅ Support for ReLU/Sigmoid activation functions
- ✅ SGD with learning rate decay
- ✅ L2 regularization (weight decay)
- ✅ Hyperparameter grid search (learning rate, hidden size, regularization)
- ✅ Comprehensive evaluation:
  - Confusion matrix (visualization + text output)
  - First layer weight visualization
  - Misclassified sample analysis (error analysis)

