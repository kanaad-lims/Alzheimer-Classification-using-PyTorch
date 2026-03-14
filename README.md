# Alzheimer-Classification-using-PyTorch

This repository contains an implementation of **Alzheimer’s disease stage classification** using a ResNet-based deep learning model written in PyTorch.
The model was trained and evaluated on MRI brain scan images, with the goal of distinguishing between different stages of Alzheimer’s disease.

---

##  Problem Statement
Alzheimer’s disease is a progressive neurological disorder that affects memory and cognition.  
Early and accurate detection of its stages can help clinicians in diagnosis and treatment planning.  

This project was developed as a part of **IEEE EMBS Internship** program. 

---

##  Dataset
The dataset used is from Kaggle:  
**[Alzheimer’s Classification Dataset](https://www.kaggle.com/datasets/kanaadlimaye/alzheimers-classification-dataset)**  

Initially the dataset was uploaded to Roboflow, where, the Preprocessing steps and Image Augmentations were performed.

- Images are categorized into four classes:
  - **MD**: Mild Demented  
  - **MoD**: Moderate Demented  
  - **ND**: Non Demented  
  - **VMD**: Very Mild Demented  

The dataset was split into:
- **Training set**
- **Validation set**
- **Test set**

CSV files provided contain filenames and one-hot encoded class labels.

---

##  Project Workflow
1. **Data Preparation**
   - Loaded CSV files for train/validation/test splits.
   - Converted one-hot encodings into class labels.
   - Converted class labels into numerical values.
   - Applied **image preprocessing and augmentation**:
     - Resize to 224×224
     - Horizontal/vertical flips
     - Rotation (±15°)
     - Normalization

2. **Model Architecture**
   - Used **ResNet** (transfer learning) with pre-trained ImageNet weights.
   - 5 FC layers with the heirarcical decreasing neurons structure (512, 256, 128, 64).
   - Added fully-connected layers with **Dropout** for regularization.

3. **Training**
   - GPU: T4 x 2 (Kaggle Notebooks)
   - Optimizer: `Adam`  
   - Loss: `cross entropy loss`  
   - Batch size: 32

5. **Evaluation**
   - Accuracy & Loss curves for train/validation.
   - Test set evaluation for generalization.
   - Classification report & confusion matrix.(F1 score, Precision, Recall) 

---

##  Results
- **Training Accuracy**: ~94%  
- **Validation Accuracy**: ~92%  
- **Test Accuracy**: ~93%  
- **Test Loss**: ~0.22  

The results indicate the model generalizes well with minimal overfitting.  

Example accuracy curves:

<img width="700" height="547" alt="download" src="https://github.com/user-attachments/assets/08fa85b7-e83a-4741-ba4d-5b9e3b48734f" />

---

##  Evaluation Metrics
- **Confusion Matrix**: Shows class-level predictions and misclassifications.
- **Classification Report**: Precision, Recall, F1-score for each class.

---

##  How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/kanaad-lims/Alzheimer-Classification-using-PyTorch.git
   cd Alzheimer-Classification-using-PyTorch
2. Create a virtual environment with python version == 3.10 (since tensorflow and keras version conflicts with the newer Python 3.13).
   ```bash
   cd Alzheimer-Classification-using-PyTorch
   python3.10 -m venv venv310
3. For Anaconda users - create a conda environment with the python version as 3.10 (same reason as stated above).
   ```bash
   conda create -n myEnv python=3.10
   conda activate myEnv
4. After the virtual environment is set, install the required dependencies as in the notebook.
   
5. When all the dependencies are installed, launch the main.py file.
   ```bash
   python main.py
