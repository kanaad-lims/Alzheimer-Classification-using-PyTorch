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
**[Alzheimer’s Classification Dataset](https://www.kaggle.com/datasets/kanaadlimaye/alzheimers-classification-dataset)** {Upvote the dataset if you find it helpful :) }

Initially the dataset was uploaded to Roboflow, where preprocessing and image augmentations were performed.

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
> [!IMPORTANT]
> The dataset is not included in this repository due to size limitations.  
  Please download it from Kaggle (using link above) and place it in the appropriate directory before running the code.

---

## Tech Stack:
- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- PIL

---
## Project Structure

```
Alzheimer-Classification-using-PyTorch/
│
├── datasets/
│   └── alzheimer_dataset.py
│
├── models/
│   └── alzheimer_model.py
│
├── training/
│   ├── train.py
│   └── evaluate.py
│
├── utils/
│   └── preprocessing.py
│
├── .gitignore
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

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
    ```
    Input Image (3 × H × W tensor)
        ↓
    ResNet-152 Backbone (ImageNet pretrained feature extractor)
            ↓
    Global Average Pooling
            ↓
    2048-dimensional feature vector
            ↓
    Batch Normalization
            ↓
    Dense(512)
            ↓
    Dense(256)
            ↓
    Dropout(0.5)
            ↓
    Dense(128)
            ↓
    Dropout(0.3)
            ↓
    Dense(64)
            ↓
    Dense(4)  (Class logits)
            ↓
    Softmax (implicitly applied within CrossEntropyLoss during training)
   ```

3. **Training**
   - GPU: Nvidia Tesla T4 x 2 (Kaggle Notebooks)
   - Optimizer: `Adam`  
   - Loss: `cross entropy loss`  
   - Batch size: 32

4. **Evaluation**
   - Accuracy & Loss curves for train/validation.
   - Test set evaluation for generalization.
   - Classification report & confusion matrix.(F1 score, Precision, Recall) 

---

##  Results
- **Training Accuracy**: ~97%  
- **Validation Accuracy**: ~94%  
- **Test Accuracy**: ~93%  
- **Test Loss**: ~0.22  

The results indicate the model generalizes well with minimal overfitting.  

Example accuracy curves:

<img width="700" height="547" alt="download" src="https://github.com/user-attachments/assets/08fa85b7-e83a-4741-ba4d-5b9e3b48734f" />

---

##  Evaluation Metrics
- **Confusion Matrix**:
  ```
  Classification Report:
  
                precision    recall  f1-score   support
  
  MD            0.92        0.94      0.93        86
  MoD           1.00        0.80      0.89         5
  ND            0.94        0.96      0.95       319
  VMD           0.94        0.91      0.92       230
  
  accuracy                            0.97       640
  macro avg     0.95        0.90      0.95       640
  weighted avg  0.94        0.94      0.95       640
  ```
---

##  How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/kanaad-lims/Alzheimer-Classification-using-PyTorch.git
   cd Alzheimer-Classification-using-PyTorch
2. Create a virtual environment with python version == 3.10+.
   ```bash
   cd Alzheimer-Classification-using-PyTorch
   python3.10 -m venv venv310
3. For Anaconda users - create a conda environment with the python version as 3.10+.
   ```bash
   conda create -n myEnv python=3.10
   conda activate myEnv
4. After the virtual environment is set, install all required dependencies from the requirements.txt
   ```bash
   pip install -r requirements.txt
   ```
   `Make sure that you have the appropriate CUDA toolkit installed for the PyTorch version if training locally.`
   
5. When all the dependencies are installed, launch the main.py file.
   ```bash
   python main.py
   ```
