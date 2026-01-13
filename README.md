# 🫁 Pneumonia Detection & Lung Severity Analysis using depp learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📌 Project Overview
This project focuses on the automated detection of pneumonia from chest X-ray images and goes a step further to quantify the **severity of the infection**. 

Unlike standard classification models that only predict "Positive" or "Negative," this system utilizes a **3-Phase Architecture** combining hybrid Deep Learning approaches, end-to-end fine-tuning, and clinical severity analysis using heatmaps.

---

## ⚙️ System Architecture

The project is executed in three distinct phases:

### **Phase 1: Hybrid Classification (Feature Extraction + ML)**
* **Goal:** Establish a strong baseline using deep features combined with traditional ML power.
* **Method:** 1.  Used **DenseNet-121** (pre-trained on ImageNet) as a feature extractor (removed the top classification layer).
    2.  Extracted feature vectors from the Global Average Pooling layer.
    3.  Trained classical ML classifiers on these features: **SVM, Logistic Regression, Random Forest, KNN, and Naive Bayes**.
    4.  Selected the best-performing ML model based on accuracy metrics.

### **Phase 2: End-to-End Deep Learning**
* **Goal:** Maximize performance by allowing the neural network to learn task-specific medical features.
* **Method:**
    1.  Fine-tuned the **DenseNet-121** model specifically for the Pneumonia dataset.
    2.  **Data Split:** 80% Training, 10% Validation, 10% Testing.
    3.  **Optimization:** Performed Hyperparameter Tuning (Learning Rate, Batch Size, Dropout) to minimize loss and prevent overfitting.

### **Phase 3: Lung Severity Analysis**
* **Goal:** Quantify the infection extent for clinical insights.
* **Method:**
    1.  Generated **Grad-CAM / Class Activation Maps (CAM)** to visualize infected regions.
    2.  Applied thresholding to isolate high-activation areas (infection hotspots).
    3.  Calculated the **Severity Percentage**:
        $$\text{Severity} (\%) = \frac{\text{Infected Area}}{\text{Total Lung Area}} \times 100$$
    4.  Categorized risk as **Low, Medium, or High**.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras (DenseNet-121)
* **Machine Learning:** Scikit-Learn (SVM, RF, KNN, etc.)
* **Computer Vision:** OpenCV (Image preprocessing)
* **Visualization:** Matplotlib, Seaborn (Heatmaps, Confusion Matrices)
* **Dataset Source:** Kaggle Chest X-Ray Images (Pneumonia)

---

## 🚀 How to Run

### **Step 1: Get Kaggle API Credentials**
Since the dataset is downloaded directly from Kaggle, you will need an API key.
1. Log in to your [Kaggle account](https://www.kaggle.com/).
2. Go to **Settings** (click on your profile picture in the top right).
3. Scroll down to the **API** section.
4. Click **"Create New Token"**.
5. This will download a file named `kaggle.json`. **Keep this file ready.**

### **Step 2: Clone the Repository**
```bash
git clone [https://github.com/SatyaKrishna2811/Pneumonia-Severity-Analysis.git](https://github.com/SatyaKrishna2811/Pneumonia-Severity-Analysis.git)
cd Pneumonia-Severity-Analysis
```
### **Step 3: Run the Code**
1. Open the notebook (e.g., in Google Colab or Jupyter).
2. Run the initial cells.
3. When prompted by the code to upload the API key, upload the `kaggle.json` file you downloaded in Step 1.
4. The script will automatically configure the permissions and download the dataset.

---

## 📊 Results & Performance

| Model Approach | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :---   | :--- |
| **Phase 1 (Best ML Model)** | 95.5% | 0.94 | 0.96 | 0.93 |
| **Phase 2 (Fine-Tuned DenseNet)** | 90.0% | 0.88 | 0.96 | 0.92 |

---

👥 Contributors
This project was developed as part of the AI Optimization Techniques for Healthcare Resource Management (OTH) course at Amrita Vishwa Vidyapeetham.

Vepuri Satya Krishna (DL.AI.U4AID24140)

Gowripriya R (DL.AI.U4AID24113)

Yaalini R (DL.AI.U4AID24043)

---

> *This project is for educational and research purposes only and should not be used as a primary diagnostic tool without clinical validation.*
