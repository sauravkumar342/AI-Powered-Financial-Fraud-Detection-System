# AI-Powered-Financial-Fraud-Detection-System
An end-to-end **Fraud Detection System** built using **Machine Learning, Deep Learning, and Graph-Based Analytics** to detect fraudulent financial transactions and uncover hidden fraud patterns such as fraud rings.


## Overview

This project aims to identify fraudulent transactions in financial systems using advanced AI techniques. It combines:

- Traditional Machine Learning models
- Deep Learning (ANN/LSTM)
- Graph-Based Fraud Detection

to improve accuracy and detect complex fraud patterns.


## Features

 Multiple ML models (Random Forest, Logistic Regression, XGBoost)  
 Deep Learning models (ANN / LSTM)  
 Real-time fraud prediction system  
 Graph-based fraud detection using NetworkX  
 Community Detection for fraud group identification  
 Fraud Ring Detection (organized fraud networks)  
 Feature engineering for improved performance  
 Visualization (Confusion Matrix, ROC Curve, Graphs)  

---

##  Technologies Used

- **Language:** Python  
- **Libraries:**  
  - Scikit-learn  
  - XGBoost  
  - TensorFlow / Keras  
  - NetworkX  
  - Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  

---

##  Project Structure
fraud_detection_project/
│
├── data/
├── models/
├── outputs/
│ ├── graphs/
│ ├── reports/
│
├── src/
│ ├── preprocessing.py
│ ├── train_model.py
│ ├── evaluate.py
│ ├── realtime.py
│ ├── graph_fraud.py
│ ├── advanced_graph.py
│ ├── deep_learning.py
│
├── config.py
├── main.py
└── requirements.txt


Run Complete Pipeline
python main.py
 Run Graph-Based Detection Only
python -m src.advanced_graph
 Real-Time Fraud Detection
python src/realtime.py


| Model               | Accuracy | AUC Score |
| ------------------- | -------- | --------- |
| Random Forest       | ~99.7%   | ~0.9999   |
| Logistic Regression | ~93%     | ~0.98     |
| XGBoost             | ~99.8%   | ~0.9999   |
