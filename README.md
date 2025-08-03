# Student Dropout Prediction Using Image Data

## Project Overview
This project builds a machine learning pipeline to predict student dropout risk based on image data (e.g., facial expressions, classroom engagement). It extends a previous tabular-data summative assignment by incorporating non-tabular, visual data for early dropout detection.

The goal is to provide educational institutions with a tool to identify at-risk students early by analyzing image-based behavioral cues.


## Problem Statement
Predict which students are likely to drop out by analyzing classroom images (students’ facial expressions that determine whether a student is engaged in class or not) using deep learning models. Early identification enables targeted intervention to improve student retention.



## Dataset
- **Source:** [Kaggle - Student Engagement Dataset](https://www.kaggle.com/datasets/joyee19/studentengagement?resource=download)  
- **Format:** Images (`.jpg`, `.png`)  
- **Classes:**  
  - Dropout  
  - Non-Dropout  



## Features & Workflow
- Image loading and preprocessing  
- Mobile training with regularization and optimization  
- Model evaluation with multiple metrics  
- Inference script for predicting new images  
- Flask API serving prediction and retraining endpoints  
- Streamlit dashboard for visualization, model uptime, and retraining interface  


## Technologies Used
- Python 3.8+  
- TensorFlow / Keras  
- scikit-learn  
- Pandas, NumPy  
- OpenCV  
- Matplotlib, Seaborn  
- Flask (API)  
- Streamlit (UI)  
- Locust (Load testing)  
- Jupyter Notebook  


## Machine Learning Models
- Convolutional Neural Network (Mobile4Dev) — best-performing model  
- Evaluation metrics used: Accuracy, Precision, Recall, F1-score, Loss  



## Results
- **Best Model:** Transfer Learning model Mobile4
- **Accuracy:** 0.97  
- **Key Insights:**  
  - Students with disengaged facial expressions had a higher dropout risk  
  - Data augmentation improved model generalization  



## How to Run

### Clone the repository  
```bash
git clone https://github.com/tamandakaunda-15/student-dropout-prediction.git
cd student-dropout-prediction
```

### Create a virtual environment
``` python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

### Install dependencies
```
pip install -r requirements.txt
Run Jupyter notebook
jupyter notebook notebook/student_dropout_prediction.ipynb
```

### Run the Flask API

``` cd api
gunicorn --bind 0.0.0.0:10000 app:app
Run the Streamlit dashboard
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### Flood Request Simulation

Load testing was performed using Locust to simulate flood requests to the prediction API endpoint.

Latency and failure rates were recorded and analyzed to evaluate API robustness under load.

Locust script: simulation/flood_test.py

Simulation results: [link to screenshot or folder]

### Project Structure

```student-dropout-prediction/
│
├── README.md
├── notebook/
│   └── student_dropout_prediction.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
├── data/
│   ├── train/
│   └── test/
├── models/
│   └── student_engagement_model.h5
├── app.py #this is the api script
│   
├── dashboard.py   #this is the user interface dashboard that contains all visualizations
│  
└── simulation/
    └── flood_test.py
```

## Contributors
Tamanda Kaunda — Data Science, Machine Learning, API, and UI development

## Demo Video
https://youtu.be/OoCAa_thkDA

## Deployment URL

https://summative-assignment-mlops.onrender.com/ #This is the Flask API deployment link

The UI had alot of bugs and internal errors to internal errors but here's the url ; https://tamandakaunda-15-sum-student-dropout-predictiondashboard-gg8a44.streamlit.app/










