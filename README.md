#  Student Dropout Prediction Using Image Data

This project builds an ML pipeline to predict student dropout using non-tabular data (images), as a continuation of the student dropout prediction summative assignmetn previously done using tabular data.


##  Project Structure

```student-dropout-prediction/
├── README.md # Project overview and documentation
├── notebook/
│ └── student_dropout_prediction.ipynb # EDA, model training and evaluation
├── src/
│ ├── preprocessing.py # Data cleaning and preparation
│ ├── model.py # Model training and evaluation
│ └── prediction.py # Predicting dropout on new data
├── data/
│ ├── train/ # Training dataset(s)
│ └── test/ # Testing dataset(s)
└── models/
└── model_name.pkl # Saved trained model(s)
```


 ## Problem Statement
The goal is to predict which students are at risk of dropping out based on visual cues (e.g., facial expressions, classroom engagement) extracted from image data. The aim is to build a model that can detect early warning signs using image-based machine learning.


## Dataset
Source: [insert dataset name or link here]
Format: Images (.jpg/.png)
Classes: Dropout vs Non-Dropout
Additional metadata:

## Features & Workflow
Image loading & preprocessing
CNN model training
Model evaluation
Inference script
Streamlit frontend (optional extension)

##  Technologies Used

- **Python 3.8+**
- **scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib / Seaborn**
- **Jupyter Notebook**

---

##  Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- Neural Networks

Each model is evaluated using standard metrics like **accuracy**, **precision**, **recall**, and **F1-score**.

---

##  Results

- Best model: **[Insert best-performing model here]**
- Accuracy: **[xx]%**
- Key insights:
  - [Add 1–2 insights, e.g., students with poor attendance were 3x more likely to drop out.]

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://tamandakaunda-15/student-dropout-prediction.git
   cd student-dropout-prediction
   
(Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies:

pip install -r requirements.txt 

```

Run the Jupyter notebook:
jupyter notebook notebook/student_dropout_prediction.ipynb

## Contributors
Tamanda Kaunda — Data Science & ML Implementation

