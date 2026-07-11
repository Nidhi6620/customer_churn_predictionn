# Customer Churn Prediction using Machine Learning, MLflow & FastAPI

## 📌 Overview

This project predicts whether a customer is likely to **churn (leave a service)** using a **Random Forest Classifier**. It demonstrates the complete machine learning workflow, including data preprocessing, model training, experiment tracking with MLflow, and deployment using FastAPI.

---

## 🚀 Features

- Data preprocessing and cleaning
- Categorical feature encoding
- Random Forest Classifier
- Model evaluation using Accuracy, Precision, Recall, and F1-Score
- MLflow experiment tracking
- Model serialization using Joblib
- FastAPI REST API for predictions
- Docker support for deployment

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- FastAPI
- Joblib
- Uvicorn
- Docker

---

## 📂 Project Structure

```
Customer-Churn-Prediction/
│
├── app.py
├── train.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── data/
│   └── churn.csv
│
├── models/
│   └── churn_model.joblib
│
├── mlruns/
├── mlflow.db
└── .gitignore
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/YashavanthA07/model-training-customer.git
cd model-training-customer
```

### Create a Virtual Environment

```bash
python -m venv env
```

### Activate the Virtual Environment

**Windows**

```bash
env\Scripts\activate
```

**Linux / macOS**

```bash
source env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

Place the dataset inside the `data` folder.

```
data/churn.csv
```

---

## 🧠 Train the Model

```bash
python train.py
```

The script will:

- Load the dataset
- Preprocess the data
- Encode categorical features
- Train a Random Forest model
- Evaluate model performance
- Save the trained model
- Log experiments using MLflow

---

## 📈 MLflow

Start the MLflow UI:

```bash
mlflow ui
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

You can view:

- Experiments
- Parameters
- Metrics
- Models
- Artifacts

---

## 🌐 Run the FastAPI Server

```bash
uvicorn app:app --reload
```

API URL:

```
http://127.0.0.1:8000
```

Swagger Documentation:

```
http://127.0.0.1:8000/docs
```

---

## 📥 Prediction API

### Endpoint

```
POST /predict
```

### Sample Request

```json
{
  "CustomerID": 101,
  "Gender": 1,
  "Age": 35,
  "Tenure": 24,
  "MonthlyCharges": 75.5,
  "TotalCharges": 1800,
  "ContractType": 1,
  "InternetService": 2,
  "PaymentMethod": 0
}
```

### Sample Response

```json
{
  "prediction": "Customer Will Stay"
}
```

or

```json
{
  "prediction": "Customer Will Churn"
}
```

---

## 🐳 Docker

### Build Docker Image

```bash
docker build -t customer-churn-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 customer-churn-api
```

---

## 📊 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

All metrics are automatically logged in MLflow.

---

## 🔮 Future Enhancements

- Hyperparameter tuning
- MLflow Model Registry
- XGBoost and LightGBM models
- CI/CD using GitHub Actions
- Cloud deployment (AWS/Azure)
- Streamlit dashboard
- User authentication

---

## 👨‍💻 Author

**Vidyanidhi G Shetty**

B.E. Computer Science (Data Science)  
Alva's Institute of Engineering and Technology


---

## ⭐ Support

If you found this project useful, please consider:

- ⭐ Starring the repository
- 🍴 Forking the project
- 🤝 Contributing through pull requests
