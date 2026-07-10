# 🌸 Iris Flower Classification

A complete Machine Learning deployment project using **FastAPI**, **Streamlit**, and **Scikit-learn**.

This application allows users to enter Iris flower measurements and predicts the flower species using a trained Machine Learning model.

---

# 📌 Project Overview

This project demonstrates a complete Machine Learning deployment workflow.

Workflow:

User
→ Streamlit Frontend
→ FastAPI Backend
→ Machine Learning Model
→ Prediction Result

---

# 🚀 Technologies Used

- Python 3.11
- FastAPI
- Streamlit
- Scikit-learn
- Joblib
- Pytest
- Requests

---

# 📂 Project Structure

```text
iris-ml-deploym/
│
├── app_api/
│   ├── main.py
│   └── schemas.py
│
├── app_streamlit/
│   └── app.py
│
├── model/
│   └── iris_model.pkl
│
├── scripts/
│   └── train_model.py
│
├── tests/
│   └── test_api.py
│
├── screenshots/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone <your-repository-url>
```

Go to the project folder

```bash
cd iris-ml-deploym
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Train the Model

```bash
python scripts/train_model.py
```

The trained model will be saved as:

```text
model/iris_model.pkl
```

---

# ▶️ Run FastAPI

```bash
python -m uvicorn app_api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

Swagger UI will open automatically.

---

# 💻 Run Streamlit

Open another terminal

```bash
python -m streamlit run app_streamlit/app.py
```

Open:

```
http://localhost:8501
```

---

# 📬 API Endpoint

POST

```
/predict
```

Example Request

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example Response

```json
{
  "prediction_id": 0,
  "prediction": "setosa",
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

---

# ✅ Testing

Run all tests

```bash
python -m pytest -v
```

Expected result

```
4 passed
```

---

# 📸 Screenshots

Project Structure

```
screenshots/01-project-structure.png
```

Model Training

```
screenshots/02-model-training.png
```

FastAPI Running

```
screenshots/03-fastapi-running.png
```

Swagger Documentation

```
screenshots/04-swagger-docs.png
```

API Prediction

```
screenshots/05-api-prediction.png
```

Streamlit Application

```
screenshots/06-streamlit-prediction.png
```

API Tests

```
screenshots/07-api-tests.png
```

---

# 👨‍💻 Author

Reza Akbari

Vanier College

AI Model Deployment Course
