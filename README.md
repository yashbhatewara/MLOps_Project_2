# 🏙️ Abu Dhabi Rent Prediction MLOps Project

## 📌 Overview

This project demonstrates a production-grade end-to-end MLOps pipeline for predicting rental prices in Abu Dhabi using real-world property data.

It reflects how modern machine learning systems are:
- Designed
- Trained
- Validated
- Deployed

Due to GitHub file size limitations, the dataset is stored in MongoDB Atlas, simulating real-world production pipelines.

---

## 🚀 Key Highlights

- Modular and scalable architecture  
- End-to-end ML lifecycle implementation  
- MongoDB-based data ingestion  
- Production-level logging and exception handling  
- Achieved **89.6% R² Score** (RMSE: AED 2,090)  
- Processed **69,059 property listings**  
- CI/CD ready using Docker and GitHub Actions  
- Deployed prediction service with web interface  

---

## 🧱 Project Architecture

```
MongoDB Atlas
↓
Data Ingestion
↓
Data Validation
↓
Data Transformation
↓
Model Training
↓
Model Evaluation
↓
Prediction Pipeline
↓
Web Application (Flask)
```

---

## 📁 Project Structure

- `components/` → Core ML pipeline components  
- `config/` → Configuration files (schema, constants)  
- `pipeline/` → Training & prediction pipelines  
- `utils/` → Utility functions  
- `artifacts/` → Generated outputs (models, logs)  

---

## ⚙️ Environment Setup

### Step 1: Create Virtual Environment

```bash
conda create -n rent python=3.10 -y
conda activate rent
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
pip list
```

---

## 📊 MongoDB Setup

### Step 4: Configure MongoDB Atlas

- Create a MongoDB Atlas account  
- Set up a free M0 cluster  
- Allow access from `0.0.0.0/0`  
- Create a database user  
- Get your connection string  

### Step 5: Set Environment Variable

#### Linux / Mac

```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@..."
```

#### Windows PowerShell

```powershell
$env:MONGODB_URL="mongodb+srv://<username>:<password>@..."
```

---

## 📝 Logging & Exception Handling

- Centralized logging system implemented  
- Custom exception handling for robustness  

---

## 🔍 EDA & Feature Engineering

- Performed Exploratory Data Analysis (EDA)  
- Created domain-specific features:

```python
sqft_per_bed = Area_in_sqft / max(Beds, 1)
total_rooms = Beds + Baths
```

---

## 📥 Data Ingestion Pipeline

- Data fetched from MongoDB using PyMongo  
- Converted into Pandas DataFrame  
- Artifact tracking implemented  

---

## ✅ Data Validation

- Schema defined in `config/schema.yaml`  
- Ensures data consistency and correctness  

---

## 🔄 Data Transformation

- Feature engineering pipeline implemented  
- Custom transformations applied  

---

## 🤖 Model Training

- Multiple models trained and evaluated  
- Best model selected based on performance  

---

## 📈 Model Evaluation

**Performance Metrics:**

- R² Score: **89.6%**  
- RMSE: **AED 2,090**  

---

## 🔮 Prediction Pipeline

- Takes user input  
- Applies transformation  
- Outputs predicted rent  

---

## 🌐 Web Application

- Built using Flask (`app.py`)  
- UI supported via `static/` and `templates/`  

### Run the Application

```bash
python app.py
```

### Access in Browser

```
http://localhost:8000
```

---

## 🔄 CI/CD Automation

- Dockerized application using `Dockerfile`  
- GitHub Actions for CI/CD pipeline  

---

## 🛠️ Tech Stack

- Python  
- MongoDB Atlas  
- PyMongo  
- Scikit-learn  
- Flask  
- MLflow  
- Docker  
- GitHub Actions  

---

## 🎯 Workflow Summary

- Data ingestion from MongoDB  
- Schema-based validation  
- Feature engineering  
- Model training and evaluation  
- Deployment via Flask  

---

## 💬 Connect

If you want to discuss:
- MLOps  
- ML Engineering  
- Deployment strategies  

Feel free to connect!

---

## ⭐ Support

If this project helped you, consider giving it a **star ⭐**