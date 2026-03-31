# 🏃 Walk vs Run Classification

A Machine Learning project that classifies human physical activity (Walking 🚶 or Running 🏃) using wearable sensor data (accelerometer + gyroscope). A **Streamlit web app** is included for real-time prediction.

---

## 📌 Problem Statement

Using motion sensor data collected from wearable devices, predict whether a person is **walking** or **running**. This is a binary classification problem using accelerometer and gyroscope readings.

---
## 📊 Dataset

- **Source:** Wearable device sensor recordings
- **Total Records:** ~88,588 rows
- **Features:** 6 sensor features + metadata

| Column            | Description                            |
|-------------------|----------------------------------------|
| `acceleration_x`  | Acceleration along X-axis              |
| `acceleration_y`  | Acceleration along Y-axis              |
| `acceleration_z`  | Acceleration along Z-axis              |
| `gyro_x`          | Gyroscope rotation along X-axis        |
| `gyro_y`          | Gyroscope rotation along Y-axis        |
| `gyro_z`          | Gyroscope rotation along Z-axis        |
| `activity`        | Target: `0` = Walk, `1` = Run          |
| `wrist`           | Wrist worn: `0` = Left, `1` = Right    |

---

## 🔬 ML Pipeline (Notebook)

### 1. Import Libraries
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `tensorflow/keras`, `joblib`

### 2. Load & Explore Data
- Shape: 88,588 rows × 11 columns
- No missing values
- Balanced activity classes (~50% walk, ~50% run)

### 3. EDA (Exploratory Data Analysis)
- **Univariate:** Activity distribution, Acceleration X histogram, Gyro Z histogram
- **Bivariate:** Boxplots of Activity vs Acceleration X, Gyro Y
- **Multivariate:** Pairplot of sensor features, Correlation heatmap
- **Time Analysis:** Activity patterns by hour of day and day of week

### 4. Data Preprocessing
- Dropped: `date`, `time`, `username`, `wrist`
- Label Encoded: `activity` column (walk=0, run=1)
- Dropped rows with NaN values

### 5. Train-Test Split
- **80%** training / **20%** testing
- `random_state=42` for reproducibility

### 6. Models Trained & Evaluated

| Model               | Notes                            |
|---------------------|----------------------------------|
| Logistic Regression | Baseline linear model            |
| Decision Tree       | Simple tree-based model          |
| Random Forest ✅    | Best performer — saved to `.pkl` |
| SVM (Linear Kernel) | Support Vector Machine           |
| RNN (Dense/LSTM)    | Neural network approach          |
| CNN (1D Dense)      | Convolutional approach           |

### 7. Hyperparameter Tuning
- `GridSearchCV` on Random Forest
- Parameters tuned: `n_estimators`, `max_depth`
- Best model saved using `joblib`

---

## ✅ Best Model: Random Forest

- **Hyperparameters:** `n_estimators=200`, `max_depth=None`
- **Accuracy:** ~99%
- **Saved as:** `model.pkl`

---

## 🖥️ Streamlit Web App

The `app.py` provides an interactive UI to enter sensor values and get real-time predictions.

### Features
- Sidebar input for 6 sensor values
- Predicts **Walk** or **Run**
- Shows **confidence percentages** for both classes

### How to Run

**Step 1: Clone the Repository**
```bash
git clone https://github.com/your-username/walk-run-classification.git
cd walk-run-classification
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Run the App**
```bash
streamlit run app.py
```

**Step 4: Open in Browser**
```
http://localhost:8501
```

---

## 📦 Requirements

Create a `requirements.txt` with:

```
streamlit
numpy
scikit-learn
joblib
pandas
matplotlib
seaborn
tensorflow
```

Install all at once:
```bash
pip install streamlit numpy scikit-learn joblib pandas matplotlib seaborn tensorflow
```

---

## 🚀 Steps to Deploy on Streamlit Cloud (Free)

1. Push this project to a **GitHub repository**
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub
3. Click **"New app"** → Select your repo → Set main file as `app.py`
4. Click **Deploy** 🎉

> ⚠️ Make sure `model.pkl` is included in the repo (file size < 100MB)

---

## 📸 App Preview

| Input | Prediction |
|-------|-----------|
| Enter acceleration & gyroscope values in sidebar | App predicts Walk 🚶 or Run 🏃 with confidence % |

---

## 🧠 Key Learnings

- Sensor data from wearables is highly effective for activity recognition
- Random Forest significantly outperforms simpler models on this dataset
- Acceleration features have clear separation between walk and run patterns
- Gyroscope captures rotational movement useful for classification

---
