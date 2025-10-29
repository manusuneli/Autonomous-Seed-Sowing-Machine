# 🌾 KASHYAP – Autonomous Seed Sowing Robot

An **AI-driven, 5G-enabled agricultural robot** that automates seed sowing with precision, intelligence, and real-time connectivity.  
Kashyap integrates **robotics**, **machine learning**, and **IoT sensors** to revolutionize seed placement, optimize sowing depth, and predict crop yield.

---

## 🚀 Overview
Kashyap is an **autonomous farming rover** that:
- Performs **precision seed sowing** using sensor feedback and AI.
- Provides **real-time monitoring** via a 5G-connected mobile app.
- Uses a **LightGBM machine learning model** to predict:
  - Crop yield (kg/hectare)
  - Optimum sowing depth based on soil and seed type.
- Uploads real-time data to the **cloud** for visualization and analytics.

---

## 🧠 Machine Learning Component

### 🎯 Objective
Use real-time agricultural and environmental data to:
1. **Predict expected crop yield**
2. **Determine optimal sowing depth**

These predictions are generated using **LightGBM regression models** trained on real crop and soil datasets.

---

### 📥 Input Features
| Parameter | Description | Type |
|------------|-------------|------|
| `Seed_Type` | Type of crop seed (e.g., Wheat, Maize, Rice) | Categorical |
| `Soil_Type` | Classified from soil image using AI | Categorical |
| `Soil_Moisture` | Volumetric water content in soil (%) | Numerical |
| `Sowing_Depth` | Depth at which seed is placed (cm) | Numerical |
| `Seed_Spacing` | Distance between consecutive seeds (cm) | Numerical |

---

### ⚙️ Machine Learning Pipeline

1. **Data Collection**
   - Real-time data from sensors (moisture, spacing, depth)

2. **Data Preprocessing**
   - Encoding categorical data (e.g., seed and soil type)
   - Normalizing numeric features for better model stability
   - Handling missing data with median imputation

3. **Model Training**
   - Algorithm: **LightGBM Regressor**
   - Objective: `regression`
   - Metric: `rmse`
   - Trained on: `indian_crop_data_realistic_v2.csv`
   - Saved as: `yield_predictor_final_v2.txt`

4. **Model Evaluation Metrics**
   - **R² (Coefficient of Determination):** Measures variance explained by the model  
   - **MAE (Mean Absolute Error):** Average magnitude of prediction errors  
   - **RMSE (Root Mean Squared Error):** Penalizes large prediction errors more heavily  

---

### 📊 Results

| Metric | Description | Typical Value |
|---------|--------------|----------------|
| **R²** | Variance explained by model | ~0.93 |
| **MAE** | Average absolute error | 120–150 kg/hectare |
| **RMSE** | Root mean squared error | ~180 kg/hectare |

**Visualization Outputs**
- 🟢 `predicted_vs_actual.png` – Scatter plot of predicted vs actual yields  
- 🔵 `binned_confusion_matrix.png` – Yield quartile classification accuracy  
- 🟠 `model_metrics.png` – Summary of R², MAE, and RMSE values  

---

### 🌱 Optimum Depth Prediction
A separate LightGBM regression model predicts **optimal seed sowing depth** based on:
- Soil moisture  
- Soil type  
- Seed type  
- Local environmental parameters  

This predicted depth is sent to the **servo-controlled plough mechanism**, which automatically adjusts the sowing depth for maximum germination success.

---

### 🧩 Integration Workflow
1. The **Raspberry Pi 5** collects real-time sensor data.  
2. The **trained LightGBM model** (`yield_predictor_final_v2.txt`) runs inference on the device.  
3. Predictions are used for:
   - **Servo motor control** (depth adjustment)
   - **Cloud analytics dashboard** (yield forecasting)
4. Results are transmitted through **5G** and logged to **Firebase Cloud**.

---

## 📡 5G and IoT Connectivity
- High-speed **real-time video streaming** and telemetry.
- Two-way communication between rover and mobile dashboard.
- Remote control: start/stop, adjust depth, reroute rover.
- Cloud logging for analytics and replay visualization.

---

## 🧰 Tech Stack
| Layer | Technology |
|-------|-------------|
| **Core Hardware** | Raspberry Pi 5 |
| **Model Framework** | LightGBM, scikit-learn, pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Database / Cloud** | Firebase |
| **Communication** | Quectel RM500Q 5G Module |
| **Control System** | Arduino UNO + Servo Motors |

---


## 📁 Repository Structure

```
Autonomous-Seed-Sowing-Machine/
│
├── predictor/                  # Inference code and saved models
│   ├── yield_predictor_final_v2.txt
│   ├── depth_predictor_model.txt
│   └── predictor_utils.py
│
├── train.ipynb                 # LightGBM model training
├── yield_train.ipynb           # Yield prediction training
├── matrix.ipynb                # EDA and correlation analysis
├── indian_crop_data_realistic_v2.csv
└── README.md                   # Project documentation
```

---


## 🧑‍🌾 Real-World Benefits
- 🔹 Optimizes sowing depth automatically for better germination  
- 🔹 Predicts expected yield in real time  
- 🔹 Reduces seed wastage by 25–30%  
- 🔹 Increases overall field productivity and efficiency  
- 🔹 Scalable across various crops and soil types  

---

## 💡 Future Enhancements
- Integrate **Deep Learning (CNN)** for improved soil classification  
- Real-time **adaptive retraining** using continuous data streams  
- Extend dataset for region-specific crop models  
- Optimize model for **Edge AI deployment** on Raspberry Pi  

---


