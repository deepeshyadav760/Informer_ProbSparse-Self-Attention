# 🌍 Informer-Based Carbon Flux Forecasting using ProbSparse Self-Attention

An end-to-end deep learning pipeline for forecasting **Net Ecosystem Exchange (NEE)** (Carbon Flux) using the **Informer architecture**. The project utilizes multi-source satellite observations and meteorological variables from **Google Earth Engine (GEE)** to model long-term ecosystem carbon dynamics across the **Sundarbans Mangrove Forest** and **Chilika Lake**.

---

## 📌 Project Overview

Forecasting carbon flux is essential for understanding ecosystem health and the global carbon cycle. Traditional sequence models struggle with long temporal dependencies due to quadratic attention complexity.

This project employs the **Informer** architecture, a Transformer variant specifically designed for **Long Sequence Time-Series Forecasting (LSTF)**. By leveraging **ProbSparse Self-Attention**, Informer significantly reduces computational complexity while maintaining high forecasting performance.

---

## ✨ Features

- End-to-end carbon flux forecasting pipeline
- Google Earth Engine data acquisition
- Multi-source satellite data integration
- Informer (AAAI 2021) architecture
- ProbSparse Self-Attention
- Long sequence forecasting
- Streamlit visualization dashboard
- MLflow experiment tracking
- CSV-based dataset for easy analysis

---

# 📂 Dataset

The dataset is stored in **CSV format** for transparency and reproducibility.

| File | Description |
|------|-------------|
| `data/raw/carbon_flux_real.csv` | Original satellite observations |
| `data/raw/carbon_flux.csv` | Dashboard-compatible processed dataset |

---

## 🌎 Data Sources

The dataset is generated from **Google Earth Engine (GEE)** using multiple remote sensing products.

| Variable | Source | Description |
|----------|--------|-------------|
| Temperature | ERA5 Land Daily | Air Temperature (°C) |
| Soil Moisture | NASA SMAP L4 | Surface Soil Moisture (m³/m³) |
| NDVI | MODIS Terra | Vegetation Greenness Index |
| Radiation | ERA5 Land Daily | Surface Solar Radiation (W/m²) |
| VPD | ERA5 | Vapor Pressure Deficit |
| NEE | MODIS GPP | Net Ecosystem Exchange (Target Variable) |

**Temporal Coverage**

- 2018 – 2024
- Daily observations

---

## 🧠 Model Architecture

This project uses the **Informer** model proposed in:

> **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting**
>
> Zhou et al., AAAI 2021 (Best Paper)

Unlike standard Transformers with **O(L²)** attention complexity, Informer reduces computation to approximately **O(L log L)**, making it suitable for long environmental time series.

---

### Key Components

### 1. ProbSparse Self-Attention

- Selects only the most informative queries
- Reduces attention computation
- Complexity:
  - Transformer → **O(L²)**
  - Informer → **O(L log L)**

---

### 2. Self-Attention Distilling

- Reduces sequence length between encoder layers
- Lowers memory consumption
- Preserves dominant temporal features

---

### 3. Generative Decoder

Predicts the complete forecasting horizon in a single forward pass instead of autoregressive decoding.

Benefits:

- Faster inference
- Lower accumulated prediction error

---

### 4. Temporal Feature Encoding

The model learns seasonal patterns using:

- Day of year
- Month
- Sin/Cos positional encoding
- Global timestamps

---

# ⚙️ Pipeline

```
Google Earth Engine
        │
        ▼
Satellite Data Collection
        │
        ▼
CSV Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Sequence Generation
        │
        ▼
Informer Model
        │
        ▼
Training
        │
        ▼
Evaluation
        │
        ▼
Dashboard Visualization
```

---

## 📁 Project Structure

```text
Carbon_Flux_Forecasting_with_ProbSparse-Self-Attention/

├── dashboard/
├── data/
│   ├── raw/
│   └── processed/
├── model/
├── fetch_gee_data.py
├── train.py
├── evaluate.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🔄 Workflow

### 1. Data Collection

`fetch_gee_data.py`

- Connects to Google Earth Engine
- Downloads MODIS, SMAP and ERA5 products
- Performs spatial averaging
- Generates CSV datasets

---

### 2. Data Preprocessing

- Missing value interpolation
- Feature scaling
- Rolling statistics
- Sequence generation
- Train/Validation/Test split

---

### 3. Model Training

`train.py`

The Informer model is trained using **PyTorch** with:

- Early Stopping
- MAE Loss
- RMSE Evaluation
- MAPE Evaluation

Experiments are logged using **MLflow**.

---

### 4. Evaluation

`evaluate.py`

Performance metrics include:

- MAE
- RMSE
- MAPE

---

### 5. Dashboard

A Streamlit application enables interactive visualization of:

- Carbon Flux predictions
- Ground truth comparison
- Historical trends
- Dataset exploration

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/deepeshyadav760/Carbon_Flux_Forecasting_with_ProbSparse-Self-Attention.git

cd Carbon_Flux_Forecasting_with_ProbSparse-Self-Attention
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🌍 Authenticate Google Earth Engine

```bash
earthengine authenticate

earthengine set_project ee-deepeshy
```

---

## 📥 Fetch Dataset

```bash
python fetch_gee_data.py
```

---

## 🏋️ Train the Model

```bash
python main.py --data real
```

---

## 📊 Evaluate

```bash
python evaluate.py
```

---

## 💻 Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📈 Experiment Tracking

Experiments are tracked using **MLflow**.

Metrics logged include:

- Training Loss
- Validation Loss
- MAE
- RMSE
- MAPE

---

## 🛠 Technologies Used

- Python
- PyTorch
- Informer
- Google Earth Engine
- Streamlit
- MLflow
- NumPy
- Pandas
- Scikit-learn

---

## 📚 References

1. Zhou et al., **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting**, AAAI 2021.

2. Google Earth Engine

3. MODIS

4. NASA SMAP

5. ERA5 Land Dataset

---

## 📄 License

This project is intended for research and educational purposes.

---

## 👨‍💻 Author

**Deepesh Yadav**

AI Engineer | Machine Learning | Remote Sensing | Time-Series Forecasting
