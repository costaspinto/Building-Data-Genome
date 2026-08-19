# Building Energy Anomaly Detection using BDG2

## Executive Summary

This project implements a **production-style, end-to-end Machine Learning pipeline** to detect anomalous building energy behavior using the **Building Data Genome Project 2 (BDG2)** dataset.

The solution identifies unusual energy-consumption patterns such as spikes, drops, and irregular usage that may indicate equipment issues, scheduling problems, meter faults, or inefficient control strategies.

The pipeline is designed for **large-scale time-series data** and uses a memory-safe batch-processing workflow suitable for Google Colab and multi-meter building datasets.

### Key Deliverables

- Automated dataset download and validation
- Memory-safe wide-to-long preprocessing
- Batch Parquet storage
- Weather and building metadata enrichment
- Time-series feature engineering
- Multiple unsupervised anomaly detection models
- Ensemble anomaly voting
- Saved model artifacts
- Prediction outputs
- EDA and evaluation plots
- Business insights report

---

## 1. Business Context and Problem Statement

### Why Anomaly Detection Matters

Building energy consumption is influenced by:

- Occupancy patterns
- HVAC schedules and setpoints
- Seasonal weather variations
- Operational controls
- Maintenance conditions

Abnormal energy behavior can result in:

- Energy waste and increased utility costs
- Undetected equipment faults
- Degraded occupant comfort
- Higher maintenance costs
- Delayed operational intervention

### Problem Statement

Manual monitoring does not scale across:

- Multiple buildings
- Multiple energy meters
- Long time periods
- Large operational datasets

**Objective:** Detect anomalous energy readings from time-series meter data using an automated ML pipeline and provide business-ready insights for operational decision-making.

---

## 2. Dataset Overview

The project uses the **Building Data Genome Project 2 (BDG2)** dataset.

### Dataset Characteristics

- Hourly meter data covering **2016 and 2017**
- Multiple meters per building
- Building metadata
- Site-level weather observations

### Meter Data

The project works with:

- `electricity.csv`
- `chilledwater.csv`
- `steam.csv`
- `hotwater.csv`
- `gas.csv`
- `water.csv`
- `irrigation.csv`
- `solar.csv`

### Contextual Data

- `metadata.csv` — building and site information
- `weather.csv` — hourly weather observations

---

## 3. Project Objectives

### Technical Objectives

- Build a stable preprocessing pipeline for large time-series datasets
- Engineer meaningful anomaly-detection features
- Train multiple unsupervised models
- Combine model predictions through ensemble voting
- Save artifacts for reproducibility and deployment readiness

### Business Objectives

- Identify when and where abnormal energy patterns occur
- Prioritize buildings requiring investigation
- Identify peak-hour anomalies
- Understand weather-related consumption behavior
- Support continuous energy-monitoring workflows

---

## 4. Technology Stack

### Language

- Python

### Environment

- Google Colab

### Data Processing

- pandas
- NumPy

### Machine Learning

- scikit-learn

### Visualization

- Matplotlib
- Seaborn

### Model Persistence

- Joblib

### Large-Scale Data Processing

- PyArrow
- Parquet

---

## 5. Solution Architecture

The project is implemented as a structured notebook with separated data-engineering, analytics, modeling, and reporting stages.

```text
BDG2 Dataset
     │
     ▼
Automated Download
     │
     ▼
Data Validation
     │
     ▼
Wide Meter Data
     │
     ▼
Batch Wide → Long Conversion
     │
     ▼
Parquet Intermediate Storage
     │
     ▼
Weather + Building Metadata
     │
     ▼
Final Modeling Dataset
     │
     ├── EDA
     │
     ├── Feature Engineering
     │
     ▼
Robust Scaling
     │
     ├── Isolation Forest
     ├── Local Outlier Factor
     └── Robust Covariance
     │
     ▼
2-of-3 Ensemble Voting
     │
     ▼
Anomaly Predictions
     │
     ▼
Business Insights
```

---

## 6. Data Pipeline

### Stage A — Project Setup

The project uses a standardized structure:

```text
data/raw/                    # Original downloaded files
data/processed/              # Processed datasets
data/processed/long_batches/ # Parquet batch outputs
models/                      # Trained model artifacts
outputs/                     # Predictions and summaries
plots/                       # EDA and model plots
reports/                     # Business reports
logs/                        # Pipeline logs
```

### Stage B — Automated Dataset Download

The pipeline uses `kagglehub` to automate dataset retrieval.

The workflow:

1. Downloads the dataset into the Colab environment.
2. Copies the raw files into the project structure.
3. Validates the expected dataset files.

### Stage C — Data Validation

Before transformation, the pipeline checks:

- File availability
- File sizes
- Required columns
- `timestamp` availability
- Meter schemas
- Weather schema
- Metadata schema

This prevents downstream failures caused by missing or malformed inputs.

---

## 7. Memory-Safe Preprocessing

BDG2 meter files are stored in **wide format**, where a single timestamp can contain hundreds or thousands of building columns.

For machine learning, the data is transformed into long format:

```text
timestamp | building_id | meter_type | value
```

### Batch Processing Strategy

Instead of converting the entire dataset in memory, the pipeline:

1. Processes building columns in chunks.
2. Melts only a subset of columns.
3. Writes each processed batch to Parquet.
4. Reuses the stored batches for downstream processing.

This approach reduces memory pressure and makes the workflow more practical for Google Colab.

---

## 8. Weather and Metadata Enrichment

Each processed batch is enriched with building and environmental context.

### Building Metadata

The pipeline uses:

- `site_id`
- `primaryspaceusage`
- `sqm`
- `sqft`
- `timezone`

### Weather Variables

The pipeline incorporates:

- `airTemperature`
- `dewTemperature`
- `windSpeed`
- `windDirection`
- `cloudCoverage`
- `precipDepth1HR`
- `seaLvlPressure`

### Join Logic

Building-level metadata is linked through:

```text
building_id → metadata
```

Weather observations are linked through:

```text
site_id + timestamp → weather
```

---

## 9. Final Dataset Construction

The final modeling dataset is constructed using streaming writes.

The workflow:

1. Read a processed Parquet batch.
2. Sample rows to keep the final dataset manageable.
3. Append the batch to a single CSV.
4. Continue until all required batches are processed.

Final dataset:

```text
data/processed/final_preprocessed_dataset.csv
```

This approach maintains:

- Multi-meter coverage
- Stable dataset size
- Memory-efficient execution
- Reproducibility

---

## 10. Exploratory Data Analysis

EDA is performed on both raw and processed data.

### EDA Outputs

The analysis includes:

- Meter-type distribution
- Raw energy-value distributions
- Log-transformed energy distributions
- Missing-value analysis
- Building-level usage rankings
- Daily consumption trends
- Weather-consumption relationships

Generated visualizations are stored in:

```text
plots/
```

---

## 11. Feature Engineering

The feature-engineering pipeline is designed specifically for time-series anomaly detection.

### Rolling Statistics

Seven-day rolling statistics are calculated using a 168-hour window:

- Rolling mean
- Rolling standard deviation

### Deviation Score

A normalized deviation score is calculated as:

```text
deviation = (value - rolling_mean) / (rolling_std + ε)
```

This measures how far the current observation deviates from its recent historical behavior.

### Lag Features

The pipeline includes:

- `lag-1` — previous hour
- `lag-24` — same hour on the previous day

### Time Features

Temporal features include:

- Hour
- Day of week
- Month
- Weekend flag

---

## 12. Anomaly Detection Strategy

The project uses **unsupervised learning** because operational anomaly labels are typically unavailable.

### Models

Three anomaly-detection algorithms are implemented:

#### 1. Isolation Forest

A tree-based outlier-detection algorithm designed to isolate unusual observations efficiently.

#### 2. Local Outlier Factor

LOF identifies observations that have substantially different local density compared with their neighbors.

The project uses LOF in novelty mode.

#### 3. Robust Covariance

`EllipticEnvelope` provides robust covariance-based outlier detection.

### Scaling

The pipeline uses:

```text
RobustScaler
```

Robust scaling reduces sensitivity to extreme values, which is particularly relevant for energy-consumption data where spikes can be genuine anomalies.

---

## 13. Ensemble Voting

Instead of relying on a single anomaly detector, the pipeline combines the predictions of all three models.

Each model produces:

```text
Normal
or
Anomaly
```

The final decision uses a **2-of-3 voting rule**:

```text
Anomaly = 1
if at least 2 of 3 models classify the observation as anomalous.
```

This ensemble strategy is intended to reduce false positives and improve the reliability of anomaly identification.

---

## 14. Model and Output Artifacts

### Saved Models

Stored in:

```text
models/
```

Artifacts include:

```text
scaler.pkl
isolation_forest.pkl
lof_model.pkl
robust_cov.pkl
feature_list.pkl
```

### Prediction Output

Stored in:

```text
outputs/
```

Main output:

```text
anomaly_predictions.csv
```

### Visualizations

Stored in:

```text
plots/
```

Includes:

- EDA plots
- Model evaluation plots
- Anomaly distributions
- Time-series anomaly plots

### Business Report

Stored in:

```text
reports/
```

Main report:

```text
business_insights_summary.txt
```

---

## 15. Business Insights and Operational Value

The pipeline supports operational teams by enabling:

- Detection of abnormal consumption patterns
- Prioritization of high-risk buildings
- Identification of peak-hour anomalies
- Investigation of weather-driven consumption behavior

Potential operational outcomes include:

- Reduced energy waste
- Faster issue investigation
- Improved equipment reliability
- Better operational decision-making
- More targeted maintenance activity

---

## 16. Recommended Execution Order

Run the project in the following sequence:

1. Set up the project environment.
2. Install dependencies.
3. Download the BDG2 dataset.
4. Validate the raw files.
5. Convert wide meter data to long format in batches.
6. Store intermediate batches as Parquet.
7. Merge weather and building metadata.
8. Build the final processed dataset.
9. Run EDA.
10. Generate time-series features.
11. Scale the modeling features.
12. Train anomaly-detection models.
13. Apply ensemble voting.
14. Export anomaly predictions.
15. Generate plots and business insights.
16. Package the project artifacts.

---

## 17. Repository Structure

```text
Building-Data-Genome/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── long_batches/
│       └── final_preprocessed_dataset.csv
│
├── models/
│   ├── scaler.pkl
│   ├── isolation_forest.pkl
│   ├── lof_model.pkl
│   ├── robust_cov.pkl
│   └── feature_list.pkl
│
├── outputs/
│   └── anomaly_predictions.csv
│
├── plots/
│   └── *.png
│
├── reports/
│   └── business_insights_summary.txt
│
├── logs/
│   └── pipeline.log
│
├── config.json
├── Building_Data_Genome.ipynb
└── README.md
```

---

## 18. Interview Talking Points

This project demonstrates practical capability in:

- Large-scale time-series preprocessing
- Memory-aware ETL design
- Wide-to-long data transformation
- Batch processing with Parquet
- Weather and metadata integration
- Time-series feature engineering
- Unsupervised anomaly detection
- Ensemble model design
- Reproducible ML artifacts
- Operational analytics
- Business-oriented ML reporting

A strong interview explanation is:

> “I built an end-to-end unsupervised anomaly-detection pipeline for building energy data. The main engineering challenge was the size and wide structure of the BDG2 meter data, so I converted it to long format in batches and stored intermediate results in Parquet. I then enriched the data with weather and building metadata, engineered rolling, lag, deviation, and temporal features, and combined Isolation Forest, LOF, and Robust Covariance using a 2-of-3 voting strategy.”

---

## 19. Future Enhancements

Potential improvements include:

### Meter-Specific Models

Train specialized anomaly detectors for:

- Electricity
- Chilled water
- Steam
- Gas
- Solar
- Water

### Threshold Optimization

Introduce configurable anomaly sensitivity and automatic threshold tuning.

### Monitoring Dashboard

Build a Streamlit dashboard for:

- Building-level anomaly monitoring
- Meter-level filtering
- Time-series exploration
- Anomaly severity
- Operational alerts

### Model Monitoring

Add:

- Data drift detection
- Model drift monitoring
- Periodic retraining
- Performance tracking

### Alerting

Integrate alerting workflows such as:

- Email notifications
- Webhooks
- Maintenance-system integrations

---

## 20. License and Usage

This repository is intended for **educational and portfolio demonstration**.

Use of the BDG2 dataset should follow the licensing and usage terms provided by the original dataset source.

---

## Author

**Costas Pinto**

MCA — Artificial Intelligence & Machine Learning

GitHub:  
https://github.com/costaspinto

Repository:  
https://github.com/costaspinto/Building-Data-Genome
