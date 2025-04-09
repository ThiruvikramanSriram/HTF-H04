# Medsetra - Proactive Healthcare with SDoH Analytics

Medsetra is a web-based application designed to provide proactive healthcare insights through Social Determinants of Health (SDoH) analytics. The application features a dashboard that visualizes healthcare demand forecasts, risk maps, community health profiles, and intervention tracking.

## Frontend

## Features

* **Dashboard**: Overview of key healthcare metrics including high-risk patients, predicted admissions, and projected savings.
* **Risk Maps**: Interactive maps displaying health risk distributions by county, with tooltips providing detailed information on selected areas.
* **Forecasts**: Predictive analytics for hospital admissions and chronic disease incidence.
* **Community Profiles**: Detailed health profiles for various communities, highlighting risk factors and interventions.
* **Interventions Tracking**: Monitor active health interventions and their impacts on community health.

## Technologies Used

* HTML, CSS, JavaScript
* D3.js for data visualization
* Chart.js for charting
* TopoJSON for geographic data representation
* Font Awesome for icons

## Installation

To run the application locally, follow these steps:

1. Clone our repository 
2. Open the `index.html` file in your web browser.
3. Ensure you have an internet connection to load external libraries (D3.js, Chart.js, etc.) from CDN.

## Usage

* Navigate through the sidebar to access different sections of the application.
* Use the search bar to find specific data or metrics.
* Interact with the heat maps to view detailed information about health risks in various counties.
* Click on the community cards to view detailed health profiles and interventions.

## Data Sources

The application uses dummy data for demonstration purposes. The following datasets are included:

* **County Data**: Simulated data for various counties, including risk scores and SDoH factors.
* **Monthly Forecast Data**: Simulated data for emergency room visits and hospital admissions.
* **Service Type Data**: Simulated data for different healthcare service types.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

* [D3.js](https://d3js.org/) for data visualization.
* [Chart.js](https://www.chartjs.org/) for charting capabilities.
* [TopoJSON](https://github.com/topojson/topojson) for geographic data representation.
* [Font Awesome](https://fontawesome.com/) for icons.

## Backend Model

## Overview

The goal of this project is to forecast healthcare demand using real-time SDoH data. The ideal model combination should:

- Handle seasonality and trends.
- Capture non-linear relationships between socio-economic factors and healthcare demand.
- Integrate temporal dependencies to account for both short-term and long-term changes.
- Provide interpretable results, especially when dealing with sensitive healthcare data.

## Model Combination

### 1. Primary Model: SARIMA + XGBoost

#### SARIMA (Seasonal Autoregressive Integrated Moving Average)
- **Purpose**: Captures seasonality and trends in healthcare demand data.
- **Strengths**:
  - Excellent for short-term forecasting.
  - Handles strong seasonal components (e.g., flu season spikes in hospital admissions).
  - Provides a baseline prediction for healthcare demand.

#### XGBoost (Extreme Gradient Boosting)
- **Purpose**: Models non-linear relationships between socio-economic variables (SDoH) and healthcare demand.
- **Strengths**:
  - Handles complex interactions and feature importance.
  - Adjusts the baseline forecast provided by SARIMA by considering socio-economic and external factors like unemployment and eviction rates.

### 2. Add-on Models for Enhancement

#### LSTM (Long Short-Term Memory)
- **Purpose**: Captures long-term dependencies in time-series data.
- **Strengths**:
  - Useful for complex temporal relationships (e.g., effects of economic downturns on demand over several months).
  - Learns from historical data and socio-economic indicators over time.

#### Random Forests
- **Purpose**: Improves upon decision trees by averaging over multiple trees to reduce overfitting.
- **Strengths**:
  - Performs well with a mix of categorical and continuous data.
  - Models complex interactions in socio-economic factors influencing healthcare demand.

#### NARNET (Nonlinear Autoregressive Neural Network)
- **Purpose**: Specifically designed for forecasting time-series data with non-linear dependencies.
- **Strengths**:
  - Captures non-linear dependencies in both historical demand and external factors (SDoH).
  - Good alternative to SARIMA for non-linear trends.

#### Vector Autoregression (VAR)
- **Purpose**: Models multiple interdependent time-series data.
- **Strengths**:
  - Captures relationships between multiple healthcare indicators or socio-economic variables affecting healthcare demand.
  - Useful for forecasting based on past values of all related variables.

## Evaluation Metrics

To evaluate the performance of these models, the following metrics are used:

### For SARIMA/XGBoost (and Random Forests, if used):
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors.
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more than MAE.
- **Mean Absolute Percentage Error (MAPE)**: Good for percentage-based errors in forecasting.
- **R-squared**: Measures how well the model explains the variance in the data.

### For LSTM and NARNET:
- **RMSE**: Evaluates the accuracy of the model’s predictions.
- **Precision, Recall, and F1-score**: Useful for classification tasks like identifying at-risk communities.

### For VAR:
- **AIC (Akaike Information Criterion)** and **BIC (Bayesian Information Criterion)**: For model selection and comparing different models based on their complexity.

## 📦 Dataset Summary
## 1️⃣ Emergency Healthcare Utilization – India
File: Emergency_Healthcare_Utilization_India.xlsx

Format: Excel Workbook

Overview: Emergency healthcare statistics in India, likely categorized by:

Region

Type of emergency

Patient demographics (age, gender)

Healthcare facility type (public/private)

Treatment outcomes

## 2️⃣ Healthcare Metadata – USA (2018)
File: metadata_usa_2018.csv

Format: CSV

Overview: 2018 metadata related to U.S. states, potentially includes:

Population

GDP per capita

Healthcare expenditure

Hospital infrastructure

Life expectancy, mortality rate

## 3️⃣ Social Determinants of Health (SDoH) – USA
File: SDoH_data (1).csv

Format: CSV

Overview: SDoH indicators across U.S. regions such as:

Education

Income level

Housing quality

Unemployment rates

Food access

Health insurance coverage

Source: https://www.ahrq.gov/sdoh/data-analytics/sdoh-data.html


BACKEND:

backend/
├── app.py              # Main Flask application file
└── templates/
    └── index.html      # Web interface template
Overview
This Flask application serves as the backend for an Emergency Healthcare Utilization Forecasting System. It processes data and generates predictions for healthcare metrics across different regions in India, with a web interface available through the index.html template.
Purpose
The system helps healthcare administrators, policy makers, and emergency response teams plan resources efficiently by providing forecasts of critical healthcare utilization metrics. By anticipating future demand, healthcare systems can optimize staffing, bed allocation, ambulance deployment, and medical supplies management.
Key Features
Multi-Region Support

Provides forecasts for different regions across India (North, South, East, West, Central, and Northeast)
Accommodates regional variations in healthcare utilization patterns

Comprehensive Metrics
Forecasts multiple critical healthcare metrics:

Emergency Cases
Hospital Bed Utilization
ICU Utilization
Ambulance Utilization
Emergency Staff Utilization
Medical Supplies Utilization

Advanced Forecasting Methods
Employs a hybrid ensemble approach combining:

SARIMA (Seasonal AutoRegressive Integrated Moving Average)
Random Forest
XGBoost
VAR (Vector AutoRegression)
LSTM (Long Short-Term Memory neural networks)

Realistic Predictions

Incorporates realistic fluctuations to simulate real-world variability
Adds appropriate noise levels, weekly patterns, and occasional spikes/dips based on metric type
Ensures predictions stay within realistic bounds

Web Interface

User-friendly web interface (index.html)
Selection of regions for prediction
Visualization of forecast results

Technical Details

Built with Flask framework
Implements multiple machine learning models from scikit-learn, statsmodels, XGBoost, and TensorFlow
Includes health check endpoint for monitoring
Handles exceptions gracefully with informative error messages

Installation and Usage

Navigate to the backend directory
Install required dependencies:
pip install flask pandas numpy scikit-learn xgboost statsmodels tensorflow

Run the application:
python app.py

Access the web interface at http://localhost:5000
Select a region and generate predictions via the interface

API Endpoints

/ - Main web interface
/predict - POST endpoint for generating predictions
/health - Health check endpoint to verify server status

This application serves as a valuable tool for emergency healthcare planning and resource allocation, helping to improve healthcare service delivery and optimize resource utilization across different regions of India.

# Backend logic

## 📌 Overview
This system implements a sophisticated healthcare demand forecasting solution that leverages multiple machine learning models and real-time Social Determinants of Health (SDoH) data to predict healthcare utilization across different metrics and regions.

## 🎯 Purpose
The Healthcare Demand Forecasting System is designed to help healthcare administrators, policy makers, and emergency response teams anticipate future healthcare needs more accurately. By incorporating SDoH data alongside traditional time series analysis, the system provides context-aware predictions that reflect the socioeconomic conditions influencing healthcare demand.

## 📚 Intended Use
This system is designed for **educational and research purposes**, showcasing a modern approach to healthcare forecasting. It integrates multiple ML techniques and contextual data sources but is **not intended for production deployment** without further validation.

---

## 🧠 Key Components

### 1. 🔮 Multi-Model Ensemble Approach
This hybrid methodology combines the strengths of multiple forecasting techniques:
- **SARIMA** – Captures seasonality, trends, and temporal patterns
- **XGBoost** – Models non-linear relationships with high accuracy
- **Random Forest** – Handles categorical and continuous data interactions
- **LSTM** – Deep learning model for complex temporal dependencies
- **Vector Autoregression (VAR)** – Models interdependence between multiple time series
- **Prophet** – Deals robustly with holidays, missing data, and trend shifts
- **Hybrid Model** – Combines outputs using adaptive weighting strategies

### 2. 📊 SDoH Data Integration
Uses real-time Social Determinants of Health (SDoH) such as:
- Unemployment rate
- Education levels
- Housing instability
- Income inequality
- Access to community resources

### 3. ⚙️ Configuration System
Fully customizable via a `config.yaml` file:
- Model selection and parameter tuning
- Data splitting and preprocessing settings
- Forecast horizon control
- Weighting strategies for ensemble models

### 4. 🔧 Advanced Features
- **Configurable Training Parameters**
- **Adaptive Model Weighting**
- **Time Series Cross-Validation**
- **Forecast Anomaly Detection**
- **Comprehensive Evaluation Metrics**



