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


