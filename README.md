# Adaptive Irrigation AIoT System
## Overview

This project presents an Adaptive Irrigation Management System that integrates Internet of Things (IoT) and Artificial Intelligence (AI) technologies to optimize irrigation decisions in agriculture.

The system collects real-time environmental data from sensors and uses machine learning models to classify irrigation requirements into different levels, enabling efficient water management.

## Features

* Real-time monitoring of environmental parameters.
* MQTT-based communication between devices and applications.
* Machine Learning models for irrigation decision-making.
* Dynamic routing and model comparison.
* Interactive dashboard for visualization and monitoring.

## Project Structure

* `dashboard1.py`: Dashboard interface.
* `Drift_retrainer.py`: Drift detection and model retraining.
* `dynamic_router.py`: Dynamic routing mechanism.
* `Mqtt_bridge.py`: MQTT bridge service.
* `Mqtt_publisher1.py`: MQTT publisher.
* `Mqtt_subscriber1.py`: MQTT subscriber.
* `model_comparison.py`: Comparison of machine learning models.
* `data/`: Dataset files.
* `models/`: Trained models.
* `exported_charts/`: Exported visualizations.

## Technologies Used

* Python
* MQTT
* Machine Learning
* IoT
* Streamlit
* Pandas
* Scikit-learn

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Nourelhouda679/adaptive_irrigation_aiot.git
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the desired modules according to the application workflow.

## Dashboard Overview

![Dashboard](Screenshot%20from%202026-06-06%2012-53-59.png)

## Real-Time Analytics

![Real-Time Analytics](Screenshot%20from%202026-06-06%2012-55-15.png)

## Adaptive AI Features

The system continuously monitors confidence scores to detect concept drift and highlights the most influential features used by the XGBoost model for irrigation decision-making.

![Adaptive AI Features](Screenshot%20from%202026-06-06%2013-28-26.png)

## Author

Developed by Nourelhouda.
