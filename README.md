# FORESA.AI
Asset Damage Prediction System
This project is a Generative AI-driven system designed to predict potential hazards to physical assets due to environmental anomalies (e.g., natural disasters, extreme weather conditions). The system leverages historical data, real-time weather API data, and advanced machine learning models to forecast the risk and severity of asset damage. Developed by Team 6 at Sri Ramakrishna Engineering College, this solution aims to enhance risk awareness, improve resource allocation, and reduce the impact of potential disasters.

Overview

The Asset Damage Prediction System consists of a multi-step approach, including data collection, model training, disaster prediction, and risk awareness generation. The system generates real-time alerts to stakeholders, enabling timely preventive measures.

Features

Calamity Prediction: Forecasts natural disasters (floods, tornadoes, tsunamis) using historical and real-time data.
Asset Damage Prediction: Evaluates potential asset damage based on asset attributes like type, age, and location.
Risk Awareness: Sends alert messages to specified contacts when high-risk scenarios are predicted.

Problem Statement

Develop a system that can:
Predict potential damage to assets within hazard zones.
Provide actionable, asset-specific precautionary measures.
Generate timely alerts to enable preventive measures.

Objectives

Forecast the extent of potential damage to assets during disasters.
Deliver distinct precautionary steps based on asset characteristics and risk levels.
Increase awareness to reduce asset loss and optimize resource allocation.

Data Collection and Preparation

Historical Data: Disaster data (rainfall, humidity, windspeed, etc.) collected for floods, tornadoes, and tsunamis.

Asset Data: Includes details about assets such as windmills and solar panels, covering age, type, price, and location.

Weather APIs: Integrates OpenWeather and Visual Crossing API data to predict weather conditions and improve disaster forecasts.

System Flowchart

Data Collection: Gather and preprocess disaster and asset data.

Weather Prediction: Use weather data to predict the likelihood of disasters.

Asset Damage Prediction: Estimate the potential damage to assets.

Risk Awareness Generation: Alert users with SMS notifications for high-risk scenarios.

Model Training

The system uses a fine-tuned GPT-2 model on historical disaster data to forecast future occurrences and estimate damage risk. Key model features include:

Data Integration: Combines historical and real-time weather data for comprehensive input.
Training Process: GPT-2 learns patterns and relationships in data to improve prediction accuracy.
Weather API Utilization: Integrates current and forecasted weather conditions to enhance prediction reliability.

Risk Awareness Alerts

When a potential disaster is forecasted, the system:
Estimates potential asset damage and calculates expected costs.
Sends real-time SMS alerts to stakeholders via Twilio, ensuring preparedness and timely response.

Technologies Used

Machine Learning Model: Fine-tuned GPT-2
Weather APIs: OpenWeather, Visual Crossing
SMS Notifications: Twilio API

Conclusion

The Asset Damage Prediction System offers a proactive approach to disaster preparedness by accurately predicting potential hazards and assessing asset damage risk in real time. Utilizing the GPT-2 model alongside historical and real-time weather data, this solution empowers stakeholders to take preventive measures that can significantly reduce asset loss and enhance safety during natural disasters. By integrating advanced AI with timely SMS alerts, the system not only increases awareness but also ensures that resources are allocated efficiently to mitigate potential impacts. This project is a step forward in leveraging AI for effective disaster management and asset protection, exemplifying the transformative potential of generative AI in critical real-world applications.

