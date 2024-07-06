# Singapore Resale Flat Price Prediction

This project is an interactive web application built using Streamlit that predicts the resale prices of flats in Singapore. It leverages advanced machine learning models to provide accurate predictions based on user inputs and historical data.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [How It Works](#how-it-works)
- [Contributing](#contributing)

## Overview

The Singapore Resale Flat Price Prediction app helps users predict the price of resale flats in Singapore. The app allows users to input key parameters such as floor area, lease commencement year, town, flat model, flat type, and address. Based on these inputs, the app provides a predicted price range using a pre-trained machine learning model.

## Features

- **Predictive Power**: Utilizes state-of-the-art machine learning algorithms for accurate price predictions.
- **User-Friendly Interface**: Intuitive interface for easy input of parameters and instant predictions.
- **Customizable Inputs**: Tailor predictions by adjusting various parameters.
- **Insightful Visualizations**: Explore trends and patterns in resale flat prices through interactive charts and graphs.

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run web.py
   ```
2. Open your browser and navigate to `http://localhost:8501`.

3. Use the sidebar to navigate between the Home, Model, and Multiple Prediction pages.

4. On the Model page, input the required parameters and click "Predict" to get the predicted price range.

5. On the Multiple Prediction page, upload a CSV file with the required data to get predictions for multiple entries.

## Data

The app requires the following input data:
- Floor Area (sqm)
- Lease Commencement Year
- Year
- Month
- Storey Range
- Town
- Flat Model
- Flat Type
- Address

## Model

The app uses a pre-trained XGBoost regression model for predicting the resale prices. The model was trained on historical resale flat data and saved as a pickle file.

## How It Works

1. **Input Your Details**: Enter details such as floor area, lease commencement year, town, flat model, and other relevant parameters.
2. **Receive Predictions**: The app processes your inputs through the machine learning model to generate accurate predictions of resale flat prices.
3. **Explore Insights**: Gain valuable insights into factors influencing resale flat prices and make informed decisions based on reliable forecasts.

## Contributing

Contributions are welcome! Please fork the repository and use a feature branch. Pull requests are warmly welcome.


## Contact

Created by [Naveen Anandhan](https://www.linkedin.com/in/naveen-anandhan-8b03b62a5/?trk=public-profile-join-page) - feel free to contact me!
