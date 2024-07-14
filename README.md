# Singapore Resale Flat Price Prediction

This project is a web application designed to predict resale flat prices in Singapore. The application leverages advanced machine learning models to provide accurate predictions based on user inputs. 

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Predictive Power**: Utilizing state-of-the-art machine learning algorithms to forecast resale flat prices with high precision.
- **User-Friendly Interface**: Intuitive interface allows users to input key parameters and receive instant predictions.
- **Customizable Inputs**: Tailor predictions by adjusting parameters such as floor area, lease commencement year, town, flat model, and more.
- **Insightful Visualizations**: Explore trends and patterns in resale flat prices through interactive charts and graphs.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/singapore-resale-flat-price-prediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd singapore-resale-flat-price-prediction
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the pre-trained models and encoders and place them in the project directory:
    - `town_encoder.pkl`
    - `flat_model_encoder.pkl`
    - `flat_type_encoder.pkl`
    - `address_encoder.pkl`
    - `scaler.pkl`
    - `ET_regression_model.pkl`

## Usage

1. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to access the application.

3. Use the sidebar to navigate between the Home, Model, and Multiple Prediction sections.

### Home

- Welcome page with an introduction to the application and its features.

### Model

- Input details such as year, town, flat model, flat type, storey range, floor area, price per sqm, lease commencement year, and address.
- Click on the "Predict" button to get the predicted resale flat price.

### Multiple Prediction

- Upload a CSV file with multiple records to predict resale flat prices for all the entries.
- Download the results with the predicted prices.

## Model Information

The application uses the `ExtraTreesRegressor` model for prediction. The data preprocessing steps include encoding categorical variables and scaling numerical features.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


## Acknowledgements

- Developed by Naveen Anandhan.
- Inspired by the real estate market trends in Singapore.
- Special thanks to the open-source community for the tools and libraries used in this project.
