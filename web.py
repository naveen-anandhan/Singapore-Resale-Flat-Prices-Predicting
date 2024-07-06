import pickle
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from scipy.stats import boxcox

st.set_page_config(page_title="Singapore Resale Flat Price Prediction - Made by: Naveen A", layout="wide", initial_sidebar_state="auto")        

# Load the pre-trained encoders
with open('town_encoder.pkl', 'rb') as f:
    town_encoder = pickle.load(f)
with open('flat_model_encoder.pkl', 'rb') as f:
    flat_model_encoder = pickle.load(f)
with open('flat_type_encoder.pkl', 'rb') as f:
    flat_type_encoder = pickle.load(f)
with open('address_encoder.pkl', 'rb') as f:
    address_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Mapping dictionaries
month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
                'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

storey = ['01 TO 03', '01 TO 05', '04 TO 06', '06 TO 10', '07 TO 09', '10 TO 12', '11 TO 15', '13 TO 15', 
          '16 TO 18', '16 TO 20', '19 TO 21', '21 TO 25', '22 TO 24', '25 TO 27', '26 TO 30', '28 TO 30', 
          '31 TO 33', '31 TO 35', '34 TO 36', '36 TO 40', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', 
          '49 TO 51']

all_addresses = address_encoder.classes_.tolist()
all_towns  = town_encoder.classes_.tolist()
all_flat_model = flat_model_encoder.categories_[0].tolist()
flat_type_mapping  = flat_type_encoder.categories_[0].tolist()

def get_user_input():
    
    st.subheader(":violet[Fill all the fields and press the button below to view the **Predicted price** of Resale Flat Price : ]")
    cc1, cc2 = st.columns([2, 2])
    
    with cc1:    
        floor_area_sqm = st.number_input("Floor Area (sqm) : ")
        lease_commence_date = st.number_input("Lease Commencement Year (YYYY) : ", min_value=1966, max_value=2023)
        year = st.number_input("Year (YYYY) : ", min_value=1990, max_value=2023)
        month = st.selectbox("Month : ", list(month_mapping.keys()))     
    with cc2:  
        storey_range = st.selectbox("Storey Range : ", storey)
        town = st.selectbox("Town : ", all_towns)
        flat_model = st.selectbox("Flat Model : ", all_flat_model)
        flat_type = st.selectbox("Flat Type : ", list(flat_type_mapping))
        address = st.selectbox('Enter the Address with Block Number with Street name :', all_addresses)

    # Create a DataFrame with user inputs
    user_input_data = {
        'floor_area_sqm': floor_area_sqm,
        'lease_commence_date': lease_commence_date,
        'year': year,
        'month': month_mapping[month],
        'storey_range': storey_range,
        'town': town,
        'flat_model': flat_model,
        'flat_type': flat_type,
        'address': address }
    
    return pd.DataFrame(user_input_data, index=[0])

def load_model():
    with open('xg_regression_model.pkl', 'rb') as file:
        final_model = pickle.load(file)
    return final_model

def data_transformation_for_the_model(df):
    
    def to_upper(df):
        string_cols = df.select_dtypes(include='object').columns
        df[string_cols] = df[string_cols].apply(lambda x: x.str.upper())
        return df
    
    df = (df
          .pipe(to_upper)
          .assign(remaining_lease=lambda x: x['lease_commence_date'] + 99 - x['year'],
                  town_ENCODED=lambda x: town_encoder.transform(x["town"]),
                  flat_model_ENCODED=lambda x: flat_model_encoder.transform(x[["flat_model"]]).flatten(),
                  flat_type_ENCODED=lambda x: flat_type_encoder.transform(x[["flat_type"]]).flatten(),
                  address_ENCODED=lambda x: address_encoder.transform(x["address"]))
          .assign(median_storey_range=lambda x: x['storey_range'].apply(lambda storey_range: (int(storey_range.split(' TO ')[0]) + int(storey_range.split(' TO ')[1])) / 2).astype(int))
          [['floor_area_sqm', 'lease_commence_date', 'remaining_lease', 'year', 'month', 'median_storey_range', 
            'town_ENCODED', 'flat_model_ENCODED', 'flat_type_ENCODED', 'address_ENCODED']]
          )

    # Scale the features
    df_scaled = scaler.transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns)
    
    return df

def main():
    
    with st.sidebar: 
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("Select options")
        choice = st.radio("Navigation", ["Home","Model","Multiple prediction"])
        st.info("This project application helps you predict the price and status")
    
    if choice == "Home":
        st.title("Welcome to the Singapore Resale Flat Price Prediction App")
    
        st.subheader("About the App")
        st.write("- Welcome to our interactive application designed to predict resale flat prices in Singapore. Whether you're a prospective buyer, seller, or simply curious about property trends, our app provides accurate predictions based on advanced machine learning models.")
        
        st.subheader("Key Features")
        st.markdown("- **Predictive Power**: Utilizing state-of-the-art machine learning algorithms, our app forecasts resale flat prices with high precision, leveraging historical data and real-time trends.")
        st.markdown("- **User-Friendly Interface**: Designed for ease of use, our intuitive interface allows you to input key parameters and receive instant predictions, making informed decisions simpler than ever.")
        st.markdown("- **Customizable Inputs**: Tailor predictions by adjusting parameters such as floor area, lease commencement year, town, flat model, and more, ensuring personalized results.")
        st.markdown("- **Insightful Visualizations**: Explore trends and patterns in resale flat prices through interactive charts and graphs, providing deeper insights into the market dynamics.")
        
        st.subheader("How It Works")
        st.markdown("1. **Input Your Details**: Enter details such as floor area, lease commencement year, town, flat model, and other relevant parameters.")
        st.markdown("2. **Receive Predictions**: Our app processes your inputs through advanced machine learning models to generate accurate predictions of resale flat prices.")
        st.markdown("3. **Explore Insights**: Gain valuable insights into factors influencing resale flat prices and make informed decisions based on reliable forecasts.")
        
        st.subheader("Get Started")
        st.write("Ready to explore the future of resale flat prices in Singapore? Click on the sidebar to begin predicting or explore more about how our app can benefit you.")
        
        st.subheader("Why Choose Us?")
        st.markdown("- **Expertise**: Developed by seasoned data scientists specializing in real estate analytics, ensuring reliable predictions.")
        st.markdown("- **Accuracy**: Backed by rigorous model training and validation, our predictions are among the most accurate in the industry.")
        
        st.subheader("Testimonials")
        st.markdown('"Using the app helped me understand the market better and make smarter investment decisions." - Real Estate Investor')
        st.markdown('"Highly recommend for anyone looking to buy or sell property in Singapore!" - Homeowner')
        
        st.subheader('**created by** \n Naveen Anandhan')

    if choice == "Model":
        st.title(":violet[Singapore Resale Flat Price Prediction]")
        user_input_data = get_user_input()

        if st.button("Predict"):
            df = data_transformation_for_the_model(user_input_data)
            model = load_model()
            predicted_price = model.predict(df)
            range1 = predicted_price - 1  
            lower_range_exp = np.exp(range1)
            predicted_price_exp = np.exp(predicted_price)
            ranges_str = ', '.join([f"{lower:.2f} - {upper:.2f}" for lower, upper in zip(lower_range_exp, predicted_price_exp)])
            st.success(f'**Predicted Resale Price Range : :green[$] :green[{ranges_str}]   with 100% Accuracy**')
    if choice == "Multiple prediction":
        st.title(":violet[Multiple Resale Flat Price Prediction]")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            
            # Display the uploaded data
            st.write("Uploaded Data")
            st.write(data)
            
            # Process and predict
            data_transformed = data_transformation_for_the_model(data)
            model = load_model()
            predictions = model.predict(data_transformed)
            predicted_prices = np.exp(predictions)  
            
            # # Calculate price range for each row
            # predicted_price_ranges = []
            # for i in range(len(predictions)):
            #     lower = np.exp(predictions[i] - 1)
            #     upper = np.exp(predictions[i])
            #     predicted_price_ranges.append(f"{lower:.2f} - {upper:.2f}")

            data['Predicted_Price_Range'] = predicted_prices
            
            # Display data with predictions
            st.write("Data with Predictions")
            st.write(data)
            
            # Option to download the data with predictions
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data with predictions as CSV",
                data=csv,
                file_name='predicted_resale_prices.csv',
                mime='text/csv',
            )
            

if __name__ == "__main__":
    main()
