import streamlit as st
import requests
import json

# Define the URL of your FastAPI endpoint
# Make sure this matches the address where your FastAPI app is running
FASTAPI_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Sound Realty Home Value Estimator",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Sound Realty Home Value Estimator")
st.write("Enter the details of the property to get an estimated selling price.")

# Use a Streamlit form to collect all user inputs
with st.form("house_prediction_form"):
    st.header("Property Details")

    # Use columns to create a two-column layout for a cleaner look
    col1, col2 = st.columns(2)
    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=0, value=3, help="Number of bedrooms (e.g., 3)")
        bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.5, step=0.5, help="Number of bathrooms (e.g., 2.5)")
        sqft_living = st.number_input("Sqft Living", min_value=1, value=1500, help="Square footage of the living area")
        sqft_lot = st.number_input("Sqft Lot", min_value=1, value=5000, help="Square footage of the lot")
        floors = st.number_input("Floors", min_value=0.5, value=1.0, step=0.5, help="Number of floors")
        waterfront = st.radio("Waterfront", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True, help="Is there a waterfront property?")
    
    with col2:
        view = st.slider("View Quality", min_value=0, max_value=4, value=0, help="Quality of view (0-4, 4 being best)")
        condition = st.slider("Condition", min_value=1, max_value=5, value=3, help="Condition of the home (1-5, 5 being excellent)")
        grade = st.slider("Grade", min_value=1, max_value=13, value=7, help="Overall grade of the home")
        sqft_above = st.number_input("Sqft Above", min_value=0, value=1500, help="Square footage of living area above ground")
        sqft_basement = st.number_input("Sqft Basement", min_value=0, value=0, help="Square footage of the basement")
        yr_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000, help="Year the house was built")
        yr_renovated = st.number_input("Year Renovated", min_value=0, value=0, help="Year of last renovation")
    
    st.subheader("Location Details")
    col3, col4, col5 = st.columns(3)
    with col3:
        zipcode = st.text_input("Zipcode", value="98103", max_chars=5, help="5-digit zip code of the property")
    with col4:
        lat = st.number_input("Latitude", value=47.65, format="%.6f", help="Latitude coordinate")
    with col5:
        long = st.number_input("Longitude", value=-122.35, format="%.6f", help="Longitude coordinate")
    
    st.subheader("Post-Sale Details (from 2015 data)")
    col6, col7 = st.columns(2)
    with col6:
        sqft_living15 = st.number_input("Sqft Living 2015", min_value=0, value=1500, help="Average living area of the 15 nearest neighbors")
    with col7:
        sqft_lot15 = st.number_input("Sqft Lot 2015", min_value=0, value=5000, help="Average lot size of the 15 nearest neighbors")

    submitted = st.form_submit_button("Get Estimated Price")

    if submitted:
        # Create a dictionary with user inputs to be sent to the API
        data_payload = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "waterfront": waterfront,
            "view": view,
            "condition": condition,
            "grade": grade,
            "sqft_above": sqft_above,
            "sqft_basement": sqft_basement,
            "yr_built": yr_built,
            "yr_renovated": yr_renovated,
            "zipcode": zipcode,
            "lat": lat,
            "long": long,
            "sqft_living15": sqft_living15,
            "sqft_lot15": sqft_lot15
        }

        # Display a loading message while waiting for the response
        st.info("Sending request to the prediction API...")
        
        try:
            # Call the FastAPI endpoint using a POST request
            response = requests.post(FASTAPI_URL, json=data_payload)
            response.raise_for_status() # This will raise an exception for bad status codes (4xx or 5xx)

            # Get the predicted price from the JSON response
            prediction_result = response.json()
            predicted_price = prediction_result.get("predicted_price")

            if predicted_price is not None:
                st.success(f"### Estimated Price: ${predicted_price:,.2f} 💰")
                st.balloons()
            else:
                st.error("Prediction could not be retrieved from the API response.")

        except requests.exceptions.ConnectionError:
            st.error("Error: Could not connect to the FastAPI server. Please ensure the server is running.")
        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP Error: {http_err} - {response.json().get('detail', 'No detailed message.')}")
        except json.JSONDecodeError:
            st.error("Error: Could not decode JSON response from the server.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown("For internal use by Sound Realty professionals.")

