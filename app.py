import streamlit as st
import pandas as pd
from tensorflow import keras
from src.preprecessing import preprocess_input as preprocess

# Load model
model = keras.models.load_model("Models/spaceship_model.h5")

st.title("🚀 Spaceship Titanic Prediction App")

# Collect user input
homeplanet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
name = st.text_input("Name", value="Name")
passenger_id = st.text_input("PassengerId" , value="0001")
cryosleep = st.selectbox("CryoSleep", [True, False])
cabin = st.text_input("Cabin", value="B/123/S")
age = st.number_input("Age", min_value=0, max_value=100, value=30)
vip = st.selectbox("VIP", [True, False])
room_service = st.number_input("RoomService", min_value=0, value=0)
food_court = st.number_input("FoodCourt", min_value=0, value=0)
shopping_mall = st.number_input("ShoppingMall", min_value=0, value=0)
spa = st.number_input("Spa", min_value=0, value=0)
vrdeck = st.number_input("VRDeck", min_value=0, value=0)

# Create DataFrame
input_data = pd.DataFrame({
    "HomePlanet": [homeplanet],
    "Destination": [destination],
    "Name": [name],
    "PassengerId": [passenger_id],
    "CryoSleep": [cryosleep],
    "Cabin": [cabin],
    "Age": [age],
    "VIP": [vip],
    "RoomService": [room_service],
    "FoodCourt": [food_court],
    "ShoppingMall": [shopping_mall],
    "Spa": [spa],
    "VRDeck": [vrdeck]
})

# TODO: Apply the same preprocessing you used during training
# Example: encoding, scaling, etc.
processed_input = preprocess(input_data)

# For demo, assume input_data is already numeric
prediction = model.predict(processed_input)
result = "Transported" if prediction[0][0] > 0.5 else "Not Transported"

st.subheader("Prediction:")
st.write(result)
