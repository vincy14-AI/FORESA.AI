import pandas as pd
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, render_template
import urllib.request
from datetime import datetime, timedelta
import pandas as pd
import logging
from twilio.rest import Client
from urllib.parse import quote
import sys
import csv
import codecs
from flask import Flask, render_template, request, redirect, url_for
import requests
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import folium
from geopy.geocoders import Nominatim
import os
app = Flask(__name__)
account_sid = 'AC8672e2030a99208be26f375d86ce38f8'
auth_token = 'e70c8c2f790943a4a8c39e30a0c9488f'
client = Client(account_sid, auth_token)


df = pd.read_excel(r"Dataset\Asset_Damage_Due_to_Natural_Disasters_Extended.xlsx")


# Extract unique asset types and locations
asset_types = df['Asset Type'].unique().tolist()
locations = df['Location'].unique().tolist()
disaster_types = ['Tornado', 'Flood', 'Tsunami']
# Load the GPT-2 model and tokenizer
model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as pad token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

API_KEY = 'K3LEU79VGLEDXC2HCH8ZETD79'  # Replace with your Visual Crossing API key
BASE_URL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline'

app = Flask(__name__)

def get_weather_data(city_name, start_date, end_date):
    """
    Fetch weather data from Visual Crossing API for a specified city between start_date and end_date.
    """
    url = f"{BASE_URL}/{city_name}/{start_date}/{end_date}"
    params = {
        'unitGroup': 'metric',
        'elements': 'datetime,precip,precipintensity,precipcover',
        'include': 'days',
        'key': API_KEY,
        'contentType': 'json'
    }

    response = requests.get(url, params=params)
    
    try:
        data = response.json()
        if 'error' in data:
            return None, f"Error: {data['error']['message']}"

        if 'days' in data and isinstance(data['days'], list):
            return data['days'], None
        else:
            return None, "Unexpected format for weather data."
    except Exception as e:
        return None, f"Error fetching data: {e}"

def preprocess_weather_data_flood(weather_data):
    if not isinstance(weather_data, list):
        return 0
    total_precipitation = sum(day.get('precip', 0) for day in weather_data if isinstance(day, dict))
    return total_precipitation

def calculate_risk_level(total_precipitation):
    if total_precipitation >= 100:
        return "High Risk"
    elif 50 <= total_precipitation < 100:
        return "Moderate Risk"
    else:
        return "Low Risk"

def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="FloodRiskApp/1.0 (your_email@example.com)")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    return None, None

def predict_high_risk_zones(start_date, end_date):
    cities = [
        
        # Major Metros
    "Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai", "Hyderabad", "Ahmedabad", "Pune","Wayanad","Nellore"

    # North India
    # "Jaipur", "Lucknow", "Kanpur", "Varanasi", "Amritsar", "Chandigarh", "Agra", "Ludhiana",
    # "Meerut", "Ghaziabad", "Noida", "Gurgaon", "Faridabad", "Jammu", "Shimla", "Dehradun",
    # "Aligarh", "Mathura", "Bareilly", "Haridwar", "Moradabad", "Firozabad", "Saharanpur",
    # "Muzaffarnagar", "Rohtak", "Hisar", "Panipat", "Karnal", "Kurukshetra",

    # South India
    "Mysore", "Coimbatore", "Madurai", "Tiruchirappalli", "Thiruvananthapuram", "Vijayawada",
    "Visakhapatnam", "Mangalore", "Kochi", "Kozhikode", "Hubli", "Belgaum", "Gulbarga", "Warangal",
    "Tirupati", "Salem", "Erode", "Vellore", "Anantapur", "Kurnool", "Nellore", "Guntur",
    "Kollam", "Thrissur", "Kannur", "Karimnagar", "Nizamabad",

    # East India
    "Bhubaneswar", "Cuttack", "Jamshedpur", "Dhanbad", "Ranchi", "Patna", "Guwahati", "Silchar",
    "Shillong", "Imphal", "Aizawl", "Agartala", "Kohima", "Gangtok", "Darjeeling", "Purnia",
    "Muzaffarpur", "Gaya", "Siliguri", "Dibrugarh", "Tezpur", "Tinsukia", "Dimapur",

    # West India
    # "Surat", "Vadodara", "Rajkot", "Nashik", "Nagpur", "Aurangabad", "Kolhapur", "Solapur",
    # "Nanded", "Udaipur", "Jodhpur", "Bikaner", "Jamnagar", "Bhavnagar", "Gandhinagar", "Porbandar",
    # "Vapi", "Valsad", "Latur", "Beed", "Satara", "Sangli", "Barmer", "Alwar", "Ajmer", "Bhilwara",
    # "Palghar", "Ratnagiri", "Panaji", "Margao",

    # Central India
    # "Bhopal", "Indore", "Jabalpur", "Gwalior", "Raipur", "Bilaspur", "Durg", "Ujjain",
    # "Sagar", "Rewa", "Satna", "Chhindwara", "Betul", "Korba", "Ambikapur", "Jagdalpur",
    # "Katni", "Ratlam", "Dewas",

    # Union Territories
    # "Port Blair", "Puducherry", "Daman", "Diu", "Leh", "Srinagar", "Kavaratti", "Aizawl",
    # "Itanagar", "Dispur", "Kargil", "Silvassa", "Nicobar", "Lakshadweep", "Manali"
    ]

    high_risk_cities = []
    moderate_risk_cities = []

    for city in cities:
        weather_data, error = get_weather_data(city, start_date, end_date)
        if weather_data:
            total_precipitation = preprocess_weather_data_flood(weather_data)
            risk_level = calculate_risk_level(total_precipitation)
            disaster_type = "flood"  # For now, we're assuming a flood. You can adjust this based on actual input.

            # Call send_sms_notification with the appropriate details
            statement = f"Flood risk identified for {city} with {total_precipitation} mm of rainfall."
            send_sms_notification(statement, risk_level.lower(), disaster_type)

            if risk_level == "High Risk":
                high_risk_cities.append((city, total_precipitation))
            elif risk_level == "Moderate Risk":
                moderate_risk_cities.append((city, total_precipitation))

    return high_risk_cities, moderate_risk_cities

def create_flood_risk_map(high_risk_cities, moderate_risk_cities):
    india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    for city, precip in high_risk_cities:
        lat, lon = get_coordinates(city)
        if lat and lon:
            folium.Marker(
                location=[lat, lon],
                popup=f"{city}: {precip} mm (High Risk)",
                icon=folium.Icon(color="red")
            ).add_to(india_map)

    for city, precip in moderate_risk_cities:
        lat, lon = get_coordinates(city)
        if lat and lon:
            folium.Marker(
                location=[lat, lon],
                popup=f"{city}: {precip} mm (Moderate Risk)",
                icon=folium.Icon(color="orange")
            ).add_to(india_map)

    map_path = os.path.join("static", "flood_risk_zones.html")
    india_map.save(map_path)
    return map_path

# Flask routes

def send_sms_notification(statement, risk_level, disaster_type):
    # Define precautionary steps for each risk level
    precautions = {
        'high': {
            'flood': "Evacuate immediately to higher ground. Avoid driving through floodwaters.",
            'tornado': "Seek shelter in a sturdy building or underground. Avoid windows and doors.",
            'tsunami': "Move to higher ground immediately. Follow local evacuation orders."
        },
        'moderate': {
            'flood': "Monitor local weather updates and be prepared to move to higher ground if needed.",
            'tornado': "Be alert for tornado warnings and seek shelter if a tornado is approaching.",
            'tsunami': "Stay away from the coast and follow updates from local authorities."
        }
    }
    
    # Extract the appropriate precautionary message based on risk level
    precaution_message = precautions.get(risk_level, {}).get(disaster_type, "Stay safe and follow local guidelines.")
    
    # Send the SMS with the precautionary message
    sms_message = f"{statement}\nPrecautionary Steps: {precaution_message}"
    # Code to send SMS goes here, e.g., using an SMS API
    try:
            recipient_numbers = ['+918668192950', '+918220490621']

            for number in recipient_numbers:
                message = client.messages.create(
                    from_='+16505437393',
                    body=sms_message,
                    to=number
                )
                logging.info(f"SMS sent with SID: {message.sid}")
    except Exception as e:
            logging.error(f"Failed to send SMS: {e}")
    print(f"Sending SMS: {sms_message}")

@app.route('/')
def home():
    return render_template('data_input.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    high_risk_cities, moderate_risk_cities = predict_high_risk_zones(start_date, end_date)

    if high_risk_cities or moderate_risk_cities:
        map_path = create_flood_risk_map(high_risk_cities, moderate_risk_cities)
        return render_template('result.html', map_path=map_path)
    else:
        return render_template('no_risk.html')




@app.route('/asset_damage_prediction', methods=['GET', 'POST'])
def predict_damage():
    if request.method == 'POST':
        # Get inputs from the form
        location = request.form['location']
        asset_type = request.form['assetType']
        disaster_type = request.form['disasterType']
        asset_age = int(request.form['assetAge'])  # Assuming the input is an integer
        asset_price = float(request.form['assetPrice'])
        # Filter the dataset based on selected location, asset type, and disaster type
        filtered_df = df[(df['Location'] == location) &
                         (df['Asset Type'] == asset_type) &
                         (df['Disaster Type'] == disaster_type)].copy()

        if filtered_df.empty:
            return render_template('result1.html', error="No data available for the selected options.")

        # Calculate the average loss ratio from the historical data
        filtered_df['Loss Ratio'] = filtered_df['Damage Cost ($)'] / filtered_df['Asset Exact Value ($)']
        avg_loss_ratio = filtered_df['Loss Ratio'].mean()

        # Estimate the loss based on the average loss ratio and age
        # Assuming an average asset value from the filtered data
        example_asset_value = filtered_df['Asset Exact Value ($)'].mean()
        estimated_loss = avg_loss_ratio * asset_price * (1 - (asset_age * 0.01))
        if estimated_loss > asset_price:
            # Scale down the estimated loss relative to the asset price using a proportion
            depreciation_factor = 1 - (asset_age * 0.01)
            scaling_factor = asset_price / estimated_loss  # Scale down based on the excess amount
            estimated_loss = asset_price * scaling_factor * depreciation_factor  # Adjust based on asset age 
        #estimated_loss = avg_loss_ratio * example_asset_value * (1 - (asset_age * 0.01))  # Simple depreciation model

        # Return the result
        return render_template('result1.html', location=location, asset_type=asset_type, 
                               disaster_type=disaster_type, asset_age=asset_age,asset_price=asset_price,
                               estimated_loss=round(estimated_loss, 2))

    return render_template('asset.html', asset_types=asset_types, locations=locations, disaster_types=disaster_types)


if __name__ == '__main__':
    app.run(debug=True)
