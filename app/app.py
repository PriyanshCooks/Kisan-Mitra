from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import numpy as np
import pandas as pd
import requests
import os
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.model import ResNet9

# Initialize Flask app
app = Flask(__name__)

# Load plant disease classification model
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Load crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# Fetch weather data for a given city
def weather_fetch(city_name):
    """
    Fetch temperature and humidity for a city.
    Args:
        city_name (str): Name of the city.
    Returns:
        tuple: (temperature in °C, humidity in %), or None on failure.
    """
    api_key = os.environ.get('WEATHER_API_KEY')
    if not api_key:
        raise ValueError("Weather API key not found in environment variables")
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {'q': city_name.strip(), 'appid': api_key}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if "main" in data:
            temperature = round(data["main"]["temp"] - 273.15, 2)  # Convert Kelvin to Celsius
            humidity = data["main"]["humidity"]
            return temperature, humidity
        return None
    except requests.exceptions.RequestException as e:
        print(f"Weather API request failed: {e}")
        return None

# Predict plant disease from an image
def predict_image(img, model=disease_model):
    """
    Transform image to tensor and predict disease label.
    Args:
        img: Image data in bytes.
        model: Trained disease classification model.
    Returns:
        str: Predicted disease label.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]

# Home page route
@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html', title='KisanMitra - Home')

# Crop recommendation form route
@app.route('/crop-recommend')
def crop_recommend():
    """Render the crop recommendation form page."""
    return render_template('crop.html', title='KisanMitra - Crop Recommendation')

# Fertilizer recommendation form route
@app.route('/fertilizer')
def fertilizer_recommendation():
    """Render the fertilizer recommendation form page."""
    return render_template('fertilizer.html', title='KisanMitra - Fertilizer Suggestion')

# Crop prediction route
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    """Process crop recommendation form and render result."""
    title = 'KisanMitra - Crop Recommendation'
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        weather_data = weather_fetch(city)
        if weather_data:
            temperature, humidity = weather_data
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = crop_recommendation_model.predict(data)[0]
            return render_template('crop-result.html', prediction=prediction, title=title)
        return render_template('try_again.html', title=title)

# Fertilizer recommendation route
@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    """Process fertilizer recommendation form and render result."""
    title = 'KisanMitra - Fertilizer Suggestion'
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['potassium'])

    df = pd.read_csv('app/Data/fertilizer.csv')
    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = max(temp.keys())
    key = temp[max_value]
    key = ('NHigh' if n < 0 else 'Nlow') if key == "N" else \
          ('PHigh' if p < 0 else 'Plow') if key == "P" else \
          ('KHigh' if k < 0 else 'Klow')

    response = Markup(fertilizer_dic[key])
    return render_template('fertilizer-result.html', recommendation=response, title=title)

# Disease prediction route
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    """Process disease prediction form and render result or form page."""
    title = 'KisanMitra - Disease Detection'
    if request.method == 'POST':
        if 'file' not in request.files or not request.files['file']:
            return render_template('disease.html', title=title)
        try:
            img = request.files['file'].read()
            prediction = predict_image(img)
            prediction = Markup(disease_dic[prediction])
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception as e:
            print(f"Disease prediction failed: {e}")
    return render_template('disease.html', title=title)

# Run the Flask app
# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT not set
#     app.run(host='0.0.0.0', port=port, debug=False)