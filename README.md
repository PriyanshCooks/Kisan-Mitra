🌾 KISAN MITRA – AI Powered Farming Assistant

A complete AI and ML-based web application designed to help farmers with Crop Recommendation, Fertilizer Suggestion, and Plant Disease Detection — all in one platform.

🚨 DISCLAIMER ⚠️

This project is developed as a Proof of Concept (POC) for educational purposes. The data sources used are publicly available datasets and may not be suitable for critical agricultural decisions without further validation. This project demonstrates the potential of AI in precision agriculture when scaled with verified datasets and real-world inputs.

💡 MOTIVATION

Agriculture plays a pivotal role in economies like India, where millions of livelihoods depend on it. With rising challenges like climate change and soil degradation, AI can empower farmers to make smarter, data-driven decisions to maximize yield and reduce losses.

Kisan Mitra brings together:

📊 Crop Recommendation based on soil nutrient data.

💡 Fertilizer Suggestion tailored to crop and soil health.

🌱 Disease Detection using AI-powered image classification from plant leaves.

📊 DATA SOURCES

Crop Recommendation Dataset (Kaggle)

Fertilizer Suggestion Dataset (custom built)

Plant Disease Dataset (PlantVillage)

📌 FEATURES

✅ Crop Recommendation System

Predicts the most suitable crop to grow based on soil nutrient values (N, P, K), temperature, humidity, pH, and rainfall.

✅ Fertilizer Suggestion System

Recommends fertilizers by identifying nutrient deficiencies or excesses in the soil relative to the chosen crop.

✅ Disease Detection System

Detects plant diseases from leaf images using Deep Learning (CNN) and provides diagnosis along with remedies.

🛠️ TECH STACK

Category

Tools & Technologies

Backend

Python, FastAPI, Machine Learning (scikit-learn), Deep Learning (TensorFlow/Keras), OpenCV

Frontend

HTML, CSS, JavaScript

Databases

PostgreSQL (optional), CSV-based lookup for quick data access

Deployment

Heroku / AWS EC2 (or as applicable), Git for version control

Others

Jupyter Notebook, pandas, NumPy, matplotlib, seaborn

🚀 DEPLOYMENT

Backend served using FastAPI with integrated ML/DL models.

Frontend built with simple HTML/CSS/JS for farmer-friendly usability.

Cloud deployment via Heroku/AWS/Render (customizable).

API endpoints tested via Postman.

🖥️ HOW TO RUN LOCALLY

Prerequisites

Git, Python 3.8+, pip

Clone the Repository

git clone https://github.com/your-username/Kisan-Mitra.git
cd Kisan-Mitra

Create Virtual Environment & Install Requirements

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

Run Backend Server

uvicorn app:app --reload

Access Web Application

Open your browser at:

http://127.0.0.1:8000/

💻 USAGE GUIDE

✅ Crop Recommendation: Enter soil N, P, K values, pH, rainfall, temperature, humidity → get best crop.

✅ Fertilizer Suggestion: Enter soil N, P, K values and selected crop → get fertilizer recommendations.

✅ Disease Detection: Upload a plant leaf image → AI predicts disease type and suggests remedies.

🌍 DEMO PREVIEWS

Feature

Preview

Crop Recommendation



Fertilizer Suggestion



Disease Detection



✅ FURTHER IMPROVEMENTS

Improve disease detection with larger and diverse datasets.

Enhance frontend design for better usability.

Add support for multiple regional languages.

Integrate real-time weather APIs for more accurate predictions.

Add mobile responsiveness.

🤝 CONTRIBUTING

Pull requests are welcome! For major changes, open an issue first to discuss what you would like to change. Contributions are highly appreciated.

📜 LICENSE

This project is licensed under the MIT License – see the LICENSE file for details.

📞 CONTACT

For queries, collaboration, or contributions, feel free to reach out via:

📧 Email: your-email@example.com

💼 LinkedIn: Your LinkedIn

⭐ GitHub: Your GitHub