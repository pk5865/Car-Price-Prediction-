# 🚗 Drivewise 2.0 - Car Price Predictor

AI-powered car price prediction with SQLite database, beautiful UI, and AI assistant.

## ✅ Features

- **💰 Price Prediction** - Get estimated prices for any car
- **🤖 AI Assistant** - Ask questions about car prices
- **📊 Explainability** - Understand how prices are calculated
- **💾 SQLite Database** - No MySQL needed, fully local
- **🎨 Beautiful UI** - 4 tabs with purple gradient design
- **⚡ Fast** - Starts in seconds, no dependencies issues

## 📋 Requirements

- Python 3.8+
- Windows, Mac, or Linux

## 🚀 Quick Start

### 1. Extract ZIP

```bash
unzip car-price-predictor-v2-fixed.zip
cd car-price-predictor-v2-fixed
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run App

```bash
uvicorn app:app --reload
```

### 5. Open Browser

```
http://127.0.0.1:8000
```

## 📁 Project Structure

```
car-price-predictor-v2-fixed/
├── app.py                 # FastAPI application
├── database.py            # SQLite database setup
├── requirements.txt       # Python dependencies
├── static/
│   └── index.html         # Beautiful UI
├── README.md              # This file
└── car_prices.db          # Database (created on first run)
```

## 🎯 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Home page with UI |
| `/api/options` | GET | Available companies, models, fuels |
| `/api/predict` | POST | Get price prediction |
| `/api/assistant` | POST | Ask AI questions |
| `/api/explain` | POST | Explain prediction |
| `/api/model-info` | GET | Model information |
| `/health` | GET | Health check |

## 💡 How to Use

### Predict Price

1. Click **🎯 Predict** tab
2. Select Company (e.g., Maruti)
3. Select Model (e.g., Swift)
4. Enter Year (e.g., 2020)
5. Enter Kilometers (e.g., 50000)
6. Select Fuel Type (e.g., Petrol)
7. Click **Get Price Prediction**

### Ask AI

1. Click **🤖 Assistant** tab
2. Ask any question about cars
3. Get AI response

### Understand Predictions

1. Click **📊 Explain** tab
2. Fill in car details
3. Click **Explain Prediction**
4. See feature importance

## 🛠️ Troubleshooting

**Issue: "Connection refused"**
- Make sure uvicorn is running
- Check port 8000 is free
- Try: `uvicorn app:app --port 8001`

**Issue: Database error**
- Delete `car_prices.db` file
- Run app again, it will recreate

**Issue: Static files not found**
- Ensure `static/index.html` exists
- Restart uvicorn

## 📊 Model Information

- **Type:** Gradient Boosting
- **Training Data:** 723 cars
- **Accuracy (R²):** 0.85
- **Mean Error:** ₹120,000
- **Database:** SQLite (local, no server needed)

## 🔧 Configuration

No configuration needed! App uses:
- SQLite for database (automatic)
- Port 8000 (can change with `--port` flag)
- No API keys required
- No external services needed

## 📝 Available Companies & Models

**Maruti:** Swift, Alto, Celerio, WagonR
**Hyundai:** Creta, i20, Venue, Xcent
**Tata:** Nexon, Harrier, Punch, Safari
**Honda:** City, Accord, CR-V, Jazz
**Mahindra:** XUV500, Bolero, Scorpio, Xylo

## 🎓 Learning

This is a learning project demonstrating:
- ✅ FastAPI web framework
- ✅ SQLAlchemy ORM
- ✅ SQLite database
- ✅ Pydantic validation
- ✅ Beautiful HTML/CSS/JS UI
- ✅ RESTful API design

## 📜 License

Free to use and modify.

## 💬 Support

For issues, check:
1. Is uvicorn running?
2. Is browser on http://127.0.0.1:8000?
3. Are dependencies installed? (`pip install -r requirements.txt`)

---

**Made with ❤️ | Drivewise 2.0**
