# ⚡ Quick Start Guide (5 Minutes)

## Step 1: Open Terminal/CMD

Navigate to the extracted folder:
```bash
cd car-price-predictor-v2-fixed
```

## Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will take 1-2 minutes.

## Step 4: Start the App

```bash
uvicorn app:app --reload
```

You should see:
```
╔════════════════════════════════════════╗
║   ✅ Drivewise 2.0 Started             ║
║   📊 Database: SQLite                  ║
║   🚗 Status: Ready                     ║
║   📍 URL: http://127.0.0.1:8000        ║
║   📚 Docs: http://127.0.0.1:8000/docs  ║
╚════════════════════════════════════════╝

INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

## Step 5: Open Browser

Click or copy-paste this link:

**http://127.0.0.1:8000**

## 🎯 Now You Can:

1. **Predict Prices** - Enter any car details to get price
2. **Ask AI** - Ask questions about car prices
3. **Explain** - See how prices are calculated
4. **View Info** - Check model statistics

## ✅ Test It

Try this example:
- Company: **Maruti**
- Model: **Swift**
- Year: **2020**
- Kilometers: **50000**
- Fuel: **Petrol**

**Click "Get Price Prediction"** → You should get a price!

## 🛑 Stop the App

Press `CTRL+C` in terminal

## 📱 API Documentation

Open your browser to:
**http://127.0.0.1:8000/docs**

This shows all available API endpoints with examples.

## ⚠️ Common Issues

### "ModuleNotFoundError"
→ Run: `pip install -r requirements.txt`

### "Address already in use"
→ Port 8000 is busy. Try:
```bash
uvicorn app:app --port 8001
```

### "Connection refused"
→ Make sure uvicorn is running (Step 4)

## 🎉 You're Done!

The app is fully functional with:
- ✅ Beautiful purple UI
- ✅ SQLite database
- ✅ 4 working tabs
- ✅ AI assistant
- ✅ Price predictions
- ✅ No MySQL needed
- ✅ No sklearn DLL issues

Enjoy! 🚗💜


cd /d "c:\Users\K\Downloads\car-price-predictor-v2-fixed\car-price-predictor-fixed"
venv\Scripts\activate
python -m pip install -r requirements.txt
python generate_car_data.py
python -m uvicorn app:app --host 127.0.0.1 --port 8002