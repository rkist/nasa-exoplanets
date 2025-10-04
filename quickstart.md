# 🚀 Quick Start Guide

Get ExoDetect AI up and running in 5 minutes!

## Step 1: Clone/Download the Project

Download all files into this structure:
```
exoplanet-detection/
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── backend/
│   ├── app.py
│   ├── models.py
│   ├── data_processing.py
│   └── requirements.txt
├── README.md
└── QUICKSTART.md
```

## Step 2: Install Backend Dependencies

```bash
# Navigate to backend folder
cd backend

# Install Python dependencies
pip install -r requirements.txt

# If you encounter issues, try:
pip install flask flask-cors pandas numpy scikit-learn xgboost tensorflow joblib
```

## Step 3: Start the Backend

```bash
# From the backend/ folder
python app.py
```

You should see:
```
Starting ExoDetect AI Backend Server...
Server running on http://localhost:5000
 * Running on http://0.0.0.0:5000
```

**Keep this terminal window open!**

## Step 4: Start the Frontend

Open a **new terminal window** and run:

```bash
# From the frontend/ folder
cd frontend
python -m http.server 8000
```

Or use any other web server you prefer.

## Step 5: Open the Application

Open your browser and go to:
```
http://localhost:8000
```

## 🎯 First Test

1. Click on **Train Model** tab
2. Leave default settings (XGBoost, Combined dataset)
3. Click **Start Training**
4. Wait for training to complete (~30 seconds)
5. Go to **Classify Data** tab
6. Enter sample values:
   - Orbital Period: `3.5`
   - Transit Duration: `2.7`
   - Planetary Radius: `1.2`
7. Click **Classify Exoplanet**

You should see a classification result! 🎉

## 🐛 Troubleshooting

### Backend won't start
- **Error: Module not found**
  - Make sure you're in the `backend/` folder
  - Run `pip install -r requirements.txt` again

- **Error: Port 5000 already in use**
  - Change the port in `backend/app.py`: `app.run(port=5001)`
  - Update `API_BASE_URL` in `frontend/js/main.js` to `http://localhost:5001/api`

### Frontend can't connect to backend
- **CORS errors**
  - Make sure Flask server is running
  - Check that `flask-cors` is installed

- **"Could not connect to server"**
  - Verify backend is running on port 5000
  - Check browser console for errors (F12)

### TensorFlow issues
- **Mac M1/M2**: Use `tensorflow-macos` instead
  ```bash
  pip install tensorflow-macos tensorflow-metal
  ```

- **No GPU warning**: This is fine, CPU training works well for this dataset

## 📊 Test with Real Data

To test batch classification:

1. Create a CSV file `test_data.csv`:
```csv
orbital_period,transit_duration,planetary_radius,eq_temp,stellar_radius,stellar_mass
3.52,2.7,1.2,288,1.0,1.0
10.5,4.1,2.3,450,1.2,1.1
50.2,6.8,0.8,200,0.9,0.95
```

2. Go to **Classify Data** → **Batch Classification**
3. Upload your CSV file
4. Download the results!

## 🎓 Next Steps

- Explore different algorithms (Random Forest, Neural Network, SVM)
- Adjust hyperparameters for better accuracy
- Try training on different datasets
- Check out the **Model Statistics** tab

## 💡 Tips

- XGBoost trains fastest (~10-20 seconds)
- Neural Networks take longer but may be more accurate (1-2 minutes)
- Adjust epochs: fewer = faster, more = potentially better accuracy
- The synthetic data is for demo purposes - integrate real NASA data for production!

## 🔗 Resources

- [NASA Exoplanet Archive](https://exoplanetarchive.ipas.caltech.edu/)
- [Kepler Data](https://archive.stsci.edu/kepler/)
- [TESS Data](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html)

---

**Need help?** Check the full README.md or open an issue!