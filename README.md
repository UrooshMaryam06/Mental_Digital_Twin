# Electricity Demand & Price Forecasting MLOps Pipeline
It is an end-to-end machine learning system that predicts electricity demand and prices using historical and weather data, and continuously updates the model using MLOps practices like deployment, monitoring, and retraining

## Run

- Start the FastAPI backend:

```bash
cd mlops_project
uvicorn main_new_v2:app --host 0.0.0.0 --port 8000
```

- Start the Streamlit frontend:

```bash
cd app-frontend
streamlit run Home.py
```
