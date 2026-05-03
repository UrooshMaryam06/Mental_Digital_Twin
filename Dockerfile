FROM python:3.10-slim
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=1000 -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main_new_v2:app", "--host", "0.0.0.0", "--port", "8000"]