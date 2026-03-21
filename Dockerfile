
FROM python:3.13-slim



WORKDIR /app



RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*



COPY requirement.txt .  



RUN pip install --no-cache-dir -r requirement.txt  gradio pandas scikit-learn mlflow



COPY . .



EXPOSE 7860



CMD ["python", "-m", "src.app.main"]

