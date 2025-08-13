FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
WORKDIR /app/rbm
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "rbm.api:app"]
