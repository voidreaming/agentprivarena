FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "vobject>=0.9.9"

COPY . .

EXPOSE 8080
