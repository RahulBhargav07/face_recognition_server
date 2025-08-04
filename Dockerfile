FROM python:3.9-slim

# Install system dependencies including g++ and build-essential
RUN apt-get update && apt-get install -y \
    g++ \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Upgrade pip first (optional but recommended)
RUN pip install --upgrade pip

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENV PORT=8080
ENV HOST=0.0.0.0

CMD ["python", "main.py"]
