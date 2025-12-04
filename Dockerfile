FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-jdk \
    procps \
    bash \
    gcc \
    g++ \
    wget \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure JAVA_HOME (default-java points to installed JDK)
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Set default values for environment variables
ENV STREAMLIT_APP=app/main_page_study_case.py
ENV DATA_PATH=""

