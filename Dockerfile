FROM python:3.11-slim

# Install system deps (bash, Java, build tools)
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

# Optional virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies (ensure Jupyter + nbconvert present)
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Ensure JAVA_HOME (default-java points to installed JDK)
ENV JAVA_HOME=/usr/lib/jvm/default-java

ENTRYPOINT ["streamlit", "run", "main_page_study_case.py", "--server.port=8501", "--server.address=0.0.0.0"]
