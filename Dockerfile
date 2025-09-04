# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency file first (for caching layers)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set Streamlit environment variable for headless mode
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose default Streamlit port
EXPOSE 8080

# Run Streamlit app using App Runner's dynamic port
CMD ["sh", "-c", "streamlit run ad_app.py --server.port=$PORT --server.address=0.0.0.0"]
