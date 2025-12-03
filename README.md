# LastFM study case

This project cover the LastFM dataset to answer to 3 questions :
- What is the top 10 songs played in the top 50 longest sessions by tracks count
- Who is the user with the longest sessions
- What is the forecast of his 3 next monthes of session duration

# Setup Instructions

### Launch streamlit with Docker 
```bash
# Build and start Jupyter Lab
docker build

### run scripts with Docker
```bash
# Build the image
docker build -t lastfm:latest . 
# Run a Python script
sudo docker run -p 8501:8501 lastfm
```