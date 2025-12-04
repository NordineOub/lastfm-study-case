# LastFM study case

This project cover the LastFM dataset to answer to 3 questions :
- What is the top 10 songs played in the top 50 longest sessions by tracks count
- Who is the user with the longest sessions
- What is the forecast of his 3 next monthes of session duration

# Setup Instructions

### Import data 

Dataset come from http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html
You can define the DATA_PATH in docker-compose.yml with the path of the downloaded file *userid-timestamp-artid-artname-traid-traname.tsv*

### Launch streamlit with Docker compose
```bash
# Build the image
docker-compose up --build
```