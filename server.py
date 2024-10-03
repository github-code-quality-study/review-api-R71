import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

# Define the allowed locations
ALLOWED_LOCATIONS = {    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
    "El Paso, Texas", "Escondido, California", "Fresno, California",
    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
    "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
    "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
}

TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def filter_reviews(self, location: str, start_date: str, end_date: str) -> list:
        """Filter reviews based on location and date range."""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None

        filtered_reviews = [
            review for review in reviews
            if (not location or review['Location'] == location)
            and (not start_dt or datetime.strptime(review['Timestamp'], TIMESTAMP_FORMAT) >= start_dt)
            and (not end_dt or datetime.strptime(review['Timestamp'], TIMESTAMP_FORMAT) <= end_dt)
            and review['Location'] in ALLOWED_LOCATIONS
        ]
        return filtered_reviews

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # parse the QUERY_STRING
            query = parse_qs(environ.get('QUERY_STRING',''))
            location = query.get('location', [None])[0]
            start_date = query.get('start_date', [None])[0]
            end_date =  query.get('end_date',[None])[0]
            filtered_reviews = reviews
            
            # Filter reviews
            filtered_reviews = self.filter_reviews(location, start_date, end_date)

            # add sentiment analysis to each review
            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            # sort reviews by sentiment compound score in descending order
            filtered_reviews = sorted(filtered_reviews, key=lambda review: review['sentiment']['compound'], reverse=True)

            # return the filtered and sorted reviews as JSON
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            try:
            # read the request body
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(content_length)

                query_string = request_body.decode('utf-8')
                query_params = parse_qs(query_string)
                
                # distructure the query parameter to riew body and location
                review_body = query_params.get('ReviewBody', '')[0]
                location = query_params.get('Location', '')[0]

                # ensure both ReviewBody and Location are provided
                if not review_body or not location:
                    raise ValueError("ReviewBody and Location are required.")

                # Ensure the location is valid
                if location not in ALLOWED_LOCATIONS:
                    raise ValueError(f"Location '{location}' is not allowed.")

                # generate a unique UUID and current timestamp
                review_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

                # create the new review entry
                new_review = {
                    'ReviewId': review_id,
                    'ReviewBody': review_body,
                    'Location': location,
                    'Timestamp': timestamp,                    
                }

                # add the new review to the reviews list
                reviews.append(new_review)

                response_body = json.dumps(new_review, indent=2).encode('utf-8')

                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                
                return [response_body]
            
            except Exception as e:
                error_response = json.dumps({'error': str(e)}).encode('utf-8')
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(error_response)))
                ])
                
                return [error_response]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
        