# from flask import Flask, request, jsonify
# from konlpy.tag import Okt
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.linear_model import SGDClassifier
# # from sklearn.utils.fixes import loguniform
# from scipy.stats import loguniform
# from sklearn.model_selection import cross_val_score

# =============================================================================
# Title: ted-movie-sentiment-analysis.py
# Created Date: 5, Aug 2025
# Updated Date: 5, Aug 2025
# Writter: Ted Jung
# Description: 
#   A Flask application for sentiment analysis of movie reviews
#   using a pre-trained model using sklearn
#   TfidfVectorizer/SGDClassifier/RandomizedSearchCV
# How to run:
# > curl -X POST -H "Content-Type: application/json" -d '{"review": "잘한선택"}' http://localhost:5001/predict
# > return {"prediction": "positive" or "negative"}
# =============================================================================


import joblib
from flask import Flask, request, jsonify
from konlpy.tag import Okt

# Create the Flask application instance
app = Flask(__name__)

# Initialize the Korean tokenizer from Konlpy
okt = Okt()

def okt_tokenizer(text):
    """
    A simple tokenizer that uses Konlpy's Okt to extract morphemes (tokens).
    This function must match the tokenizer used when the model was trained.
    """
    return okt.morphs(text)

# Load the pre-trained model and TfidfVectorizer pipeline
# Make sure the file 'ted_naver_movie_sgd_model.joblib' is in the same directory
try:
    loaded_model = joblib.load('ted_naver_movie_sgd_model.joblib')
except FileNotFoundError:
    print("Error: The model file 'ted_naver_movie_sgd_model.joblib' was not found.")
    print("Please make sure the file is in the same directory as this script.")
    # Exit gracefully if the model is not found
    exit()


@app.route('/')
def home():
    """
    A simple home route to confirm the API is running.
    """
    return "Welcome to the Movie Review Sentiment Analysis API!"


# The fix is here: remove the 'review' argument from the function signature.
# Flask handles the request object, and you access its data from inside the function.
@app.route('/predict', methods=['POST'])
def predict():
    """
    This route handles POST requests with a JSON body to make a prediction.
    It expects a JSON object with a key 'review'.
    """
    # Use a try-except block to gracefully handle malformed requests
    try:
        # Get the JSON data from the request body
        data = request.get_json(force=True)

        # Extract the review text from the JSON data.
        # This will raise a KeyError if the 'review' key is missing.
        review_text = data['review']

        # The model's predict method expects a list of strings, so we wrap the review.
        predictions = loaded_model.predict([review_text])

        # The prediction is likely a NumPy array, which needs to be converted to a list
        # before being returned as JSON.
        prediction_result = predictions.tolist()

        # Return the prediction in a structured JSON response
        return jsonify({'prediction': prediction_result[0]})

    except KeyError:
        # Handle the case where the 'review' key is missing in the JSON payload
        return jsonify({'error': 'Missing required key "review" in JSON body'}), 400
    except Exception as e:
        # Handle any other unexpected errors
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    # The host='0.0.0.0' makes the server accessible from outside localhost,
    # which is important for external clients (like your curl command).
    app.run(host='0.0.0.0', port=5001, debug=True)