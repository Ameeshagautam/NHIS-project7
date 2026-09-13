"""
Disaster Tweet Classifier - Flask Web App
Loads the trained model and vectorizer, and serves a simple web UI
where users can paste tweet text and get a prediction.
"""
import re
import string
import pickle

from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

# --- One-time setup: make sure NLTK data is available ---
for pkg in ['stopwords', 'punkt', 'punkt_tab', 'wordnet']:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_SYMBOL_PATTERN = re.compile(r'#')
NUMBER_PATTERN = re.compile(r'\d+')
PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def clean_text(text):
    """Same cleaning pipeline used during training - must match exactly."""
    text = text.lower()
    text = text.replace('&amp;', ' ')
    text = URL_PATTERN.sub('', text)
    text = MENTION_PATTERN.sub('', text)
    text = HASHTAG_SYMBOL_PATTERN.sub('', text)
    text = NUMBER_PATTERN.sub('', text)
    text = text.translate(PUNCT_TABLE)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


# --- Load the trained model and vectorizer ---
with open('disaster_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


def predict_tweet(raw_text):
    """Runs the same preprocessing + feature engineering used in training,
    then returns (label, confidence_percent)."""
    cleaned = clean_text(raw_text)
    tfidf_features = vectorizer.transform([cleaned])

    extra_features = [[
        len(raw_text),
        1 if '#' in raw_text else 0,
        1 if '@' in raw_text else 0,
    ]]

    combined = hstack([tfidf_features, extra_features])

    prediction = model.predict(combined)[0]
    probability = model.predict_proba(combined)[0][prediction]

    label = 'Disaster' if prediction == 1 else 'Not Disaster'
    confidence = round(probability * 100, 1)
    return label, confidence


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    confidence = None
    tweet_text = ''

    if request.method == 'POST':
        tweet_text = request.form.get('tweet_text', '').strip()
        if tweet_text:
            result, confidence = predict_tweet(tweet_text)

    return render_template(
        'index.html',
        result=result,
        confidence=confidence,
        tweet_text=tweet_text,
    )


if __name__ == '__main__':
    app.run(debug=True)
