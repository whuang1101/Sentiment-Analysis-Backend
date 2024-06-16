from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import praw
from credentials import client_id, client_secret, username, password
import re
app = Flask(__name__)
CORS(app)
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
def custom_tokenizer(text, **kwargs):
    return tokenizer(text, truncation=True, padding=True, **kwargs)

# Customize the tokenizer to truncate inputs
def custom_tokenizer(text, **kwargs):
    return tokenizer(text, truncation=True, padding=True, **kwargs)
@app.route("/members")
def members():
    return {"members":"Wilson"}

@app.route("/posts/<path:url>", methods =['GET'])
def get_posts(url):
    if not is_valid_reddit_url(url):
        return jsonify({"error": "Invalid URL format"}), 400
    positive,neutral,negative,title = get_reddit_comments(  url)

    return jsonify({"positive":positive, "negative":negative, "neutral":neutral, "title": title}), 200


def get_reddit_comments(url):
    reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=True, username=username, password = password)
    submission = reddit.submission(url=url)
    title = submission.title
    positive, neutral,negative = sentiment_analysis(submission.comments)
    return positive,neutral,negative, title
def sentiment_analysis(comments):
    sentiment_analysis = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=custom_tokenizer
    )   
    positive = []
    neutral = []
    negative = []
    a = 0
    for comment in comments:
        a += 1
        if a == 100: break
        if hasattr(comment, 'body') and comment.body:
            content = comment.body
            analysis = sentiment_analysis(content)
            if analysis[0]['label'] == "POS":
                positive.append({"content": content, 'sentiment': analysis[0]['score']})
            elif analysis[0]['label'] == "NEG":
                negative.append({"content": content, 'sentiment': analysis[0]['score']})
            elif analysis[0]['label'] == "NEU":
                neutral.append({"content": content, 'sentiment': analysis[0]['score']})
    positive = sorted(positive, key=lambda x: x['sentiment'], reverse=True)
    negative = sorted(negative, key=lambda x: x['sentiment'], reverse=True)
    neutral = sorted(neutral, key=lambda x: x['sentiment'], reverse=True)
    return positive, neutral,negative

def is_valid_reddit_url(url):
    # Define the regular expression pattern
    pattern = r'^https:\/\/www\.reddit\.com\/r\/[^\/]+\/comments\/.*$'
    # Test the URL against the pattern
    return bool(re.match(pattern, url))

if __name__  == "__main__":
    app.run()
    