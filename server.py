from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import praw
from credentials import client_id, client_secret, username, password
app = Flask(__name__)
CORS(app)
sentiment_analysis = pipeline(model="Dmyadav2001/Sentimental-Analysis")

@app.route("/members")
def members():
    return {"members":"Wilson"}
@app.route("/posts/<path:url>", methods =['GET'])
def get_posts(url):
    post_data = url
    positive,neutral,negative, title = get_reddit_comments(url)
    return jsonify({"positive":positive, "negative":negative, "neutral":neutral, "title":title}), 200


def get_reddit_comments(url):
    reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=True, username=username, password = password)
    submission = reddit.submission(url=url)
    positive, neutral,negative = sentiment_analysis(submission.comments)
    title = submission.title
    return positive,neutral,negative, title
def sentiment_analysis(comments):

    positive = []
    neutral = []
    negative = []
    for comment in comments:
        content = comment.body
        analysis = sentiment_analysis(content)
        if analysis[0]['label'] == "LABEL_1":
            positive.append({"content": content, 'sentiment': analysis[0]['label']})
        elif analysis[0]['label'] == "LABEL_2":
            neutral.append({"content": content, 'sentiment': analysis[0]['label']})
        else:
            negative.append({"content": content, 'sentiment': analysis[0]['label']})
    return positive, neutral,negative
if __name__  == "__main__":
    app.run(debug=True)
    