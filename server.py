from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import praw
from credentials import client_id, client_secret, username, password
app = Flask(__name__)
CORS(app)

@app.route("/members")
def members():
    return {"members":"Wilson"}
@app.route("/posts/<path:url>", methods =['GET'])
def get_posts(url):
    post_data = url
    print("hey")

    positive,neutral,negative = get_reddit_comments(url)
    return jsonify({"positive":positive, "negative":negative, "neutral":neutral}), 200


def get_reddit_comments(url):
    reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=True, username=username, password = password)
    submission = reddit.submission(url=url)
    positive, neutral,negative = sentiment_analysis(submission.comments)
    return positive,neutral,negative
def sentiment_analysis(comments):
    sentiment_analysis = pipeline(model="Dmyadav2001/Sentimental-Analysis")
    positive = []
    neutral = []
    negative = []
    for comment in comments:
        if hasattr(comment, 'body') and comment.body:
            content = comment.body
            if len(content) < 512:
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
    