from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

app = Flask(__name__)
app.model = pickle.load(open("model.pickle", "rb"))


@app.route("/api/american", methods=["POST"])
def american():
    content = request.json
    predicted = app.model.predict([content["text"]])
    return jsonify({"is_american": int(predicted[0]), "version": "teste", "model_date": "teste"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
