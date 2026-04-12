from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "artifacts", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "artifacts", "preprocessing.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "email" not in data:
        return jsonify({"error": "No email text provided"}), 400

    email_text = data["email"]

    email_vec = vectorizer.transform([email_text])
    pred = model.predict(email_vec)

    return jsonify({"result": "SPAM" if pred[0] == 1 else "NOT_SPAM"})

if __name__ == "__main__":
    app.run(debug=True)