from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# load trained model & vectorizer
model = pickle.load(open("artifacts/model.pkl", "rb"))
vectorizer = pickle.load(open("artifacts/preprocessing.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction API (JS fetch uses this)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "email" not in data:
        return jsonify({"error": "No email text provided"}), 400

    email_text = data["email"]

    # vectorize
    email_vec = vectorizer.transform([email_text])

    # predict
    pred = model.predict(email_vec)

    if pred[0] == 1:
        return jsonify({"result": "SPAM"})
    else:
        return jsonify({"result": "NOT_SPAM"})

if __name__ == "__main__":
    app.run(debug=True)
