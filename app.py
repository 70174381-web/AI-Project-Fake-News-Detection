from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    news = request.form['news']

    # Convert text to numbers
    data = vectorizer.transform([news])

    # Prediction
    prediction = model.predict(data)[0]

    # Result logic
    if prediction == 1:
        result = "Fake News ❌"
    else:
        result = "Real News ✅"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)