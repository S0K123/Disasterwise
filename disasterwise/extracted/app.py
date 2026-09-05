from flask import Flask, request, jsonify

app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return "Disaster Risk Detection Backend Running"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    rainfall = data['rainfall']
    temperature = data['temperature']
    river_level = data['river_level']

    # Simple prediction logic (example)
    if rainfall > 200 or river_level > 5:
        risk = "High Disaster Risk"
    else:
        risk = "Low Risk"

    return jsonify({"prediction": risk})
if __name__ == '__main__':
    app.run(debug=True)