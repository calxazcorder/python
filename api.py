from flask import Flask, request, jsonify
import joblib
import pandas as pd

from flask_cors import CORS
from utils import select_features_column

app = Flask(__name__)
CORS(app)  # Allow all origins — or use CORS(app, origins=["https://your-frontend.com"])

# Load model and preprocessors once
# def select_features_column(X):
#     return X['features']

model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
y_encoder = joblib.load("y_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("1. Getting request data...")
        data = request.get_json()  # JSON from frontend
        print(f"2. Received data: {data}")

        # Convert price category strings to numbers
        print(f"3. Original price_category: {data['price_category']}")
        if data['price_category'] == '<20':
            data['price_category'] = 1
        elif data['price_category'] == '<50':
            data['price_category'] = 2
        elif data['price_category'] == '100+':
            data['price_category'] = 4
        elif data['price_category'] == '<100':
            data['price_category'] = 3
        
        print(f"4. Converted price_category: {data['price_category']}")

        # Convert JSON to DataFrame (columns must match training X)
        print("5. Creating DataFrame...")
        X_new = pd.DataFrame([{
            'region': data['region'],
            'price_category': data['price_category'],
            'cuisine': data['cuisine'],
            'rating': data['rating'],
            'features': data['features']
        }])
        print(f"6. DataFrame created: {X_new}")
      
        # Apply preprocessing
        print("7. Applying preprocessing...")
        X_transformed = preprocessor.transform(X_new)
        print("8. Preprocessing completed")

        # Predict
        print("9. Making prediction...")
        y_pred_encoded = model.predict(X_transformed)
        print(f"10. Prediction encoded: {y_pred_encoded}")

        # Convert numeric prediction back to original restaurant names
        print("11. Converting prediction...")
        y_pred = y_encoder.inverse_transform(y_pred_encoded)
        print(f"12. Final prediction: {y_pred}")

        return jsonify({"prediction": y_pred[0]})
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(f"ERROR TYPE: {type(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8001))
    app.run(host="0.0.0.0", port=port, debug=False)
