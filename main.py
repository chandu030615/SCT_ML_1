from flask import Flask, request, jsonify, render_template
from src.data_loader import load_data
from src.model import HousePricePredictor
import numpy as np

app = Flask(__name__, 
            template_folder='src/templates',
            static_folder='src/static')

# Load and train the model
X_train, X_test, y_train, y_test = load_data()
model = HousePricePredictor()
model.train(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create feature vector
        features = np.array([
            data['overallQual'],
            data['grLivArea'],
            data['garageCars'],
            data['garageArea'],
            data['totalBsmtSF']
        ]).reshape(1, -1)
        
        # Make prediction
        predicted_price = model.predict(features)[0]
        
        # Calculate confidence interval (example implementation)
        confidence_margin = predicted_price * 0.1  # 10% margin
        confidence_interval = [
            predicted_price - confidence_margin,
            predicted_price + confidence_margin
        ]
        
        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'confidence_interval': [round(x, 2) for x in confidence_interval]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/feature_importance')
def feature_importance():
    # Get feature importance from the model
    importance = model.feature_importance()
    features = ['Overall Quality', 'Living Area', 'Garage Cars', 'Garage Area', 'Basement SF']
    
    return jsonify({
        'features': features,
        'importance': importance.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)