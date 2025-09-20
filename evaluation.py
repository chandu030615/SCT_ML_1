from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X, y_true):
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    
    return rmse, r2