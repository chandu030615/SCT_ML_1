import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def create_feature_interactions(df, features):
    """Create interaction features between numerical columns"""
    interactions = {}
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feature_name = f"{features[i]}_{features[j]}_interaction"
            interactions[feature_name] = df[features[i]] * df[features[j]]
    return pd.DataFrame(interactions)

def create_polynomial_features(df, features, degree=2):
    """Create polynomial features up to specified degree"""
    polynomials = {}
    for feature in features:
        for d in range(2, degree + 1):
            polynomials[f"{feature}_power_{d}"] = df[feature] ** d
    return pd.DataFrame(polynomials)

def load_data(test_size=0.2, random_state=42):
    # Load training data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'train.csv')
    train_data = pd.read_csv(data_path)
    
    # Phase 1: Basic features
    basic_features = ['1stFlrSF', 'BedroomAbvGr', 'FullBath']
    
    # Phase 2: Top correlated features
    advanced_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
    
    # Phase 3: Additional engineered features
    categorical_features = ['MSZoning', 'Neighborhood', 'BldgType']
    numerical_features = advanced_features.copy()
    
    # Create derived features
    train_data['TotalSF'] = train_data['1stFlrSF'] + train_data['2ndFlrSF'] + train_data['TotalBsmtSF']
    train_data['Age'] = train_data['YrSold'] - train_data['YearBuilt']
    train_data['IsNew'] = (train_data['YrSold'] - train_data['YearBuilt'] <= 2).astype(int)
    train_data['HasPool'] = train_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    train_data['TotalBaths'] = train_data['FullBath'] + 0.5 * train_data['HalfBath'] + \
                               train_data['BsmtFullBath'] + 0.5 * train_data['BsmtHalfBath']
    
    # Add derived features to numerical features list
    derived_features = ['TotalSF', 'Age', 'IsNew', 'HasPool', 'TotalBaths']
    numerical_features.extend(derived_features)
    
    # Create feature interactions and polynomials
    interactions = create_feature_interactions(train_data, numerical_features)
    polynomials = create_polynomial_features(train_data, numerical_features)
    
    # Combine all numerical features
    X_numeric = pd.concat([train_data[numerical_features], interactions, polynomials], axis=1)
    
    # Process categorical features
    X_categorical = train_data[categorical_features]
    
    # Create preprocessing pipelines
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, X_numeric.columns),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Combine all features and preprocess
    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = train_data['SalePrice']
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    return train_test_split(X_processed, y, test_size=test_size, random_state=random_state)