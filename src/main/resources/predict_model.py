"""
Python script that loads the trained insurance model and makes predictions.
Called from Java via subprocess with command-line arguments.
Dynamically loads feature names and encoders from model config.

Usage:
  python predict_model.py age=<val> gender=<val> bmi=<val> kids=<val> smoker=<val> location=<val> \
                          income=<val> employment=<val> health_score=<val> exercise_frequency=<val> \
                          education=<val> marital_status=<val> years_insured=<val> [--model <model_name>]
  
  OR legacy positional format:
  python predict_model.py <age> <gender> <bmi> <kids> <smoker> <location> [--model <model_name>]
  
Model options: random_forest, gradient_boosting, decision_tree, linear_regression, ridge, lasso, 
               elasticnet, svr, knn

Parameters:
  age: Integer (18-100)
  gender: male, female
  bmi: Float (15-50)
  kids: Integer (0-10)
  smoker: yes/no, true/false, 1/0
  location: northeast, southeast, southwest, northwest
  income: low, medium, high, very_high
  employment: employed, self_employed, unemployed, retired
  health_score: Integer (1-10)
  exercise_frequency: Integer (0-7) days per week
  education: high_school, bachelor, master, phd
  marital_status: single, married, divorced, widowed
  years_insured: Integer (0-50)
"""

import sys
import joblib
import numpy as np
import os
import json

def main():
    if len(sys.argv) < 2:
        print("Error: Expected at least 1 argument", file=sys.stderr)
        sys.exit(1)

    model_dir = 'src/main/resources/models'
    selected_model = 'random_forest'  # Default model
    
    # Check if --model argument is provided
    args = sys.argv[1:]
    if '--model' in args:
        model_idx = args.index('--model')
        if model_idx + 1 < len(args):
            selected_model = args[model_idx + 1]
            # Remove --model and its value from args
            args = args[:model_idx] + args[model_idx + 2:]
            sys.argv = [sys.argv[0]] + args
    
    try:
        # Load model configuration
        config_path = os.path.join(model_dir, 'model_config.json')
        if not os.path.exists(config_path):
            print("Error: Model config not found. Train a model first.", file=sys.stderr)
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        feature_names = config.get('feature_names', [])
        encoders_list = config.get('encoders', [])
        available_models = config.get('models', ['random_forest'])
        
        # Validate selected model
        if selected_model not in available_models:
            print(f"Error: Invalid model '{selected_model}'. Available models: {available_models}", file=sys.stderr)
            sys.exit(1)
        
        # Load the selected model
        model_files = {
            'random_forest': 'insurance_model.pkl',
            'decision_tree': 'insurance_model_decision_tree.pkl',
            'linear_regression': 'insurance_model_linear_regression.pkl',
            'gradient_boosting': 'insurance_model_gradient_boosting.pkl',
            'svr': 'insurance_model_svr.pkl',
            'knn': 'insurance_model_knn.pkl',
            'ridge': 'insurance_model_ridge.pkl',
            'lasso': 'insurance_model_lasso.pkl',
            'elasticnet': 'insurance_model_elasticnet.pkl'
        }
        
        model_file = model_files.get(selected_model)
        if not model_file:
            print(f"Error: Model file not found for '{selected_model}'", file=sys.stderr)
            sys.exit(1)
        
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' does not exist. Train the model first.", file=sys.stderr)
            sys.exit(1)
        
        model = joblib.load(model_path)
        
        # Load encoders
        encoders = {}
        for encoder_name in encoders_list:
            encoder_path = os.path.join(model_dir, f'label_encoder_{encoder_name}.pkl')
            if os.path.exists(encoder_path):
                encoders[encoder_name] = joblib.load(encoder_path)
        
        # Parse input arguments into a dict
        # Support both key=value format and legacy positional format
        input_dict = {}
        positional_args = [arg for arg in sys.argv[1:] if not arg.startswith('--') and sys.argv[sys.argv.index(arg) - 1] != '--model']
        
        if len(positional_args) > 0 and '=' in positional_args[0]:
            # key=value format
            for arg in positional_args:
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    # Normalize boolean values
                    if value.lower() in ('true', 'false'):
                        input_dict[key] = 'yes' if value.lower() == 'true' else 'no'
                    else:
                        input_dict[key] = value
        else:
            # Legacy positional format - first 6 args are required, rest are new features
            if len(positional_args) < 6:
                print(f"Error: Expected at least 6 positional arguments for legacy format, got {len(positional_args)}. Args: {positional_args}", file=sys.stderr)
                sys.exit(1)
            
            # Legacy arguments
            age_val = positional_args[0]
            gender_val = positional_args[1].lower()
            bmi_val = positional_args[2]
            kids_val = positional_args[3]
            smoker_val = positional_args[4].lower()
            location_val = positional_args[5]
            
            if smoker_val in ('true', '1', 'yes'):
                smoker_val = 'yes'
            elif smoker_val in ('false', '0', 'no'):
                smoker_val = 'no'
            
            input_dict = {
                'age': age_val,
                'gender': gender_val,
                'bmi': bmi_val,
                'kids': kids_val,
                'smoker': smoker_val,
                'location': location_val,
            }
            
            # Optional new features from positional args (index 6+)
            new_feature_names = ['income', 'employment', 'health_score', 'exercise_frequency',
                                 'education', 'marital_status', 'years_insured']
            for idx, feature_name in enumerate(new_feature_names):
                if len(positional_args) > 6 + idx:
                    input_dict[feature_name] = positional_args[6 + idx]
        
        # Build feature vector based on config
        features = []
        for feature in feature_names:
            if feature.endswith('_encoded'):
                # Categorical feature - find the original column
                col_name = feature.replace('_encoded', '')
                if col_name in encoders:
                    # Try multiple possible input keys
                    value = None
                    if col_name in input_dict:
                        value = input_dict[col_name]
                    elif col_name == 'marital' and 'marital_status' in input_dict:
                        value = input_dict['marital_status']
                    elif col_name == 'employment' and 'employment_status' in input_dict:
                        value = input_dict['employment_status']
                    
                    if value is None:
                        print(f"Error: Missing value for field '{col_name}'. Available keys: {list(input_dict.keys())}", file=sys.stderr)
                        sys.exit(1)
                    
                    try:
                        encoded_value = encoders[col_name].transform([str(value).lower()])[0]
                        features.append(encoded_value)
                    except ValueError as ve:
                        print(f"Error: Unknown value '{value}' for field '{col_name}'. Valid values: {list(encoders[col_name].classes_)}", file=sys.stderr)
                        sys.exit(1)
            else:
                # Numeric feature
                value = None
                
                # Handle feature name variations
                if feature in input_dict:
                    value = input_dict[feature]
                elif feature == 'kids' and 'children' in input_dict:
                    value = input_dict['children']
                elif feature == 'smoker_int' and 'smoker' in input_dict:
                    # Convert yes/no to 1/0
                    smoker_val = input_dict['smoker']
                    value = 1 if smoker_val in ('yes', '1', 'true', True) else 0
                elif feature == 'exercise_frequency' and 'exercise' in input_dict:
                    value = input_dict['exercise']
                elif feature == 'years_insured' and 'years' in input_dict:
                    value = input_dict['years']
                
                if value is not None:
                    try:
                        features.append(float(value))
                    except ValueError:
                        print(f"Error: Invalid numeric value for '{feature}': {value}", file=sys.stderr)
                        sys.exit(1)
                else:
                    print(f"Error: Missing value for field '{feature}'. Available keys: {list(input_dict.keys())}", file=sys.stderr)
                    sys.exit(1)
        
        # Make prediction
        if not features:
            print("Error: No features provided", file=sys.stderr)
            sys.exit(1)
        
        features_array = np.array([features])
        
        # Handle models that require feature scaling (SVR, KNN)
        if selected_model in ('svr', 'knn'):
            # Load the scaler
            scaler_path = os.path.join(model_dir, 'svr_scaler.pkl')
            if not os.path.exists(scaler_path):
                print(f"Error: Scaler file '{scaler_path}' does not exist. Train the model first.", file=sys.stderr)
                sys.exit(1)
            scaler = joblib.load(scaler_path)
            features_array = scaler.transform(features_array)
        
        prediction = model.predict(features_array)[0]
        
        # Output prediction with model name in JSON format for easier parsing
        output = {
            'model': selected_model,
            'prediction': float(prediction)
        }
        print(json.dumps(output))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
