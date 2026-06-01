# ML Models Added to Insurance Cost Prediction Application

## Overview
Your application now includes **9 machine learning models** for predicting insurance costs (up from 3 original models). This provides flexibility in choosing the best-performing model for your use case.

---

## Available Models

### 1. **Random Forest** ⭐ (DEFAULT)
- **Type:** Ensemble (Tree-based)
- **Description:** Combines multiple decision trees to make predictions, averaging their results
- **Pros:** High accuracy, handles non-linear relationships well, robust to outliers
- **Cons:** Slower predictions, less interpretable
- **File:** `insurance_model.pkl`

### 2. **Gradient Boosting** 🚀
- **Type:** Ensemble (Sequential)
- **Description:** Builds trees sequentially, each correcting errors of previous trees
- **Pros:** Often achieves highest accuracy, good generalization
- **Cons:** Slower to train, requires careful tuning
- **File:** `insurance_model_gradient_boosting.pkl`

### 3. **Decision Tree**
- **Type:** Tree-based
- **Description:** Single tree making decisions based on feature values
- **Pros:** Fast, interpretable, no scaling needed
- **Cons:** Prone to overfitting, less accurate than ensemble methods
- **File:** `insurance_model_decision_tree.pkl`

### 4. **Linear Regression**
- **Type:** Linear
- **Description:** Models relationships as linear combinations of features
- **Pros:** Simple, interpretable, fast
- **Cons:** Assumes linear relationships, may underfit complex data
- **File:** `insurance_model_linear_regression.pkl`

### 5. **Ridge Regression** 🔧
- **Type:** Linear with L2 Regularization
- **Description:** Linear regression with penalty for large coefficients
- **Pros:** Prevents overfitting, handles multicollinearity
- **Cons:** Less sparse than Lasso
- **File:** `insurance_model_ridge.pkl`

### 6. **Lasso Regression** ✨
- **Type:** Linear with L1 Regularization
- **Description:** Linear regression with sparse penalty
- **Pros:** Feature selection (sets some coefficients to zero), interpretable
- **Cons:** May struggle with correlated features
- **File:** `insurance_model_lasso.pkl`

### 7. **Elastic Net** ⚙️
- **Type:** Linear with L1 + L2 Regularization
- **Description:** Combines Ridge and Lasso regularization
- **Pros:** Best of both worlds - handles correlated features, does feature selection
- **Cons:** More hyperparameters to tune
- **File:** `insurance_model_elasticnet.pkl`

### 8. **Support Vector Regressor (SVR)** 💪
- **Type:** Kernel-based
- **Description:** Finds optimal hyperplane using support vectors in high-dimensional space
- **Pros:** Effective in high dimensions, versatile with different kernels
- **Cons:** Slower for large datasets, requires feature scaling (automatic)
- **Scaling:** Uses `svr_scaler.pkl`
- **File:** `insurance_model_svr.pkl`

### 9. **K-Nearest Neighbors (KNN)** 👥
- **Type:** Instance-based
- **Description:** Predicts based on average of k nearest training examples
- **Pros:** Simple, no training phase (lazy learning), intuitive
- **Cons:** Slow predictions, sensitive to feature scaling (automatic)
- **Scaling:** Uses `svr_scaler.pkl` (shared with SVR)
- **File:** `insurance_model_knn.pkl`

---

## Using the New Models

### Via API Endpoint
```bash
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "male",
    "bmi": 28.5,
    "kids": 2,
    "smoker": false,
    "location": "northeast",
    "income": "high",
    "employment": "employed",
    "healthScore": 8,
    "exerciseFrequency": 4,
    "education": "bachelor",
    "maritalStatus": "married",
    "yearsInsured": 5,
    "model": "gradient_boosting"  // Specify the model
  }'
```

### Get Available Models
```bash
curl http://localhost:8080/api/models
```

This returns:
```json
{
  "available_models": ["random_forest", "gradient_boosting", "decision_tree", "linear_regression", "ridge", "lasso", "elasticnet", "svr", "knn"],
  "default_model": "random_forest",
  "descriptions": {
    "random_forest": "Random Forest - Ensemble method combining multiple decision trees",
    ...
  }
}
```

### Via Python Script
```bash
python src/main/resources/predict_model.py \
  age=35 gender=male bmi=28.5 kids=2 smoker=no \
  location=northeast income=high employment=employed \
  health_score=8 exercise_frequency=4 education=bachelor \
  marital_status=married years_insured=5 \
  --model gradient_boosting
```

---

## Training the Models

To train all 9 models with your data:

```bash
# Make sure Python virtual environment is activated
python train_model.py
```

This script will:
1. ✅ Generate synthetic training data
2. ✅ Train all 9 models
3. ✅ Evaluate performance metrics (R², RMSE, MAE)
4. ✅ Generate comparison visualizations
5. ✅ Save model files
6. ✅ Save encoders for categorical features
7. ✅ Update `model_config.json` with metrics

### Output Files Generated
- `insurance_model.pkl` - Random Forest (default)
- `insurance_model_gradient_boosting.pkl`
- `insurance_model_decision_tree.pkl`
- `insurance_model_linear_regression.pkl`
- `insurance_model_ridge.pkl`
- `insurance_model_lasso.pkl`
- `insurance_model_elasticnet.pkl`
- `insurance_model_svr.pkl`
- `svr_scaler.pkl` - Feature scaler for SVR/KNN
- `label_encoder_*.pkl` - Encoders for categorical features (6 files)
- `model_config.json` - Configuration with metrics

### Visualizations Generated
- `model_metrics_comparison.png` - Compare R², RMSE, MAE across models
- `actual_vs_predicted.png` - Actual vs predicted values for each model
- `residuals_plot.png` - Residual analysis
- `metrics_table.png` - Detailed metrics table

---

## Model Comparison

After training, check the visualizations in `src/main/resources/static/visualizations/` to compare:

1. **Accuracy (R² Score)** - Higher is better (0-1 scale)
   - Measures how well the model explains variance
   - 0.9+ is excellent, 0.7-0.9 is good, <0.5 is poor

2. **Error Metrics**
   - **RMSE (Root Mean Squared Error)** - Lower is better
   - **MAE (Mean Absolute Error)** - Lower is better
   - Measured in dollars ($)

3. **Train vs Test Performance**
   - Large gap indicates overfitting
   - Balanced performance indicates good generalization

---

## Performance Characteristics

| Model | Speed | Accuracy | Interpretability | Scaling Required |
|-------|-------|----------|------------------|------------------|
| Random Forest | Fast | High | Medium | No |
| Gradient Boosting | Slow | Very High | Low | No |
| Decision Tree | Very Fast | Medium | Very High | No |
| Linear Regression | Very Fast | Low | Very High | No |
| Ridge | Very Fast | Medium | Very High | No |
| Lasso | Very Fast | Medium | Very High | No |
| Elastic Net | Very Fast | Medium | Very High | No |
| SVR | Slow | High | Low | Yes |
| KNN | Slow (pred) | Medium | Medium | Yes |

---

## Recommendations

### For Production Use:
- **Best Overall:** `gradient_boosting` (highest accuracy)
- **Best Speed:** `linear_regression`, `ridge`, `lasso`, `elasticnet`
- **Best Interpretability:** `linear_regression`, `ridge`, `lasso`, `decision_tree`

### For Your Use Case:
1. Train all models with `python train_model.py`
2. Compare metrics in generated visualizations
3. Choose the model with best R² score on test data
4. Use that model as default: Update `default_model` in PredictController.java

---

## Next Steps

1. **Train the models:**
   ```bash
   python train_model.py
   ```

2. **Review the metrics:**
   Check visualizations in `src/main/resources/static/visualizations/`

3. **Test predictions:**
   ```bash
   curl -X POST http://localhost:8080/api/predict \
     -H "Content-Type: application/json" \
     -d '{"age": 35, "gender": "male", ..., "model": "gradient_boosting"}'
   ```

4. **Update frontend (optional):**
   Use the `/api/models` endpoint to dynamically populate model selection dropdown

---

## File Locations

- **Training Script:** `train_model.py`
- **Prediction Script:** `src/main/resources/predict_model.py`
- **Model Files:** `src/main/resources/models/`
- **Config File:** `src/main/resources/models/model_config.json`
- **Controller:** `src/main/java/com/example/demo/controller/PredictController.java`
- **Visualizations:** `src/main/resources/static/visualizations/`

---

**Happy predicting! 🎉**
