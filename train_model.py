"""
Insurance Cost Prediction Model Training Script
Trains a Random Forest model on synthetic insurance data and saves it as a serialized model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style('whitegrid')
sns.set_palette('husl')

# Generate synthetic training data
np.random.seed(42)
n_samples = 500

# Create synthetic dataset with expanded features
data = {
    'age': np.random.randint(18, 80, n_samples),
    'gender': np.random.choice(['male', 'female'], n_samples),
    'bmi': np.random.uniform(15, 50, n_samples),
    'kids': np.random.randint(0, 5, n_samples),
    'smoker': np.random.choice([True, False], n_samples),
    'location': np.random.choice(['northeast', 'southeast', 'southwest', 'northwest'], n_samples),
    # New features for improved prediction
    'income_level': np.random.choice(['low', 'medium', 'high', 'very_high'], n_samples),
    'employment_status': np.random.choice(['employed', 'self_employed', 'unemployed', 'retired'], n_samples),
    'health_score': np.random.randint(1, 11, n_samples),  # 1-10 scale
    'exercise_frequency': np.random.randint(0, 7, n_samples),  # days per week
    'education_level': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], n_samples),
    'marital_status': np.random.choice(['single', 'married', 'divorced', 'widowed'], n_samples),
    'years_insured': np.random.randint(0, 50, n_samples),
}

df = pd.DataFrame(data)

# Generate insurance costs based on features (synthetic ground truth)
# Base cost + age factor + BMI factor + smoker factor + kids factor + location factor + new features
base_cost = 3000
df['cost'] = (
    base_cost +
    (df['age'] * 50) +  # Older age = higher cost
    (df['bmi'] * 100) +  # Higher BMI = higher cost
    (df['kids'] * 500) +  # More kids = higher cost
    (df['smoker'].astype(int) * 5000) +  # Smokers pay much more
    ((df['location'] == 'northeast').astype(int) * 2000) +  # Northeast is more expensive
    ((df['location'] == 'southeast').astype(int) * 1500) +  # Southeast is moderately expensive
    # New feature factors
    ((df['income_level'] == 'low').astype(int) * 1500) +  # Lower income = slightly higher (risk)
    ((df['income_level'] == 'high').astype(int) * -500) +  # Higher income = slight discount
    ((df['income_level'] == 'very_high').astype(int) * -1000) +  # Very high income = more discount
    ((df['employment_status'] == 'unemployed').astype(int) * 2000) +  # Unemployed = higher risk
    ((df['employment_status'] == 'self_employed').astype(int) * 800) +  # Self-employed = slightly higher
    ((df['employment_status'] == 'retired').astype(int) * 1200) +  # Retired = higher risk
    ((11 - df['health_score']) * 300) +  # Lower health score = higher cost
    ((7 - df['exercise_frequency']) * 200) +  # Less exercise = higher cost
    ((df['education_level'] == 'phd').astype(int) * -600) +  # Higher education = discount
    ((df['education_level'] == 'master').astype(int) * -400) +  # Master's = slight discount
    ((df['education_level'] == 'high_school').astype(int) * 300) +  # High school = slight premium
    (df['years_insured'] * 50) +  # Longer insurance history = experienced driver
    np.random.normal(0, 500, n_samples)  # Add some noise
)

# Ensure positive costs
df['cost'] = df['cost'].clip(lower=500)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nCost statistics:")
print(df['cost'].describe())

# Preprocess: encode categorical variables
le_gender = LabelEncoder()
le_location = LabelEncoder()
le_income = LabelEncoder()
le_employment = LabelEncoder()
le_education = LabelEncoder()
le_marital = LabelEncoder()

df['gender_encoded'] = le_gender.fit_transform(df['gender'])
df['location_encoded'] = le_location.fit_transform(df['location'])
df['income_encoded'] = le_income.fit_transform(df['income_level'])
df['employment_encoded'] = le_employment.fit_transform(df['employment_status'])
df['education_encoded'] = le_education.fit_transform(df['education_level'])
df['marital_encoded'] = le_marital.fit_transform(df['marital_status'])
df['smoker_int'] = df['smoker'].astype(int)

# Features and target - includes new features
X = df[['age', 'gender_encoded', 'bmi', 'kids', 'smoker_int', 'location_encoded', 
         'income_encoded', 'employment_encoded', 'health_score', 'exercise_frequency',
         'education_encoded', 'marital_encoded', 'years_insured']]
y = df['cost']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
models = {}
metrics = {}

print("\n" + "="*60)
print("Training Multiple Models")
print("="*60)

# 1. Random Forest
print("\n[1] Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)
rf_train_score = r2_score(y_train, rf_train_pred)
rf_test_score = r2_score(y_test, rf_test_pred)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
models['random_forest'] = rf_model
metrics['random_forest'] = {
    'train_r2': rf_train_score,
    'test_r2': rf_test_score,
    'train_rmse': rf_train_rmse,
    'test_rmse': rf_test_rmse,
    'train_mae': rf_train_mae,
    'test_mae': rf_test_mae,
    'test_predictions': rf_test_pred,
    'train_predictions': rf_train_pred
}
print(f"    Train R²: {rf_train_score:.4f}, Test R²: {rf_test_score:.4f}")
print(f"    Train RMSE: ${rf_train_rmse:.2f}, Test RMSE: ${rf_test_rmse:.2f}")
print(f"    Train MAE: ${rf_train_mae:.2f}, Test MAE: ${rf_test_mae:.2f}")

# 2. Decision Tree
print("\n[2] Training Decision Tree...")
dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)
dt_train_score = r2_score(y_train, dt_train_pred)
dt_test_score = r2_score(y_test, dt_test_pred)
dt_train_rmse = np.sqrt(mean_squared_error(y_train, dt_train_pred))
dt_test_rmse = np.sqrt(mean_squared_error(y_test, dt_test_pred))
dt_train_mae = mean_absolute_error(y_train, dt_train_pred)
dt_test_mae = mean_absolute_error(y_test, dt_test_pred)
models['decision_tree'] = dt_model
metrics['decision_tree'] = {
    'train_r2': dt_train_score,
    'test_r2': dt_test_score,
    'train_rmse': dt_train_rmse,
    'test_rmse': dt_test_rmse,
    'train_mae': dt_train_mae,
    'test_mae': dt_test_mae,
    'test_predictions': dt_test_pred,
    'train_predictions': dt_train_pred
}
print(f"    Train R²: {dt_train_score:.4f}, Test R²: {dt_test_score:.4f}")
print(f"    Train RMSE: ${dt_train_rmse:.2f}, Test RMSE: ${dt_test_rmse:.2f}")
print(f"    Train MAE: ${dt_train_mae:.2f}, Test MAE: ${dt_test_mae:.2f}")

# 3. Linear Regression
print("\n[3] Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)
lr_train_score = r2_score(y_train, lr_train_pred)
lr_test_score = r2_score(y_test, lr_test_pred)
lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
lr_test_mae = mean_absolute_error(y_test, lr_test_pred)
models['linear_regression'] = lr_model
metrics['linear_regression'] = {
    'train_r2': lr_train_score,
    'test_r2': lr_test_score,
    'train_rmse': lr_train_rmse,
    'test_rmse': lr_test_rmse,
    'train_mae': lr_train_mae,
    'test_mae': lr_test_mae,
    'test_predictions': lr_test_pred,
    'train_predictions': lr_train_pred
}
print(f"    Train R²: {lr_train_score:.4f}, Test R²: {lr_test_score:.4f}")
print(f"    Train RMSE: ${lr_train_rmse:.2f}, Test RMSE: ${lr_test_rmse:.2f}")
print(f"    Train MAE: ${lr_train_mae:.2f}, Test MAE: ${lr_test_mae:.2f}")

# 4. Gradient Boosting
print("\n[4] Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
gb_train_pred = gb_model.predict(X_train)
gb_test_pred = gb_model.predict(X_test)
gb_train_score = r2_score(y_train, gb_train_pred)
gb_test_score = r2_score(y_test, gb_test_pred)
gb_train_rmse = np.sqrt(mean_squared_error(y_train, gb_train_pred))
gb_test_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))
gb_train_mae = mean_absolute_error(y_train, gb_train_pred)
gb_test_mae = mean_absolute_error(y_test, gb_test_pred)
models['gradient_boosting'] = gb_model
metrics['gradient_boosting'] = {
    'train_r2': gb_train_score,
    'test_r2': gb_test_score,
    'train_rmse': gb_train_rmse,
    'test_rmse': gb_test_rmse,
    'train_mae': gb_train_mae,
    'test_mae': gb_test_mae,
    'test_predictions': gb_test_pred,
    'train_predictions': gb_train_pred
}
print(f"    Train R²: {gb_train_score:.4f}, Test R²: {gb_test_score:.4f}")
print(f"    Train RMSE: ${gb_train_rmse:.2f}, Test RMSE: ${gb_test_rmse:.2f}")
print(f"    Train MAE: ${gb_train_mae:.2f}, Test MAE: ${gb_test_mae:.2f}")

# 5. Support Vector Regressor (requires feature scaling)
print("\n[5] Training Support Vector Regressor...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
svr_train_pred = svr_model.predict(X_train_scaled)
svr_test_pred = svr_model.predict(X_test_scaled)
svr_train_score = r2_score(y_train, svr_train_pred)
svr_test_score = r2_score(y_test, svr_test_pred)
svr_train_rmse = np.sqrt(mean_squared_error(y_train, svr_train_pred))
svr_test_rmse = np.sqrt(mean_squared_error(y_test, svr_test_pred))
svr_train_mae = mean_absolute_error(y_train, svr_train_pred)
svr_test_mae = mean_absolute_error(y_test, svr_test_pred)
models['svr'] = svr_model
models['svr_scaler'] = scaler  # Save the scaler for SVR
metrics['svr'] = {
    'train_r2': svr_train_score,
    'test_r2': svr_test_score,
    'train_rmse': svr_train_rmse,
    'test_rmse': svr_test_rmse,
    'train_mae': svr_train_mae,
    'test_mae': svr_test_mae,
    'test_predictions': svr_test_pred,
    'train_predictions': svr_train_pred
}
print(f"    Train R²: {svr_train_score:.4f}, Test R²: {svr_test_score:.4f}")
print(f"    Train RMSE: ${svr_train_rmse:.2f}, Test RMSE: ${svr_test_rmse:.2f}")
print(f"    Train MAE: ${svr_train_mae:.2f}, Test MAE: ${svr_test_mae:.2f}")

# 6. K-Nearest Neighbors
print("\n[6] Training K-Nearest Neighbors...")
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)  # Use scaled features
knn_train_pred = knn_model.predict(X_train_scaled)
knn_test_pred = knn_model.predict(X_test_scaled)
knn_train_score = r2_score(y_train, knn_train_pred)
knn_test_score = r2_score(y_test, knn_test_pred)
knn_train_rmse = np.sqrt(mean_squared_error(y_train, knn_train_pred))
knn_test_rmse = np.sqrt(mean_squared_error(y_test, knn_test_pred))
knn_train_mae = mean_absolute_error(y_train, knn_train_pred)
knn_test_mae = mean_absolute_error(y_test, knn_test_pred)
models['knn'] = knn_model
metrics['knn'] = {
    'train_r2': knn_train_score,
    'test_r2': knn_test_score,
    'train_rmse': knn_train_rmse,
    'test_rmse': knn_test_rmse,
    'train_mae': knn_train_mae,
    'test_mae': knn_test_mae,
    'test_predictions': knn_test_pred,
    'train_predictions': knn_train_pred
}
print(f"    Train R²: {knn_train_score:.4f}, Test R²: {knn_test_score:.4f}")
print(f"    Train RMSE: ${knn_train_rmse:.2f}, Test RMSE: ${knn_test_rmse:.2f}")
print(f"    Train MAE: ${knn_train_mae:.2f}, Test MAE: ${knn_test_mae:.2f}")

# 7. Ridge Regression
print("\n[7] Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_train_pred = ridge_model.predict(X_train)
ridge_test_pred = ridge_model.predict(X_test)
ridge_train_score = r2_score(y_train, ridge_train_pred)
ridge_test_score = r2_score(y_test, ridge_test_pred)
ridge_train_rmse = np.sqrt(mean_squared_error(y_train, ridge_train_pred))
ridge_test_rmse = np.sqrt(mean_squared_error(y_test, ridge_test_pred))
ridge_train_mae = mean_absolute_error(y_train, ridge_train_pred)
ridge_test_mae = mean_absolute_error(y_test, ridge_test_pred)
models['ridge'] = ridge_model
metrics['ridge'] = {
    'train_r2': ridge_train_score,
    'test_r2': ridge_test_score,
    'train_rmse': ridge_train_rmse,
    'test_rmse': ridge_test_rmse,
    'train_mae': ridge_train_mae,
    'test_mae': ridge_test_mae,
    'test_predictions': ridge_test_pred,
    'train_predictions': ridge_train_pred
}
print(f"    Train R²: {ridge_train_score:.4f}, Test R²: {ridge_test_score:.4f}")
print(f"    Train RMSE: ${ridge_train_rmse:.2f}, Test RMSE: ${ridge_test_rmse:.2f}")
print(f"    Train MAE: ${ridge_train_mae:.2f}, Test MAE: ${ridge_test_mae:.2f}")

# 8. Lasso Regression
print("\n[8] Training Lasso Regression...")
lasso_model = Lasso(alpha=10.0)
lasso_model.fit(X_train, y_train)
lasso_train_pred = lasso_model.predict(X_train)
lasso_test_pred = lasso_model.predict(X_test)
lasso_train_score = r2_score(y_train, lasso_train_pred)
lasso_test_score = r2_score(y_test, lasso_test_pred)
lasso_train_rmse = np.sqrt(mean_squared_error(y_train, lasso_train_pred))
lasso_test_rmse = np.sqrt(mean_squared_error(y_test, lasso_test_pred))
lasso_train_mae = mean_absolute_error(y_train, lasso_train_pred)
lasso_test_mae = mean_absolute_error(y_test, lasso_test_pred)
models['lasso'] = lasso_model
metrics['lasso'] = {
    'train_r2': lasso_train_score,
    'test_r2': lasso_test_score,
    'train_rmse': lasso_train_rmse,
    'test_rmse': lasso_test_rmse,
    'train_mae': lasso_train_mae,
    'test_mae': lasso_test_mae,
    'test_predictions': lasso_test_pred,
    'train_predictions': lasso_train_pred
}
print(f"    Train R²: {lasso_train_score:.4f}, Test R²: {lasso_test_score:.4f}")
print(f"    Train RMSE: ${lasso_train_rmse:.2f}, Test RMSE: ${lasso_test_rmse:.2f}")
print(f"    Train MAE: ${lasso_train_mae:.2f}, Test MAE: ${lasso_test_mae:.2f}")

# 9. Elastic Net
print("\n[9] Training Elastic Net...")
elasticnet_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
elasticnet_model.fit(X_train, y_train)
elasticnet_train_pred = elasticnet_model.predict(X_train)
elasticnet_test_pred = elasticnet_model.predict(X_test)
elasticnet_train_score = r2_score(y_train, elasticnet_train_pred)
elasticnet_test_score = r2_score(y_test, elasticnet_test_pred)
elasticnet_train_rmse = np.sqrt(mean_squared_error(y_train, elasticnet_train_pred))
elasticnet_test_rmse = np.sqrt(mean_squared_error(y_test, elasticnet_test_pred))
elasticnet_train_mae = mean_absolute_error(y_train, elasticnet_train_pred)
elasticnet_test_mae = mean_absolute_error(y_test, elasticnet_test_pred)
models['elasticnet'] = elasticnet_model
metrics['elasticnet'] = {
    'train_r2': elasticnet_train_score,
    'test_r2': elasticnet_test_score,
    'train_rmse': elasticnet_train_rmse,
    'test_rmse': elasticnet_test_rmse,
    'train_mae': elasticnet_train_mae,
    'test_mae': elasticnet_test_mae,
    'test_predictions': elasticnet_test_pred,
    'train_predictions': elasticnet_train_pred
}
print(f"    Train R²: {elasticnet_train_score:.4f}, Test R²: {elasticnet_test_score:.4f}")
print(f"    Train RMSE: ${elasticnet_train_rmse:.2f}, Test RMSE: ${elasticnet_test_rmse:.2f}")
print(f"    Train MAE: ${elasticnet_train_mae:.2f}, Test MAE: ${elasticnet_test_mae:.2f}")

print("\n" + "="*60)
print("Model Comparison")
print("="*60)
for model_name, model_metrics in metrics.items():
    print(f"\n{model_name.upper()}:")
    print(f"  Test R²: {model_metrics['test_r2']:.4f}")
    print(f"  Test RMSE: ${model_metrics['test_rmse']:.2f}")
    print(f"  Test MAE: ${model_metrics['test_mae']:.2f}")

# Generate Visualizations
print("\n" + "="*60)
print("Generating Visualizations")
print("="*60)

visualization_dir = 'src/main/resources/static/visualizations'
os.makedirs(visualization_dir, exist_ok=True)

# 1. Metrics Comparison Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

model_names = list(metrics.keys())
model_labels = [name.replace('_', ' ').title() for name in model_names]

# R² Score Comparison
r2_scores = [metrics[m]['test_r2'] for m in model_names]
axes[0].bar(model_labels, r2_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
axes[0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
axes[0].set_title('R² Score (Higher is Better)', fontsize=13)
axes[0].set_ylim([0, 1])
for i, v in enumerate(r2_scores):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# RMSE Comparison
rmse_scores = [metrics[m]['test_rmse'] for m in model_names]
axes[1].bar(model_labels, rmse_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
axes[1].set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
axes[1].set_title('RMSE (Lower is Better)', fontsize=13)
for i, v in enumerate(rmse_scores):
    axes[1].text(i, v + 20, f'${v:.2f}', ha='center', fontweight='bold')

# MAE Comparison
mae_scores = [metrics[m]['test_mae'] for m in model_names]
axes[2].bar(model_labels, mae_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
axes[2].set_ylabel('MAE ($)', fontsize=12, fontweight='bold')
axes[2].set_title('MAE (Lower is Better)', fontsize=13)
for i, v in enumerate(mae_scores):
    axes[2].text(i, v + 20, f'${v:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
metrics_chart_path = os.path.join(visualization_dir, 'model_metrics_comparison.png')
plt.savefig(metrics_chart_path, dpi=300, bbox_inches='tight')
print(f"\n[+] Metrics comparison saved: {metrics_chart_path}")
plt.close()

# 2. Actual vs Predicted - All Models in One Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Actual vs Predicted Values (Test Set)', fontsize=16, fontweight='bold')

colors = ['#3498db', '#e74c3c', '#2ecc71']
for idx, (model_name, color) in enumerate(zip(model_names, colors)):
    y_pred = metrics[model_name]['test_predictions']
    
    # Scatter plot
    axes[idx].scatter(y_test, y_pred, alpha=0.6, color=color, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[idx].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    # Labels and title
    axes[idx].set_xlabel('Actual Cost ($)', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Predicted Cost ($)', fontsize=12, fontweight='bold')
    model_label = model_name.replace('_', ' ').title()
    r2 = metrics[model_name]['test_r2']
    axes[idx].set_title(f'{model_label}\nR² = {r2:.4f}', fontsize=13)
    axes[idx].legend(loc='upper left')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
actual_vs_pred_path = os.path.join(visualization_dir, 'actual_vs_predicted.png')
plt.savefig(actual_vs_pred_path, dpi=300, bbox_inches='tight')
print(f"[+] Actual vs Predicted chart saved: {actual_vs_pred_path}")
plt.close()

# 3. Residuals Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Residual Plots (Prediction Errors)', fontsize=16, fontweight='bold')

for idx, (model_name, color) in enumerate(zip(model_names, colors)):
    y_pred = metrics[model_name]['test_predictions']
    residuals = y_test - y_pred
    
    # Residual plot
    axes[idx].scatter(y_pred, residuals, alpha=0.6, color=color, edgecolors='black', linewidth=0.5)
    axes[idx].axhline(y=0, color='black', linestyle='--', lw=2)
    
    # Labels
    axes[idx].set_xlabel('Predicted Cost ($)', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Residuals ($)', fontsize=12, fontweight='bold')
    model_label = model_name.replace('_', ' ').title()
    axes[idx].set_title(f'{model_label}', fontsize=13)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
residuals_path = os.path.join(visualization_dir, 'residuals_plot.png')
plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
print(f"[+] Residuals plot saved: {residuals_path}")
plt.close()

# 4. Detailed Metrics Table
metrics_df = pd.DataFrame({
    'Model': [m.replace('_', ' ').title() for m in model_names],
    'Train R²': [metrics[m]['train_r2'] for m in model_names],
    'Test R²': [metrics[m]['test_r2'] for m in model_names],
    'Train RMSE': [metrics[m]['train_rmse'] for m in model_names],
    'Test RMSE': [metrics[m]['test_rmse'] for m in model_names],
    'Train MAE': [metrics[m]['train_mae'] for m in model_names],
    'Test MAE': [metrics[m]['test_mae'] for m in model_names]
})

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=metrics_df.round(4).values,
                colLabels=metrics_df.columns,
                cellLoc='center',
                loc='center',
                colColours=['#3498db']*len(metrics_df.columns))

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(len(metrics_df.columns)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(metrics_df) + 1):
    for j in range(len(metrics_df.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('white')

plt.title('Detailed Model Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
table_path = os.path.join(visualization_dir, 'metrics_table.png')
plt.savefig(table_path, dpi=300, bbox_inches='tight')
print(f"[+] Metrics table saved: {table_path}")
plt.close()

print(f"\n[+] All visualizations saved to: {visualization_dir}")

# Save all models and encoders
model_dir = 'src/main/resources/models'
os.makedirs(model_dir, exist_ok=True)

print("\n" + "="*60)
print("Saving Models")
print("="*60)

# Save Random Forest (default model for backward compatibility)
rf_path = os.path.join(model_dir, 'insurance_model.pkl')
joblib.dump(models['random_forest'], rf_path)
print(f"\n[+] Random Forest saved to: {rf_path}")

# Save Decision Tree
dt_path = os.path.join(model_dir, 'insurance_model_decision_tree.pkl')
joblib.dump(models['decision_tree'], dt_path)
print(f"[+] Decision Tree saved to: {dt_path}")

# Save Linear Regression
lr_path = os.path.join(model_dir, 'insurance_model_linear_regression.pkl')
joblib.dump(models['linear_regression'], lr_path)
print(f"[+] Linear Regression saved to: {lr_path}")

# Save Gradient Boosting
gb_path = os.path.join(model_dir, 'insurance_model_gradient_boosting.pkl')
joblib.dump(models['gradient_boosting'], gb_path)
print(f"[+] Gradient Boosting saved to: {gb_path}")

# Save Support Vector Regressor
svr_path = os.path.join(model_dir, 'insurance_model_svr.pkl')
joblib.dump(models['svr'], svr_path)
print(f"[+] Support Vector Regressor saved to: {svr_path}")

# Save SVR Scaler
svr_scaler_path = os.path.join(model_dir, 'svr_scaler.pkl')
joblib.dump(models['svr_scaler'], svr_scaler_path)
print(f"[+] SVR Scaler saved to: {svr_scaler_path}")

# Save K-Nearest Neighbors
knn_path = os.path.join(model_dir, 'insurance_model_knn.pkl')
joblib.dump(models['knn'], knn_path)
print(f"[+] K-Nearest Neighbors saved to: {knn_path}")

# Save Ridge Regression
ridge_path = os.path.join(model_dir, 'insurance_model_ridge.pkl')
joblib.dump(models['ridge'], ridge_path)
print(f"[+] Ridge Regression saved to: {ridge_path}")

# Save Lasso Regression
lasso_path = os.path.join(model_dir, 'insurance_model_lasso.pkl')
joblib.dump(models['lasso'], lasso_path)
print(f"[+] Lasso Regression saved to: {lasso_path}")

# Save Elastic Net
elasticnet_path = os.path.join(model_dir, 'insurance_model_elasticnet.pkl')
joblib.dump(models['elasticnet'], elasticnet_path)
print(f"[+] Elastic Net saved to: {elasticnet_path}")

# Save encoders
le_gender_path = os.path.join(model_dir, 'label_encoder_gender.pkl')
le_location_path = os.path.join(model_dir, 'label_encoder_location.pkl')
le_income_path = os.path.join(model_dir, 'label_encoder_income.pkl')
le_employment_path = os.path.join(model_dir, 'label_encoder_employment.pkl')
le_education_path = os.path.join(model_dir, 'label_encoder_education.pkl')
le_marital_path = os.path.join(model_dir, 'label_encoder_marital.pkl')

joblib.dump(le_gender, le_gender_path)
joblib.dump(le_location, le_location_path)
joblib.dump(le_income, le_income_path)
joblib.dump(le_employment, le_employment_path)
joblib.dump(le_education, le_education_path)
joblib.dump(le_marital, le_marital_path)

print(f"\n[+] Gender encoder saved to: {le_gender_path}")
print(f"[+] Location encoder saved to: {le_location_path}")
print(f"[+] Income encoder saved to: {le_income_path}")
print(f"[+] Employment encoder saved to: {le_employment_path}")
print(f"[+] Education encoder saved to: {le_education_path}")
print(f"[+] Marital Status encoder saved to: {le_marital_path}")

# Save model configuration with all metrics
config_path = os.path.join(model_dir, 'model_config.json')

# Convert numpy types to Python native types for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

config = {
    'models': ['random_forest', 'gradient_boosting', 'decision_tree', 'linear_regression', 'ridge', 'lasso', 'elasticnet', 'svr', 'knn'],
    'default_model': 'random_forest',
    'feature_names': ['age', 'gender_encoded', 'bmi', 'kids', 'smoker_int', 'location_encoded',
                      'income_encoded', 'employment_encoded', 'health_score', 'exercise_frequency',
                      'education_encoded', 'marital_encoded', 'years_insured'],
    'encoders': ['gender', 'location', 'income', 'employment', 'education', 'marital'],
    'metrics': {
        'random_forest': convert_to_serializable(metrics['random_forest']),
        'gradient_boosting': convert_to_serializable(metrics['gradient_boosting']),
        'decision_tree': convert_to_serializable(metrics['decision_tree']),
        'linear_regression': convert_to_serializable(metrics['linear_regression']),
        'ridge': convert_to_serializable(metrics['ridge']),
        'lasso': convert_to_serializable(metrics['lasso']),
        'elasticnet': convert_to_serializable(metrics['elasticnet']),
        'svr': convert_to_serializable(metrics['svr']),
        'knn': convert_to_serializable(metrics['knn'])
    },
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test))
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"[+] Configuration saved to: {config_path}")

# Test prediction with sample data from all models
print("\n" + "="*60)
print("Sample Predictions")
print("="*60)
# age, gender_encoded, bmi, kids, smoker_int, location_encoded, income_encoded, employment_encoded,
# health_score, exercise_frequency, education_encoded, marital_encoded, years_insured
sample_input = np.array([[45, 1, 28.5, 2, 0, 2, 1, 0, 7, 4, 1, 1, 10]])
print("\nInput: age=45, gender=female, bmi=28.5, kids=2, smoker=False, location=southeast,")
print("       income=medium, employment=employed, health_score=7, exercise=4 days/week,")
print("       education=bachelor, marital=married, years_insured=10")
print("\nPredictions:")
for model_name, model_obj in models.items():
    prediction = model_obj.predict(sample_input)[0]
    print(f"  {model_name.replace('_', ' ').title()}: ${prediction:.2f}")
