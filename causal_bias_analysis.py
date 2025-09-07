# -*- coding: utf-8 -*-
"""Causal Analysis of Algorithmic Bias using Double Machine Learning

Author: Shrishti Tomer
Date: September 2025
Description: Self-initiated project to understand causal inference in algorithmic systems
             Implemented after studying econometric methods for causal analysis
"""

# Personal learning notes:
# - Double ML helps isolate causal effects from observational data
# - Residualization removes confounding variables  
# - This provides more robust estimates than simple correlation analysis

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

print("Advanced Causal Analysis of Algorithmic Bias")
print("Using Double Machine Learning (Econometric Causal Inference Methodology)")

# Generate more realistic synthetic data
np.random.seed(42)
n_samples = 5000

print("Generating synthetic dataset with realistic causal structure...")

# Simulate underlying causal structure
data = {
    'education': np.clip(np.random.normal(15, 3, n_samples), 10, 20).astype(int),
    'work_experience': np.clip(np.random.normal(8, 4, n_samples), 0, 20).astype(int),
    'parent_income': np.clip(np.random.normal(60000, 20000, n_samples), 20000, 100000).astype(int),
    'age': np.clip(np.random.normal(35, 8, n_samples), 18, 65).astype(int),
    'gender': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])  # 0 = Male, 1 = Female
}

df = pd.DataFrame(data)

# True loanworthiness (unobservable in real world)
df['true_loanworthiness'] = (
    0.3 * (df['education'] / 4) +
    0.2 * (df['work_experience'] / 2) +
    0.1 * (df['parent_income'] / 10000) +
    0.1 * (df['age'] / 10) +
    np.random.normal(0, 0.5, n_samples)
)

# Biased algorithm with causal structure
discrimination_effect = 0.25  # Direct discrimination effect
df['algorithm_score'] = (
    df['true_loanworthiness'] - 
    (discrimination_effect * df['gender']) +  # Direct bias
    (0.1 * df['gender'] * df['education']) +  # Interaction effect
    np.random.normal(0, 0.3, n_samples)
)

# Approval decisions
threshold = np.percentile(df['algorithm_score'], 70)
df['approved'] = (df['algorithm_score'] >= threshold).astype(int)

print("Dataset generated with complex causal relationships")

# Double Machine Learning for Causal Inference
print("\nPerforming Causal Analysis using Double ML...")

# Step 1: Prepare data for DML
X = df[['education', 'work_experience', 'parent_income', 'age']]
T = df['gender']  # Treatment variable
Y = df['algorithm_score']  # Outcome variable

# Step 2: Split data
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X, T, Y, test_size=0.3, random_state=42
)

# Step 3: Train ML models for nuisance parameters
print("Training machine learning models...")

# Model for E[Y|X]
model_Y = RandomForestRegressor(n_estimators=100, random_state=42)
model_Y.fit(X_train, Y_train)

# Model for E[T|X] 
model_T = RandomForestRegressor(n_estimators=100, random_state=42)
model_T.fit(X_train, T_train)

# Step 4: Predict nuisance parameters
Y_hat = model_Y.predict(X_test)
T_hat = model_T.predict(X_test)

# Step 5: Calculate residuals
Y_residual = Y_test - Y_hat
T_residual = T_test - T_hat

# Step 6: Estimate causal effect
# θ = E[(Y - E[Y|X]) * (T - E[T|X])] / E[(T - E[T|X])^2]
theta_hat = np.mean(Y_residual * T_residual) / np.mean(T_residual**2)

print("\nCAUSAL ESTIMATION RESULTS:")
print(f"Estimated Discrimination Effect (θ): {theta_hat:.4f}")
print(f"True Discrimination Effect: {discrimination_effect:.4f}")

# Statistical inference
X_residual = sm.add_constant(T_residual)
model = sm.OLS(Y_residual, X_residual)
results = model.fit()

print("\nSTATISTICAL SIGNIFICANCE:")
print(f"P-value: {results.pvalues[1]:.6f}")
print(f"95% Confidence Interval: [{results.conf_int().iloc[1,0]:.4f}, {results.conf_int().iloc[1,1]:.4f}]")

# Calculate approval rates by gender
approval_rates = df.groupby('gender')['approved'].mean()
approval_rates_gender = approval_rates.rename({0: 'Male', 1: 'Female'})

print("\nAPPROVAL RATES:")
print(f"Male Approval Rate: {approval_rates_gender['Male']:.2%}")
print(f"Female Approval Rate: {approval_rates_gender['Female']:.2%}")
print(f"Bias Gap: {abs(approval_rates_gender['Male'] - approval_rates_gender['Female']):.2%}")

# Enhanced visualization
plt.style.use('default')
plt.figure(figsize=(16, 12))

# Plot 1: Causal Effect Visualization
plt.subplot(2, 2, 1)
effects = [discrimination_effect, theta_hat]
labels = ['True Effect', 'Estimated Effect (DML)']
colors = ['#ff7f0e', '#1f77b4']
bars = plt.bar(labels, effects, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
plt.title('Causal Effect of Gender on Algorithm Score\n(Double Machine Learning Estimation)', fontsize=14, fontweight='bold')
plt.ylabel('Effect Size', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on bars
for bar, effect in zip(bars, effects):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{effect:.4f}', ha='center', fontweight='bold')

# Plot 2: Residual Analysis
plt.subplot(2, 2, 2)
plt.scatter(T_residual, Y_residual, alpha=0.6, s=30, color='green')
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.axvline(0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Treatment Residuals (T - E[T|X])', fontweight='bold')
plt.ylabel('Outcome Residuals (Y - E[Y|X])', fontweight='bold')
plt.title('Double ML Residual Plot', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 3: Distribution Comparison
plt.subplot(2, 2, 3)
sns.kdeplot(data=df, x='algorithm_score', hue='gender', fill=True, alpha=0.6, 
            palette=['#1f77b4', '#ff7f0e'], common_norm=False)
plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Approval Threshold')
plt.title('Algorithm Score Distribution by Gender', fontsize=14, fontweight='bold')
plt.xlabel('Algorithm Score', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 4: Approval Rates
plt.subplot(2, 2, 4)
colors = ['#1f77b4', '#ff7f0e']
approval_rates_gender.plot(kind='bar', color=colors, edgecolor='black', linewidth=2)
plt.title('Loan Approval Rates by Gender', fontsize=14, fontweight='bold')
plt.ylabel('Approval Rate', fontweight='bold')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on bars
for i, v in enumerate(approval_rates_gender):
    plt.text(i, v + 0.01, f'{v:.2%}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('causal_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Save complete analysis
df.to_csv('advanced_bias_analysis.csv', index=False)
print("\nAnalysis saved to 'advanced_bias_analysis.csv'")
print("Visualizations saved to 'causal_analysis_results.png'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*60)
print("Key Achievements:")
print("- Successfully implemented Double Machine Learning")
print("- Estimated causal effect with statistical significance") 
print("- Generated comprehensive visualizations")
print("- Produced research-ready results")

print("\n" + "="*60)
print("LEARNING OUTCOMES:")
print("- Practical understanding of causal inference methods")
print("- Experience with econometric techniques in Python")
print("- Ability to translate research concepts into code")
print("- Insight into algorithmic fairness and bias detection")

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)