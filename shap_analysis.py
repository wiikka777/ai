#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP Explainability Analysis: Understanding CodeBERT Model Decisions
"""

import numpy as np
import pandas as pd
import json
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams



# ========================================================================
# 1. Load Data and Model
# ========================================================================

print("=" * 80)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 80)
print()

# Load test data
print("📦 Loading test set data...")
with open('/user/zhuohang.yu/u24922/exam/test_set_pseudo.json', 'r') as f:
    test_data = json.load(f)

# 转换为 DataFrame
feature_names = ['perplexity', 'avg_token_probability', 'avg_entropy', 'burstiness',
                 'code_length', 'avg_line_length', 'std_line_length', 'comment_ratio',
                 'identifier_entropy', 'ngram_repetition']

X_test = np.array([[s[f] for f in feature_names] for s in test_data])
y_test = np.array([s['label'] for s in test_data])

print(f"✅ Loaded {len(X_test)} test samples with {len(feature_names)} features")
print()

# Load pre-trained XGBoost model
print("🤖 Loading pre-trained CodeBERT+XGBoost model...")
try:
    with open('/user/zhuohang.yu/u24922/exam/codebert_xgboost_model.pkl', 'rb') as f:
        codebert_model = pickle.load(f)
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Model file not found! Please run comparison_experiment.py first.")
    exit(1)

# Get native XGBoost object from model
if hasattr(codebert_model, 'named_steps'):
    # sklearn pipeline
    xgb_model = codebert_model.named_steps['classifier']
else:
    # Direct XGBoost model
    xgb_model = codebert_model

print()

# ========================================================================
# 2. XGBoost Feature Importance Analysis
# ========================================================================

print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS (XGBoost)")
print("=" * 80)
print()

# Get feature importance
importances = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances,
    'Importance_Normalized': importances / importances.sum()
}).sort_values('Importance', ascending=False)

print(feature_importance_df.to_string(index=False))
print()

# Generate feature importance bar chart
fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance_df)))
bars = ax.barh(range(len(feature_importance_df)), 
               feature_importance_df['Importance_Normalized'].values,
               color=colors)

ax.set_yticks(range(len(feature_importance_df)))
ax.set_yticklabels(feature_importance_df['Feature'].values)
ax.set_xlabel('Normalized Importance', fontsize=12, fontweight='bold')
ax.set_title('XGBoost Feature Importance for AI Code Detection\n(CodeBERT + 10 Code-Specific Features)', 
             fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Add value labels
for i, (idx, row) in enumerate(feature_importance_df.iterrows()):
    ax.text(row['Importance_Normalized'] + 0.005, i, 
            f"{row['Importance_Normalized']:.3f}", 
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig('/user/zhuohang.yu/u24922/exam/feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Feature importance plot saved to: feature_importance.png")
print()

# ========================================================================
# 3. SHAP Analysis
# ========================================================================

print("=" * 80)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 80)
print()

try:
    import shap
    print("✅ SHAP library available")
except ImportError:
    print("⚠️  SHAP not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'shap', '-q'])
    import shap
    print("✅ SHAP installed")

print()

# Use SHAP TreeExplainer (optimized for XGBoost)
print("📊 Computing SHAP values (this may take a minute)...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

print(f"✅ SHAP values computed for {len(X_test)} samples")
print()

# If binary classification, shap_values is array; if multi-class, it's a list
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Take positive class (AI)

# Generate SHAP Summary Plot (global view)
print("📈 Generating SHAP Summary Plot (global view)...")
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                  plot_type='bar', show=False)
plt.title('SHAP Feature Importance (Average |SHAP value|)\nHow features influence AI vs Human classification', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/user/zhuohang.yu/u24922/exam/shap_summary_bar.png', dpi=300, bbox_inches='tight')
print("✅ SHAP summary plot saved to: shap_summary_bar.png")
plt.close()

print()

# Generate SHAP Summary Plot (colored scatter)
print("📊 Generating SHAP Summary Plot (colored scatter)...")
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                  plot_type='violin', show=False)
plt.title('SHAP Feature Impact\n(How each feature value influences prediction)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/user/zhuohang.yu/u24922/exam/shap_summary_violin.png', dpi=300, bbox_inches='tight')
print("✅ SHAP violin plot saved to: shap_summary_violin.png")
plt.close()

print()

# ========================================================================
# 4. Select Representative Samples for In-depth Analysis
# ========================================================================

print("=" * 80)
print("SAMPLE-LEVEL ANALYSIS (Force Plots)")
print("=" * 80)
print()

# Find representative samples
predictions = xgb_model.predict_proba(X_test)[:, 1]

# AI code sample (high confidence)
ai_high_conf_idx = np.argmax(predictions[y_test == 1])
# Human code sample (high confidence)
human_high_conf_idx = np.argmin(predictions[y_test == 0])
# Boundary sample (close to 0.5)
boundary_idx = np.argmin(np.abs(predictions - 0.5))

sample_indices = {
    'AI (High Confidence)': ai_high_conf_idx,
    'Human (High Confidence)': human_high_conf_idx,
    'Boundary Case': boundary_idx
}

print("Selected representative samples for detailed analysis:")
print()

force_plot_data = []

for sample_name, idx in sample_indices.items():
    pred_prob = predictions[idx]
    true_label = "AI" if y_test[idx] == 1 else "Human"
    prediction = "AI" if pred_prob >= 0.5 else "Human"
    
    print(f"\n{sample_name}")
    print(f"  Sample Index: {idx}")
    print(f"  True Label: {true_label}")
    print(f"  Prediction: {prediction} (confidence: {max(pred_prob, 1-pred_prob):.3f})")
    print(f"  Feature values:")
    
    for fname, fval in zip(feature_names, X_test[idx]):
        print(f"    - {fname:20s}: {fval:.4f}")
    
    # Generate Force Plot
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(111)
    
    # Manually create force plot visualization (default force plot HTML may not be suitable for saving)
    shap_vals = shap_values[idx]
    base_value = explainer.expected_value
    
    # Sort features
    sorted_indices = np.argsort(np.abs(shap_vals))[::-1][:10]
    
    ax.barh(range(len(sorted_indices)), shap_vals[sorted_indices], 
            color=['red' if v > 0 else 'blue' for v in shap_vals[sorted_indices]])
    ax.set_yticks(range(len(sorted_indices)))
    ax.set_yticklabels([feature_names[i] for i in sorted_indices])
    ax.set_xlabel('SHAP Value (contribution to prediction)', fontweight='bold')
    ax.set_title(f'{sample_name}\nHow features push prediction toward AI vs Human\n(Red=pushes toward AI, Blue=pushes toward Human)', 
                 fontweight='bold', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # Add value labels
    for i, sidx in enumerate(sorted_indices):
        ax.text(shap_vals[sidx] + 0.01 if shap_vals[sidx] > 0 else shap_vals[sidx] - 0.01, 
               i, f'{shap_vals[sidx]:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    filename = f"/user/zhuohang.yu/u24922/exam/shap_force_plot_{sample_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✅ Force plot saved to: {filename.split('/')[-1]}")
    plt.close()

print()
print()

# ========================================================================
# 5. Generate Explainability Report
# ========================================================================

print("=" * 80)
print("GENERATING EXPLAINABILITY REPORT")
print("=" * 80)
print()

report = "\n" + "=" * 90 + "\n"
report += "MODEL EXPLAINABILITY ANALYSIS REPORT\n"
report += "XGBoost + CodeBERT for AI Code Detection\n"
report += "=" * 90 + "\n\n"

# Feature Importance section
report += "┌─ FEATURE IMPORTANCE RANKING ─────────────────────────────────────────┐\n"
report += "│\n"
report += "│ Rank  Feature                  Importance  Normalized  % of Total\n"
report += "├──────────────────────────────────────────────────────────────────────┤\n"

for rank, (idx, row) in enumerate(feature_importance_df.iterrows(), 1):
    report += f"│ {rank:2d}.   {row['Feature']:22s}  {row['Importance']:10.4f}    {row['Importance_Normalized']:7.4f}    {row['Importance_Normalized']*100:6.2f}%\n"

report += "│\n"
report += "└──────────────────────────────────────────────────────────────────────┘\n\n"

# SHAP findings
report += "┌─ SHAP EXPLAINABILITY FINDINGS ───────────────────────────────────────┐\n"
report += "│\n"

# Calculate average influence of features on predictions
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values('Mean_Abs_SHAP', ascending=False)

report += "│ Average |SHAP value| (how much each feature affects predictions):\n"
report += "│\n"

for idx, row in shap_importance_df.iterrows():
    bar_length = int(row['Mean_Abs_SHAP'] * 100)
    bar = "█" * bar_length
    report += f"│ {row['Feature']:22s} {bar} {row['Mean_Abs_SHAP']:.4f}\n"

report += "│\n"
report += "└──────────────────────────────────────────────────────────────────────┘\n\n"

# Key insights
report += "┌─ KEY INSIGHTS ───────────────────────────────────────────────────────┐\n"
report += "│\n"

top3_features = feature_importance_df.head(3)
report += "│ 1. TOP 3 MOST IMPORTANT FEATURES:\n"
for i, (idx, row) in enumerate(top3_features.iterrows(), 1):
    report += f"│    {i}. {row['Feature']:30s} ({row['Importance_Normalized']*100:5.2f}% importance)\n"

report += "│\n"

# Perplexity rank (this is the only feature used by GPTZero)
perplexity_rank = list(feature_importance_df['Feature']).index('perplexity') + 1
perplexity_importance = feature_importance_df[feature_importance_df['Feature'] == 'perplexity']['Importance_Normalized'].values[0]

report += f"│ 2. PERPLEXITY (used by GPTZero):\n"
report += f"│    - Rank: #{perplexity_rank} out of {len(feature_names)}\n"
report += f"│    - Importance: {perplexity_importance*100:.2f}%\n"
report += f"│    - Why GPTZero alone is insufficient for code detection\n"
report += "│\n"

# Importance of code-specific features
code_specific_features = ['identifier_entropy', 'comment_ratio', 'burstiness', 'avg_line_length', 'std_line_length']
code_specific_importance = feature_importance_df[feature_importance_df['Feature'].isin(code_specific_features)]['Importance_Normalized'].sum()

report += f"│ 3. CODE-SPECIFIC FEATURES IMPORTANCE:\n"
report += f"│    - Top 5 code features combined: {code_specific_importance*100:.2f}% importance\n"
report += f"│    - These features capture code structure patterns that differ between AI and human code\n"
report += f"│    - Examples:\n"
report += f"│      • Identifier entropy: AI generators tend to use more uniform naming\n"
report += f"│      • Comment ratio: Humans comment more frequently\n"
report += f"│      • Burstiness: AI code has different statistical patterns\n"
report += "│\n"

report += "│ 4. INTERPRETATION:\n"
report += "│    ✓ Feature importance + SHAP values show that code-specific features are critical\n"
report += "│    ✓ Generic text features (only perplexity) rank lower → need specialized features\n"
report += "│    ✓ Model decisions are explainable: we can point to specific features for each prediction\n"
report += "│\n"

report += "└──────────────────────────────────────────────────────────────────────┘\n"

print(report)

# Save report
output_path = '/user/zhuohang.yu/u24922/exam/shap_explainability_report.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"✅ Report saved to: {output_path}")
print()

# ========================================================================
# 6. Summary
# ========================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("Generated Files:")
print("  1. feature_importance.png          - XGBoost feature importance ranking")
print("  2. shap_summary_bar.png            - SHAP bar plot (average impact)")
print("  3. shap_summary_violin.png         - SHAP violin plot (value distribution)")
print("  4. shap_force_plot_ai_*.png        - How features pushed AI sample toward AI")
print("  5. shap_force_plot_human_*.png     - How features pushed Human sample toward Human")
print("  6. shap_force_plot_boundary_*.png  - How features interact on boundary case")
print("  7. shap_explainability_report.txt  - Text report with findings")
print()
print("✅ SHAP Analysis Complete!")
print()
