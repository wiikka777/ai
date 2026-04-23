"""
Step 3: Threshold Calibration and Performance Evaluation (XGBoost Version)
Input: experiment_results.csv
Output:
  - threshold_analysis.png - F1 Score vs Threshold curve
  - threshold_analysis_fusion.png - Fusion model (XGBoost) threshold curve
  - roc_curve.png - ROC curve (single feature)
  - roc_curve_fusion.png - Fusion model ROC curve (XGBoost)
  - confusion_matrix.png - Confusion matrix (single feature)
  - confusion_matrix_fusion.png - Confusion matrix (XGBoost fusion model)
  - performance_metrics.txt - Detailed performance metrics
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve, 
    auc, 
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

print("=" * 80)
print("Step 3: Threshold Calibration and Performance Evaluation (Sklearn Version)")
print("=" * 80)

# 1. Load data
print("\n[1] Loading experiment data...")

df = pd.read_csv("experiment_results.csv")


print(f"✓ Loaded {len(df)} samples")

# Prepare labels (1 for AI, 0 for Student)
y_true = (df['label'] == 'AI').astype(int)

# 2. Find optimal threshold for Perplexity
print("\n[2] Finding optimal threshold for Perplexity...")
thresholds = np.arange(1.0, 10.0, 0.1)

# Used to store metrics at different thresholds
metrics_history = {
    'f1': [],
    'acc': [],
    'prec': [],
    'rec': []
}

for threshold in thresholds:
    # Predict: if PPL < threshold, then predict 'AI' (label=1)
    y_pred = (df['perplexity'] < threshold).astype(int)
    
    metrics_history['f1'].append(f1_score(y_true, y_pred, zero_division=0))
    metrics_history['acc'].append(accuracy_score(y_true, y_pred))
    metrics_history['prec'].append(precision_score(y_true, y_pred, zero_division=0))
    metrics_history['rec'].append(recall_score(y_true, y_pred, zero_division=0))

# Find best threshold based on F1 Score
best_idx = np.argmax(metrics_history['f1'])
best_threshold = thresholds[best_idx]
best_f1 = metrics_history['f1'][best_idx]
best_accuracy = metrics_history['acc'][best_idx]
best_precision = metrics_history['prec'][best_idx]
best_recall = metrics_history['rec'][best_idx]

print(f"✓ Optimal Threshold: {best_threshold:.2f}")
print(f"  - F1 Score: {best_f1:.4f}")
print(f"  - Accuracy: {best_accuracy:.4f}")
print(f"  - Precision: {best_precision:.4f}")
print(f"  - Recall: {best_recall:.4f}")

# 3. Multi-feature fusion with XGBoost
print("\n[3] Training multi-feature fusion model (XGBoost)...")
# Prefer all strong features
all_features = ['perplexity', 'avg_token_probability', 'avg_entropy', 'burstiness',
                'code_length', 'avg_line_length', 'std_line_length', 'comment_ratio', 
                'identifier_entropy', 'ngram_repetition']
feature_cols = [col for col in all_features if col in df.columns]

if len(feature_cols) < 2:
    print(f"  ⚠ Warning: Only {len(feature_cols)} feature(s) available.")
    if len(feature_cols) == 0:
        feature_cols = ['perplexity']
else:
    print(f"  ✓ Using {len(feature_cols)} features: {feature_cols}")

X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train XGBoost with 5-fold cross-validation
xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=(y_true == 0).sum() / (y_true == 1).sum(),  # Handle class imbalance
    verbosity=0
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_proba_fusion = cross_val_predict(xgb_clf, X_scaled, y_true, cv=cv, method='predict_proba')[:, 1]

fusion_auc = roc_auc_score(y_true, y_proba_fusion)
print(f"  ✓ XGBoost Model ROC AUC (5-fold CV): {fusion_auc:.4f}")

# Find optimal threshold for fusion model
fusion_thresholds = np.linspace(0.0, 1.0, 201)
fusion_metrics = {
    'f1': [],
    'acc': [],
    'prec': [],
    'rec': []
}

for threshold in fusion_thresholds:
    y_pred_fusion = (y_proba_fusion >= threshold).astype(int)
    fusion_metrics['f1'].append(f1_score(y_true, y_pred_fusion, zero_division=0))
    fusion_metrics['acc'].append(accuracy_score(y_true, y_pred_fusion))
    fusion_metrics['prec'].append(precision_score(y_true, y_pred_fusion, zero_division=0))
    fusion_metrics['rec'].append(recall_score(y_true, y_pred_fusion, zero_division=0))

fusion_best_idx = np.argmax(fusion_metrics['f1'])
fusion_best_threshold = fusion_thresholds[fusion_best_idx]
fusion_best_f1 = fusion_metrics['f1'][fusion_best_idx]
fusion_best_acc = fusion_metrics['acc'][fusion_best_idx]
fusion_best_prec = fusion_metrics['prec'][fusion_best_idx]
fusion_best_rec = fusion_metrics['rec'][fusion_best_idx]

print(f"  ✓ Fusion Optimal Threshold: {fusion_best_threshold:.4f}")
print(f"    - F1 Score: {fusion_best_f1:.4f}")
print(f"    - Accuracy: {fusion_best_acc:.4f}")
print(f"    - Precision: {fusion_best_prec:.4f}")
print(f"    - Recall: {fusion_best_rec:.4f}")

# 4. Plot F1 Score vs Threshold (Single Feature)
print("\n[4] Plotting F1 Score vs Threshold (Single Feature)...")
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(thresholds, metrics_history['f1'], 'b-', linewidth=2, label='F1 Score')
ax.plot(thresholds, metrics_history['acc'], 'g--', linewidth=2, label='Accuracy')
ax.plot(thresholds, metrics_history['prec'], 'r:', linewidth=2, label='Precision')
ax.plot(thresholds, metrics_history['rec'], 'm-.', linewidth=2, label='Recall')

# Mark the best threshold
ax.axvline(best_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.scatter([best_threshold], [best_f1], color='red', s=200, zorder=5, 
           label=f'Best Threshold={best_threshold:.2f}')
ax.text(best_threshold + 0.2, best_f1, f'F1={best_f1:.4f}', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.set_xlabel("Perplexity Threshold", fontsize=12, fontweight='bold')
ax.set_ylabel("Score", fontsize=12, fontweight='bold')
ax.set_title("Performance Metrics vs Perplexity Threshold", fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig("threshold_analysis.png", dpi=300, bbox_inches='tight')
print("✓ Saved: threshold_analysis.png")
plt.close()

# Plot fusion model threshold analysis
print("\n[5] Plotting XGBoost Model F1 Score vs Threshold...")
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(fusion_thresholds, fusion_metrics['f1'], 'b-', linewidth=2, label='F1 Score')
ax.plot(fusion_thresholds, fusion_metrics['acc'], 'g--', linewidth=2, label='Accuracy')
ax.plot(fusion_thresholds, fusion_metrics['prec'], 'r:', linewidth=2, label='Precision')
ax.plot(fusion_thresholds, fusion_metrics['rec'], 'm-.', linewidth=2, label='Recall')

ax.axvline(fusion_best_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.scatter([fusion_best_threshold], [fusion_best_f1], color='red', s=200, zorder=5,
           label=f'Best Threshold={fusion_best_threshold:.2f}')
ax.text(fusion_best_threshold + 0.02, fusion_best_f1, f'F1={fusion_best_f1:.4f}', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.set_xlabel("XGBoost Probability Threshold", fontsize=12, fontweight='bold')
ax.set_ylabel("Score", fontsize=12, fontweight='bold')
ax.set_title("Performance Metrics vs XGBoost Probability Threshold", fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig("threshold_analysis_fusion.png", dpi=300, bbox_inches='tight')
print("✓ Saved: threshold_analysis_fusion.png")
plt.close()
# 6. Generate ROC Curve (Single Feature)
print("\n[6] Generating ROC curve (Single Feature)...")
# The larger the expected score of sklearn's roc_curve, the more it resembles a Positive (AI).
# But our logic is that the smaller the PPL, the more it resembles AI.
# Therefore, we take a negative number: - The larger the PPL is, the smaller the PPL is, which means it is more like AI.
y_scores = -df['perplexity'] 

#Calculate ROC and AUC using sklearn 
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve for AI Code Detection (Single Feature)', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: roc_curve.png (AUC = {roc_auc:.4f})")
plt.close()

# 7. Generate ROC Curve for XGBoost Model
print("\n[7] Generating ROC curve (XGBoost Model)...")
fpr_fusion, tpr_fusion, _ = roc_curve(y_true, y_proba_fusion)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr_fusion, tpr_fusion, color='darkorange', lw=2, label=f'ROC curve (AUC = {fusion_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve for AI Code Detection (XGBoost Model)', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("roc_curve_fusion.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: roc_curve_fusion.png (AUC = {fusion_auc:.4f})")
plt.close()

# 8. Generate confusion matrix at best threshold (Single Feature)
print("\n[8] Confusion matrix at optimal threshold (Single Feature)...")
y_pred_best = (df['perplexity'] < best_threshold).astype(int)

# Calculate the confusion matrix using sklearn 
cm = confusion_matrix(y_true, y_pred_best)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

classes = ['Student', 'AI']
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticklabels(classes, fontsize=11)

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=20, fontweight='bold')

ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_title(f'Confusion Matrix (Perplexity Threshold = {best_threshold:.2f})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")
plt.close()

# 9. Generate confusion matrix for XGBoost Model
print("\n[9] Confusion matrix at optimal threshold (XGBoost Model)...")
y_pred_fusion = (y_proba_fusion >= fusion_best_threshold).astype(int)
cm_fusion = confusion_matrix(y_true, y_pred_fusion)
tn_f, fp_f, fn_f, tp_f = cm_fusion.ravel()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm_fusion, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

classes = ['Student', 'AI']
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticklabels(classes, fontsize=11)

thresh = cm_fusion.max() / 2.
for i in range(cm_fusion.shape[0]):
    for j in range(cm_fusion.shape[1]):
        ax.text(j, i, format(cm_fusion[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm_fusion[i, j] > thresh else "black",
                fontsize=20, fontweight='bold')

ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_title(f'Confusion Matrix (XGBoost Model, Threshold = {fusion_best_threshold:.2f})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_matrix_fusion.png", dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix_fusion.png")
plt.close()

# 10. Save detailed performance report
print("\n[10] Saving performance metrics report...")
with open("performance_metrics.txt", "w") as f:
    f.write("=" * 80 + "\n")
    f.write("AI CODE DETECTION - PERFORMANCE EVALUATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Dataset Summary:\n")
    f.write(f"  - Total samples: {len(df)}\n")
    f.write(f"  - AI samples: {sum(y_true)}\n")
    f.write(f"  - Student samples: {len(y_true) - sum(y_true)}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("MODEL 1: SINGLE FEATURE (PERPLEXITY)\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Optimal Perplexity Threshold: {best_threshold:.4f}\n\n")
    
    f.write(f"Performance Metrics at Optimal Threshold:\n")
    f.write(f"  - Accuracy:  {best_accuracy:.4f} ({best_accuracy*100:.2f}%)\n")
    f.write(f"  - Precision: {best_precision:.4f} ({best_precision*100:.2f}%)\n")
    f.write(f"  - Recall:    {best_recall:.4f} ({best_recall*100:.2f}%)\n")
    f.write(f"  - F1 Score:  {best_f1:.4f}\n\n")
    
    f.write(f"ROC Analysis:\n")
    f.write(f"  - AUC Score: {roc_auc:.4f}\n\n")
    
    f.write(f"Confusion Matrix:\n")
    f.write(f"                Predicted Student  Predicted AI\n")
    f.write(f"True Student    {tn:^17}  {fp:^12}\n")
    f.write(f"True AI         {fn:^17}  {tp:^12}\n\n")
    
    f.write(f"Interpretation:\n")
    f.write(f"  - True Negatives (TN):  {tn} (correctly identified Student code)\n")
    f.write(f"  - False Positives (FP): {fp} (Student code misclassified as AI)\n")
    f.write(f"  - False Negatives (FN): {fn} (AI code misclassified as Student)\n")
    f.write(f"  - True Positives (TP):  {tp} (correctly identified AI code)\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("MODEL 2: XGBOOST FUSION MODEL\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Features Used ({len(feature_cols)}):\n")
    for i, feat in enumerate(feature_cols, 1):
        f.write(f"  {i}. {feat}\n")
    f.write(f"\n")
    
    f.write(f"Optimal XGBoost Probability Threshold: {fusion_best_threshold:.4f}\n\n")
    
    f.write(f"Performance Metrics at Optimal Threshold:\n")
    f.write(f"  - Accuracy:  {fusion_best_acc:.4f} ({fusion_best_acc*100:.2f}%)\n")
    f.write(f"  - Precision: {fusion_best_prec:.4f} ({fusion_best_prec*100:.2f}%)\n")
    f.write(f"  - Recall:    {fusion_best_rec:.4f} ({fusion_best_rec*100:.2f}%)\n")
    f.write(f"  - F1 Score:  {fusion_best_f1:.4f}\n\n")
    
    f.write(f"ROC Analysis:\n")
    f.write(f"  - AUC Score: {fusion_auc:.4f}\n\n")
    
    f.write(f"Confusion Matrix:\n")
    f.write(f"                Predicted Student  Predicted AI\n")
    f.write(f"True Student    {tn_f:^17}  {fp_f:^12}\n")
    f.write(f"True AI         {fn_f:^17}  {tp_f:^12}\n\n")
    
    f.write(f"Interpretation:\n")
    f.write(f"  - True Negatives (TN):  {tn_f} (correctly identified Student code)\n")
    f.write(f"  - False Positives (FP): {fp_f} (Student code misclassified as AI)\n")
    f.write(f"  - False Negatives (FN): {fn_f} (AI code misclassified as Student)\n")
    f.write(f"  - True Positives (TP):  {tp_f} (correctly identified AI code)\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("IMPROVEMENT ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"AUC Improvement:      {fusion_auc - roc_auc:+.4f} ({(fusion_auc - roc_auc)/roc_auc*100:+.2f}%)\n")
    f.write(f"F1 Score Improvement: {fusion_best_f1 - best_f1:+.4f} ({(fusion_best_f1 - best_f1)/best_f1*100:+.2f}%)\n")
    f.write(f"Accuracy Improvement: {fusion_best_acc - best_accuracy:+.4f} ({(fusion_best_acc - best_accuracy)/best_accuracy*100:+.2f}%)\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("FUSION MODEL (Logistic Regression with Multiple Features)\n")
    f.write("-" * 80 + "\n\n")
    
    f.write(f"Features Used: {', '.join(feature_cols)}\n")
    f.write(f"Fusion Model ROC AUC (5-fold CV): {fusion_auc:.4f}\n\n")
    
    f.write(f"Optimal Fusion Probability Threshold: {fusion_best_threshold:.4f}\n\n")
    
    f.write(f"Performance Metrics at Optimal Threshold (Fusion):\n")
    f.write(f"  - Accuracy:  {fusion_best_acc:.4f} ({fusion_best_acc*100:.2f}%)\n")
    f.write(f"  - Precision: {fusion_best_prec:.4f} ({fusion_best_prec*100:.2f}%)\n")
    f.write(f"  - Recall:    {fusion_best_rec:.4f} ({fusion_best_rec*100:.2f}%)\n")
    f.write(f"  - F1 Score:  {fusion_best_f1:.4f}\n\n")
    
    f.write(f"Confusion Matrix (Fusion):\n")
    f.write(f"                Predicted Student  Predicted AI\n")
    f.write(f"True Student    {tn_f:^17}  {fp_f:^12}\n")
    f.write(f"True AI         {fn_f:^17}  {tp_f:^12}\n\n")
    
    f.write(f"Interpretation (Fusion):\n")
    f.write(f"  - True Negatives (TN):  {tn_f} (correctly identified Student code)\n")
    f.write(f"  - False Positives (FP): {fp_f} (Student code misclassified as AI)\n")
    f.write(f"  - False Negatives (FN): {fn_f} (AI code misclassified as Student)\n")
    f.write(f"  - True Positives (TP):  {tp_f} (correctly identified AI code)\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("COMPARISON\n")
    f.write("-" * 80 + "\n")
    f.write(f"Single Feature (Perplexity) vs Fusion Model:\n")
    f.write(f"  Metric         | Perplexity | Fusion Model | Improvement\n")
    f.write(f"  Accuracy       | {best_accuracy:.4f}     | {fusion_best_acc:.4f}      | {fusion_best_acc - best_accuracy:+.4f}\n")
    f.write(f"  Precision      | {best_precision:.4f}     | {fusion_best_prec:.4f}      | {fusion_best_prec - best_precision:+.4f}\n")
    f.write(f"  Recall         | {best_recall:.4f}     | {fusion_best_rec:.4f}      | {fusion_best_rec - best_recall:+.4f}\n")
    f.write(f"  F1 Score       | {best_f1:.4f}     | {fusion_best_f1:.4f}      | {fusion_best_f1 - best_f1:+.4f}\n")
    f.write(f"  AUC            | {roc_auc:.4f}     | {fusion_auc:.4f}      | {fusion_auc - roc_auc:+.4f}\n")
    
    f.write("=" * 80 + "\n")

print("✓ Saved: performance_metrics.txt")

# 7. Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n[Single Feature Model] Perplexity Threshold: {best_threshold:.4f}")
print(f"  → When perplexity < {best_threshold:.2f}, classify as AI-generated")
print(f"  → When perplexity >= {best_threshold:.2f}, classify as Student-written")
print(f"\n  Performance at this threshold:")
print(f"    • Accuracy:  {best_accuracy*100:.2f}%")
print(f"    • Precision: {best_precision*100:.2f}%")
print(f"    • Recall:    {best_recall*100:.2f}%")
print(f"    • F1 Score:  {best_f1:.4f}")
print(f"    • AUC:       {roc_auc:.4f}")

print(f"\n[Fusion Model] Logistic Regression")
print(f"  Features: {feature_cols}")
print(f"  Best Threshold: {fusion_best_threshold:.4f}")
print(f"\n  Performance at this threshold:")
print(f"    • Accuracy:  {fusion_best_acc*100:.2f}%")
print(f"    • Precision: {fusion_best_prec*100:.2f}%")
print(f"    • Recall:    {fusion_best_rec*100:.2f}%")
print(f"    • F1 Score:  {fusion_best_f1:.4f}")
print(f"    • AUC:       {fusion_auc:.4f}")

print(f"\n[Improvement]")
improvement_acc = fusion_best_acc - best_accuracy
improvement_f1 = fusion_best_f1 - best_f1
improvement_auc = fusion_auc - roc_auc
print(f"  • Accuracy improvement: {improvement_acc:+.4f} ({improvement_acc*100:+.2f}%)")
print(f"  • F1 improvement:       {improvement_f1:+.4f}")
print(f"  • AUC improvement:      {improvement_auc:+.4f}")

print("\n" + "=" * 80)
print("✓ Step 3 Completed!")
print("=" * 80)