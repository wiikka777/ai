# AI Code Detection Experiment

This repository contains the complete experimental setup and results for AI-generated code detection research.

## Project Overview

Detection of AI-generated code using perplexity-based analysis with CodeBERT language model.

## Dataset

- **Student Code**: 100 samples from real coursework submissions
- **AI-Generated Code**: 60 samples generated using AI tools
- **Programming Language**: C

## Key Features Analyzed

1. **Perplexity**: Measures code predictability
2. **Average Token Probability**: Measures usage of common coding patterns
3. **Burstiness**: Measures variation in code complexity

## Results Summary

- **Optimal Threshold**: 3.70 (perplexity)
- **Accuracy**: 61.25%
- **Precision**: 49.11%
- **Recall**: 91.67%
- **F1 Score**: 0.6395
- **AUC**: 0.7325

## Statistical Significance

- Perplexity: p < 0.001, Cohen's d = -0.83 (large effect)
- Token Probability: p < 0.001, Cohen's d = 0.98 (large effect)
- Burstiness: p = 0.844, Cohen's d = -0.006 (not significant)

## Repository Structure

```
ai-test/
├── method.py                    # Core AICodeAnalyzer class
├── batch_inference.py           # Step 1: Batch analysis script
├── visualization.py             # Step 2: Generate plots
├── threshold_calibration.py     # Step 3: Threshold optimization
├── statistical_testing.py       # Step 4: Statistical tests
├── experiment_results.csv       # Extracted features dataset
├── EXPERIMENT_SUMMARY.txt       # Complete results summary
├── performance_metrics.txt      # Performance evaluation
├── statistical_tests.txt        # Statistical test details
└── figures/                     # Generated visualizations
    ├── ppl_distribution.png
    ├── scatter_ppl_burstiness.png
    ├── feature_comparison.png
    ├── threshold_analysis.png
    ├── roc_curve.png
    ├── confusion_matrix.png
    └── effect_size_analysis.png
```

## Usage

### Run All Experiments

```bash
source ../hpc_gpu_venv/bin/activate
python batch_inference.py      # Step 1: Extract features
python visualization.py         # Step 2: Generate plots
python threshold_calibration.py # Step 3: Find optimal threshold
python statistical_testing.py   # Step 4: Statistical validation
```

### Run Individual Analysis

```python
from method import AICodeAnalyzer

analyzer = AICodeAnalyzer()
result = analyzer.analyze_code(your_code_string)
print(result)
```

## Requirements

- Python 3.11+
- PyTorch
- Transformers (Hugging Face)
- pandas
- numpy
- matplotlib

## Model

- **Base Model**: microsoft/codebert-base-mlm
- **Architecture**: RoBERTa-based masked language model
- **Pre-training**: Trained on 6 programming languages

## Key Findings

1. ✅ AI-generated code shows significantly **lower perplexity** (more predictable)
2. ✅ AI-generated code uses **more high-frequency tokens** (more templated)
3. ❌ **Burstiness** does not distinguish AI from student code effectively

## References

- Bulla et al. (2024): "EX-CODE: A Robust and Explainable Model to Detect AI-Generated Code"
- Xu & Sheng (2024): "variation of code line perplexity"

## Author

Research conducted for thesis on AI-generated code detection in educational settings.

## License

Research and educational use only.



# Analysis Results 
## 1) Subset A FPR Calibration (Human-only, pre-2022)
- Data: 5000 samples, all treated as Human ground truth.
- Detector: DetectGPT.
- Recommended threshold: `0.998900`.
- Achieved FPR: `4.94%` (`247/5000`), meeting the target `<= 5%`.
- Three-class outcome at recommended threshold:
  - Pred AI (False Positive): `247` (4.94%)
  - Pred Human (True Negative): `1545` (30.90%)
  - Pred Unknown: `3208` (64.16%)

Interpretation:
- The threshold successfully controls false positives to the required level.
- A large `Unknown` region remains, indicating many short/insufficient samples for robust detector decision.

## 2) Subset B Cross-Model Consistency (post-2024, no gold labels)
- Matrix source: `scheme2_post2024_consistency_matrix.csv` (5000 aligned samples).
- Rows = GPTZero decision, Columns = DetectGPT decision.

Observed matrix:
- GPTZero AI row: `[13, 107, 92]`
- GPTZero Human row: `[254, 1855, 1085]`
- GPTZero Unknown row: `[7, 113, 1474]`

Key derived indicators:
- Exact agreement (diagonal): `(13 + 1855 + 1474) / 5000 = 66.84%`
- AI/Human direct conflict: `(107 + 254) / 5000 = 7.22%`
- AI consensus (AI,AI): `13/5000 = 0.26%`
- Unknown-involved decisions: `2771/5000 = 55.42%` (from grouped export)

Interpretation:
- The two models mostly agree on Human/Unknown regions.
- High-confidence AI consensus is rare (`13` samples), useful for manual inspection.
- Unknown-driven uncertainty is substantial and should be explicitly reported as a limitation.

## 3) What Characterizes Predicted-AI Code? (post-2024, DetectGPT recommended)
- Comparison set:
  - Predicted AI: `236`
  - Predicted Human: `2113`
  - Unknown excluded: `2651`

Main feature differences (AI minus Human, mean level):
- Code length (`chars`): `+306.75`
- Non-empty lines: `+15.19`
- Comment lines: `+5.20`
- Control keywords: `+1.79`
- Brace max depth: `+0.24`
- English-comment hint probability: `+0.146`

Interpretation:
- Predicted-AI code is longer, more structured, and more comment-heavy on average.
- This aligns with the meeting question about distinguishable AI-style characteristics.


