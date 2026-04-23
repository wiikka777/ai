#!/bin/bash
# Master script to run all 4 experimental steps

echo "================================================================================"
echo "AI CODE DETECTION - COMPLETE EXPERIMENTAL PIPELINE"
echo "================================================================================"
echo ""

# Activate virtual environment
source ../hpc_gpu_venv/bin/activate

# Step 1: Batch Inference
echo "Step 1/4: Running batch inference..."
python batch_inference.py
if [ $? -ne 0 ]; then
    echo "ERROR: Batch inference failed"
    exit 1
fi
echo ""

# Step 2: Visualization
echo "Step 2/4: Generating visualizations..."
python visualization.py
if [ $? -ne 0 ]; then
    echo "ERROR: Visualization failed"
    exit 1
fi
echo ""

# Step 3: Threshold Calibration
echo "Step 3/4: Performing threshold calibration..."
python threshold_calibration.py
if [ $? -ne 0 ]; then
    echo "ERROR: Threshold calibration failed"
    exit 1
fi
echo ""

# Step 4: Statistical Testing
echo "Step 4/4: Running statistical tests..."
python statistical_testing.py
if [ $? -ne 0 ]; then
    echo "ERROR: Statistical testing failed"
    exit 1
fi
echo ""

echo "================================================================================"
echo "ALL STEPS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
echo ""
echo "Generated Files:"
echo "  Data:"
echo "    - experiment_results.csv"
echo ""
echo "  Visualizations:"
echo "    - ppl_distribution.png"
echo "    - scatter_ppl_burstiness.png"
echo "    - feature_comparison.png"
echo "    - threshold_analysis.png"
echo "    - roc_curve.png"
echo "    - confusion_matrix.png"
echo "    - effect_size_analysis.png"
echo ""
echo "  Reports:"
echo "    - performance_metrics.txt"
echo "    - statistical_tests.txt"
echo "    - EXPERIMENT_SUMMARY.txt"
echo ""
echo "================================================================================"
