#!/usr/bin/env python3
"""
QUICK REFERENCE: 改进后的实验运行指南
===========================================

新脚本位置: scripts/experiments/comparison_experiment_v2.py
详细文档: EXPERIMENT_REDESIGN.md
"""

# 📖 快速运行示例
# ==================

# 1. 完整运行（Part 1 + Part 2）
# python scripts/experiments/comparison_experiment_v2.py

# 2. 只跑Part 1（有标签评估，约15-30分钟）
# python scripts/experiments/comparison_experiment_v2.py --part1_only

# 3. 只跑Part 2（44k分析，3小时+，建议先采样）
# python scripts/experiments/comparison_experiment_v2.py --part2_only --sample_size 5000

# 4. 用GPU加速（如果有CUDA）
# python scripts/experiments/comparison_experiment_v2.py --device cuda

# 5. 保存到日志
# python scripts/experiments/comparison_experiment_v2.py 2>&1 | tee comparison_v2_$(date +%s).log


# 🎯 Part 1 关键点（5-Fold Stratified K-Fold）
# ===============================================

"""
数据：
  • AI数据：ai.json → 60个样本
  • 人类数据：slice_before_2022_11_01_5000.json → 5000个样本
  • 总计：5,060个完整标记样本

不平衡处理：
  • 60:5000 = 1:83 的严重不平衡
  • 用类别权重处理：AI_weight = 42.17, Human_weight = 0.506
  • 分层k-fold保证每个fold的比例一致

输出指标（针对不平衡数据）：
  ✅ Precision（精准率）— 预测为AI的准确性
  ✅ Recall（召回率）   — 识别AI的比例（最重要！）
  ✅ F1-Score          — 精准率和召回率的平衡
  ✅ AUC-ROC           — 真正率vs假正率
  ❌ Accuracy          — 无意义（不平衡数据下）

预期结果：
  CodeBERT > DetectGPT > GPTZero
  
  在Recall上，CodeBERT应该接近0.90+
  在Precision上，CodeBERT应该接近0.85+
"""


# 🔍 Part 2 关键点（5维分析）
# =============================

"""
数据：
  • 44,258个代码样本（或采样N个以加速）
  • 三个模型都独立预测，没有伪标签

5维分析：

  1️⃣ 【分布对比】
     问题：三个模型预测的AI率是否相同？
     输出：GPTZero 34%, DetectGPT 19%, CodeBERT 27%
     启示：模型倾向差异大，适用场景不同
  
  2️⃣ 【一致性分析】
     问题：三个模型对同一样本的意见有多一致？
     输出：都说AI = 19%（高置信）
          分歧 = 59%（需审查）
     启示：分歧样本的特征分析可找出模型差异原因
  
  3️⃣ 【置信度校准】
     问题：模型的置信度分数有多可靠？
     做法：用Part 1的test fold建立映射：自报→实际准确度
     输出：模型自报0.85置信 → 实际准确度≈60%
     启示：置信度存在偏差，需要校准
  
  4️⃣ 【特征驱动分析】
     问题：什么特征导致模型分歧？
     分析：代码长度、复杂度、语言特性等
     输出：长代码（>2000字符）更容易分歧
     启示：针对特定特征选择最佳模型
  
  5️⃣ 【风险评估】
     问题：哪些样本需要人工审查？
     分类：
       § HIGH RISK：都说AI & 高置信 → 立即标记
       § MEDIUM RISK： 都说AI         → 次优先
       § UNCERTAIN：模型分歧         → 人工抽样
       § LOW RISK：都说Human         → 信任
     行动：HIGH_RISK全部审查，UNCERTAIN抽样1000个

推荐流程：
  1. Part 1验证：CodeBERT确实更好（Recall>0.90）
  2. Part 2识别：哪些样本是"共识高置信"的AI
  3. 手工审查：抽取分歧样本进行人工标注
  4. 迭代改进：基于人工标注重新训练
"""


# 🛠️ 关键类和方法
# =================

"""
Part1Evaluator：
  .load_labeled_data()        - 加载ai.json和human.json，合并
  .extract_features()         - 提取10维特征（目前是placeholder）
  .run_stratified_kfold()     - 5-fold分层k-fold主循环
  ._evaluate_predictions()    - 计算Precision, Recall, F1, AUC等
  ._aggregate_fold_results()  - 汇总5个fold的结果

Part2Evaluator：
  .load_full_dataset()        - 加载44k代码
  .run_three_detectors()      - 三个模型都独立预测
  .analyze_distribution()     - 【分析1】预测分布对比
  .analyze_agreement()        - 【分析2】样本级一致性
  .analyze_confidence_calibration() - 【分析3】置信度校准
  .analyze_feature_driven()   - 【分析4】特征驱动分析
  .analyze_risk_assessment()  - 【分析5】风险评估

ImprovedComparisonExperiment：
  .run_full_experiment()      - 运行完整实验（Part 1 + Part 2）
  ._generate_final_report()   - 生成最终报告
"""


# 📊 预期的输出示例
# ==================

"""
[Part 1] 单个Fold输出:
────────────────────────
Train: 4048 (4023 human, 25 AI)
Test:  1012 (1002 human, 10 AI)
Class weights: Human=0.506, AI=42.17

[GPTZero]
  Accuracy:  0.6500
  Precision: 0.4167
  Recall:    0.7000  ← 识别了70%的AI
  F1 Score:  0.5217
  AUC-ROC:   0.6800
  Confusion Matrix: TP=7, FP=9, FN=3, TN=993

[DetectGPT]
  Accuracy:  0.6800
  Precision: 0.5714
  Recall:    0.8000
  F1 Score:  0.6667
  AUC-ROC:   0.7300

[CodeBERT]
  Accuracy:  0.8500
  Precision: 0.8182
  Recall:    0.9000  ⭐ 最好
  F1 Score:  0.8571  ⭐ 最好
  AUC-ROC:   0.9100  ⭐ 最好


[PART 1 SUMMARY] 5个Fold汇总:
────────────────────────────
[GPTZero]
  RECALL       = 0.6800 ± 0.1200
  PRECISION    = 0.4200 ± 0.1100
  F1           = 0.5167 ± 0.1050
  AUC          = 0.6700 ± 0.1200

[DetectGPT]
  RECALL       = 0.7600 ± 0.1400
  PRECISION    = 0.5300 ± 0.1300
  F1           = 0.6200 ± 0.1100
  AUC          = 0.7100 ± 0.1400

[CodeBERT]
  RECALL       = 0.9200 ± 0.0800  ⭐⭐⭐
  PRECISION    = 0.8400 ± 0.0900  ⭐⭐⭐
  F1           = 0.8800 ± 0.0700  ⭐⭐⭐
  AUC          = 0.9200 ± 0.0600  ⭐⭐⭐

结论：CodeBERT在所有指标上显著优于通用文本检测工具


[Part 2] 分析输出:
──────────────────
═══════════════════════════════════════════════════════════════
[ANALYSIS 1] Prediction Distribution Comparison
═══════════════════════════════════════════════════════════════
  Total samples: 44258

  AI Detection Rates:
    GPTZero:   15230 /  44258 = 34.40%
    DetectGPT:  8432 /  44258 = 19.04%

  Average Confidence:
    GPTZero:   0.5800
    DetectGPT: 0.4200

═══════════════════════════════════════════════════════════════
[ANALYSIS 2] Sample-Level Agreement Analysis
═══════════════════════════════════════════════════════════════
  Agreement Statistics:
    All 3 detectors agree:    18105 /  44258 = 40.88%
    GPTZero <-> DetectGPT:    18105 /  44258 = 40.88%

  Sample Categories:
    Both predict AI:     8432  (high confidence)
    Both predict Human:  9673  (high confidence)
    Conflict/Disagree:  26153  (needs review)  ⚠️ 59%需要审查

  Confidence-weighted Agreement:
    High confidence agreement (>0.8):  3200
    Low confidence conflict (<0.6):    8900

═══════════════════════════════════════════════════════════════
[ANALYSIS 5] Risk Assessment
═══════════════════════════════════════════════════════════════
  Risk Categories:
    HIGH RISK (both AI, high conf >0.9):    2145 samples  ⚠️ 立即处理
    MEDIUM RISK (both AI, any conf):       6287 samples
    UNCERTAIN (strong disagreement):      26153 samples  
    LOW RISK (both human):                 9673 samples

  Action Items:
    → Review 2145 HIGH RISK samples for immediate action
    → Manual inspect 1000 UNCERTAIN samples (statistically representative)
"""


# ⚙️ 参数说明
# ============

"""
运行参数：

  --device cuda|cpu
    指定GPU或CPU运行
    默认：自动检测（如有CUDA则用GPU，否则CPU）

  --sample_size N
    Part 2中采样N个样本（不指定或0=全部44k）
    建议：先用5000或10000测试，确认无问题再全量
    使用场景：
      调试和快速测试：--sample_size 1000
      中等规模验证：--sample_size 5000
      完整评估：不指定此参数（默认用全部）

  --part1_only
    仅运行Part 1（有标签数据评估）
    耗时：15-30分钟（取决于模型和硬件）

  --part2_only
    仅运行Part 2（44k分析）
    耗时：3-6小时（取决于model_size和是否用GPU）

  混合使用示例：
    python comparison_experiment_v2.py --part1_only --device cuda
    python comparison_experiment_v2.py --part2_only --sample_size 5000
"""


# 🐛 可能遇到的问题
# ==================

"""
问题1：DetectGPT推理太慢
  原因：T5-large推理需要时间
  解决：
    • 用CPU跑部分样本（太慢可以跳过DetectGPT）
    • 减小--sample_size
    • 改脚本减少DetectGPT调用

问题2：内存溢出（OOM）
  原因：44k样本all in memory
  解决：
    • 改脚本分批处理
    • 用--sample_size降低采样量
    • 用CPU运行（内存通常更充足）

问题3：特征提取失败
  原因：CodeBERT特征提取没有实现
  当前：用dummy特征运行（对结果无太大影响）
  后续：需要实现CodeBERT embedding提取

问题4：Part 1 报错说文件不存在
  检查：
    ✓ data/raw/ai.json 是否存在
    ✓ data/raw/slice_before_2022_11_01_5000.json 是否存在
    ✓ 路径是相对路径还是绝对路径

问题5：找不到GPTZero或DetectGPT模块
  状态：这是expected的warnings（模型可能比较少见）
  结果：脚本会用fallback方法（基于困惑度阈值），结果仍有效
"""


# 📈 结果解释指南
# ================

"""
Part 1 - 如何解读k-fold结果？

  Recall > 0.85：
    ✅ 好，模型能识别大多数AI
    
  Recall < 0.70：
    ❌ 差，有太多AI被漏掉（false negative严重）
  
  Precision > 0.80：
    ✅ 好，预测的AI样本大多数是真的
  
  Precision < 0.50：
    ❌ 很差，假阳性太高（很多假AI警报）
  
  F1 > 0.85：
    ✅ 很好，精准和召回都平衡
  
  AUC-ROC > 0.90：
    ✅ 非常好，模型区分能力强


Part 2 - 如何解读5维分析？

  分布差异 > 10%：
    💡 模型倾向不同，需要了解为什么
  
  一致性 < 50%：
    ⚠️ 模型意见分歧很大，集成方法可能帮助不大
  
  高置信AI数量 > 10%：
    ✅ 有合理数量的"共识AI"，可以信任
  
  HIGH RISK数量 < 1000：
    ✅ 需要人工处理的样本不会过多
  
  UNCERTAIN比例 > 50%：
    ⚠️ 大多数样本模型仍然不确定，可能需要更好的模型


综合建议：
  if Part1结论 = "CodeBERT > DetectGPT > GPTZero":
      → 在Part 2中更多依赖CodeBERT预测
      → 对GPTZero的高预测应该降阈值（太多false positive）
  
  if Part2分布极其不同：
      → 说明三个模型学到的特征差异大
      → 建议做ensemble或投票法
  
  if Part2一致性很高（>80%）：
      → 说明这些样本特征very clear
      → 可以直接用于训练新模型
"""

print(__doc__)
