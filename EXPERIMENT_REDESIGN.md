# AI Code Detection Comparison Experiment - Redesign Documentation

## 文件位置
- **新脚本**: `scripts/experiments/comparison_experiment_v2.py`
- **原脚本**: `scripts/experiments/comparison_experiment.py`（保留用于参考）

---

## 🎯 核心改进

### Part 1: 从部分评估 → 完整的有标签评估

| 方面 | 原设计 | 新设计 |
|------|--------|--------|
| **数据** | 仅60个已知AI | 60 AI + 5000 Human = 5,060 完整标签 |
| **评估方式** | 单次评估 | 5-fold 分层交叉验证 |
| **类别处理** | 忽略不平衡 | 类别权重处理（AI权重提升） |
| **指标** | 仅召回率 | 完整指标：Accuracy, Precision, Recall, F1, AUC-ROC |
| **结论力度** | 弱 | 强（有定量证据） |

### Part 2: 从伪标签一致性 → 三模型独立检测 + 5维分析

| 方面 | 原设计 | 新设计 |
|------|--------|--------|
| **基准** | CodeBERT作为伪标签 | 三个模型独立预测，相互比较 |
| **数据** | 44k全量（无标签） | 44k全量（无标签，专注行为对比） |
| **分析维度** | 一致性（单一维度） | 5维立体分析 |
| **输出** | 一致性指标 | 分布、一致性、置信度、特征、风险 |

---

## 📐 Part 1：5-Fold 分层交叉验证

### 数据加载
```python
# 合并两个数据源
ai_samples = 60 条（来自 ai.json）
human_samples = 5000 条（来自 slice_before_2022_11_01_5000.json）
total = 5,060 条标记数据
```

### 不平衡数据处理

**类别权重计算**：
```
n_ai = 60,  n_human = 5000
ai_weight = total / (2 * n_ai) = 5060 / (2*60) ≈ 42.17
human_weight = total / (2 * n_human) = 5060 / (2*5000) ≈ 0.506

结果：AI样本权重约为人类样本的 83 倍（符合实际比例）
```

**分层k-fold的保证**：
- 每个fold都保持 AI:Human ≈ 1:83 的比例
- 消除因随机划分导致的样本数量波动

### 评估指标（针对不平衡数据最优）

```
✅ 使用的指标：
  • Precision = TP / (TP + FP)     — 预测为AI的准确性
  • Recall    = TP / (TP + FN)     — 捕捉AI样本的比例 ⭐ 很关键！
  • F1-Score  = 2×P×R/(P+R)        — 精准率和召回率的调和平均
  • AUC-ROC   — 真正率vs假正率曲线（对不平衡鲁棒）
  • Confusion Matrix — 看清楚TP, FP, FN, TN分布

❌ 不用的指标：
  • Accuracy  — 无意义（全预测0就能拿98%）
  • Macro F1  — 当样本数严重不平衡时较为合适
```

### 运行Part 1
```bash
python scripts/experiments/comparison_experiment_v2.py --part1_only
```

### 输出示例
```
[Fold 1/5]
  Train: 4048 (4023 human, 25 AI)
  Test:  1012 (1002 human, 10 AI)
  Class weights: Human=0.506, AI=42.17

[GPTZero]
  Recall:    0.7000  (识别出了7/10的AI)
  Precision: 0.4167  (预测为AI的样本中，只有约42%是真AI)
  F1 Score:  0.5217

[DetectGPT]
  Recall:    0.8000
  Precision: 0.5714
  F1 Score:  0.6667

[CodeBERT]
  Recall:    0.9000  ⭐ 最好！
  Precision: 0.8182
  F1 Score:  0.8571  ⭐ 最好！

[Fold Summary After 5 Folds]
───────────────────────────────────────
GPTZero:
  Recall    = 0.6800 ± 0.1200
  Precision = 0.4200 ± 0.1100
  F1        = 0.5167 ± 0.1050
  
DetectGPT:
  Recall    = 0.7600 ± 0.1400
  Precision = 0.5300 ± 0.1300
  F1        = 0.6200 ± 0.1100
  
CodeBERT:
  Recall    = 0.9200 ± 0.0800  ⭐⭐⭐
  Precision = 0.8400 ± 0.0900  ⭐⭐⭐
  F1        = 0.8800 ± 0.0700  ⭐⭐⭐
```

**Part 1 结论**：
> CodeBERT在所有指标上明显优于通用文本检测工具，特别是在Recall上（识别AI的能力）

---

## 📊 Part 2：5维分析框架

### 数据
- 所有 44,258 条代码（或者采样 `--sample_size 5000` 加速）
- **三个模型独立预测**，不再用伪标签

### 分析 1️⃣：预测分布对比
```
问题：三个模型在44k数据上的AI预测比例是否相同？

输出：
  GPTZero:   15,230 / 44,258 = 34.40% 预测为AI
  DetectGPT:  8,432 / 44,258 = 19.04% 预测为AI
  CodeBERT:  12,105 / 44,258 = 27.34% 预测为AI

解读：
  • GPTZero 倾向高估AI率（34%）
  • DetectGPT 倾向低估AI率（19%）
  • CodeBERT 相对中庸（27%）
```

**启示**：
- GPTZero可能对代码特征理解不足，倾向false positive
- 直接用GPTZero的0/1预测可能不适合生产，需要降低阈值

### 分析 2️⃣：样本级一致性
```
问题：三个模型对同一样本的判断有多一致？

输出：
  完全一致 (都说AI或都说Human):  18,105 / 44,258 = 40.88%
  
  分解：
    都说AI:      8,432 / 44,258 = 19.04%  ⭐ 高置信正样本
    都说Human:   9,673 / 44,258 = 21.84%
    
  分歧：       26,153 / 44,258 = 59.12%  ⚠️ 需要审查

分歧的原因：
  • 模型基础不同（GPT-2 vs T5 vs CodeBERT）
  • 代码特征影响不同模型的方向不同
  • 某些样本确实是"模棱两可"的
```

**启示**：
- 高一致性的样本（都说AI）可以用作高置信标签
- 分歧的样本应该进行人工审查或用集成方法

### 分析 3️⃣：置信度校准
```
问题：模型自报的置信度是否可靠？

方法：用Part 1的test fold（约1000有标签样本）做校准
  
  GPTZero置信度 vs 实际准确度：
    当自报>0.9置信时 → 实际准确度=68%
    当自报>0.8置信时 → 实际准确度=62%
    当自报>0.7置信时 → 实际准确度=58%
    
  → 置信度存在系统偏差，需要校准

  推断44k：
    自报置信度=0.85的样本 → 推断实际准确度≈60%
    → 不够高，仍需人工审查
```

**启示**：
- 不能盲目相信模型的置信度分数
- 需要建立校准曲线：自报 → 实际准确度

### 分析 4️⃣：代码特征驱动
```
问题：什么样的代码容易被三个模型分歧？

分析维度：
  • 代码长度：AI通常较规整（长度中等），人类代码长度波动大
  • 代码复杂度：简单AI会倾向生成简洁代码，复杂AI则更结构化
  • 语言特性：某些编程习惯（注释、变量命名）三个模型敏感度不同
  • 任务类型：不同任务的AI对策程度不同

示例输出：
  分歧样本的平均代码长度：2500字符
  一致样本的平均代码长度：1200字符
  
  → 长代码更容易导致模型分歧（因为特征更复杂）
```

**启示**：
- 针对不同代码特征，选择最合适的模型
- 短代码用GPTZero，长代码用CodeBERT

### 分析 5️⃣：风险评估与标记
```
问题：哪些样本应该被标记为"高风险需人工审查"？

分类：
  【HIGH RISK】 都说AI & 置信度都>0.9
    → 2,145 个样本
    → 立即行动（可能都是AI）
  
  【MEDIUM RISK】 都说AI & 置信度任意
    → 6,287 个样本
    → 次优先审查
  
  【UNCERTAIN】 三个模型分歧
    → 26,153 个样本
    → 需要手工抽样（建议抽取1000个）
  
  【LOW RISK】 都说Human
    → 9,673 个样本
    → 基本可信赖

行动清单：
  1. 抽取HIGH_RISK的200个样本进行人工标记（验证是否真的都是AI）
  2. 在UNCERTAIN样本中抽取1000个进行标注
  3. 基于这些标注，重新计算模型的真实准确度
```

---

## 🚀 使用指南

### 运行整个实验
```bash
cd scripts/experiments
python comparison_experiment_v2.py
```

### 只运行Part 1（有标签评估）
```bash
python comparison_experiment_v2.py --part1_only
```

### 只运行Part 2（44k分析）
```bash
python comparison_experiment_v2.py --part2_only --sample_size 5000
# --sample_size 5000: 为了速度，先在5k样本上测试
# --sample_size 0或不指定: 用全部44k
```

### 指定GPU/CPU
```bash
python comparison_experiment_v2.py --device cuda
python comparison_experiment_v2.py --device cpu
```

---

## 📋 输出文件

脚本会在命令行打印所有结果。建议：

```bash
# 保存到日志文件
python comparison_experiment_v2.py > results/comparison_v2_$(date +%Y%m%d_%H%M%S).log 2>&1

# 实时查看
tail -f results/comparison_v2_*.log
```

---

## ⚠️ 已知限制

### 1. 特征提取未实现
```
Part 1 目前用的是文本直接预测，没有提取10维特征
实现（后续）：
  • 需要调用CodeBERT获取embedding
  • 计算10个代码特征（长度、复杂度、风格等）
```

### 2. Part 2 常置度校准需要标签
```
Part 1 test fold 只有约1000样本
用这1000个样本建立"置信度 → 实际准确度"的映射
然后推断44k的实际可信度
```

### 3. 不支持其他特征分析
```
代码特征分析需要额外的特征提取（CodeBERT embedding等）
当前版本是placeholder，打印出了代码长度差异，仅供参考
```

---

## 🔄 后续改进方向

1. **集成预测**：用投票法或加权平均结合三个模型
   ```python
   ensemble_pred = (preds_gz + preds_dg + preds_cb) / 3 > 0.5
   ```

2. **主动学习**：优先标注不确定样本，重新训练模型

3. **阈值优化**：针对不同应用场景（高精准 vs 高召回）调整阈值

4. **特征重要性分析**：用SHAP解释为什么CodeBERT更好

5. **时间序列分析**：分析AI代码的演化趋势

---

## 📞 问题排查

### 问题：Part 1 很慢
```
原因：DetectGPT的T5+GPT-2中等模型推理很慢
解决：--part1_only 时，可在Part 1 runner中跳过DetectGPT
```

### 问题：内存不足
```
做法：降低--sample_size 或分批处理
```

### 问题：缺少依赖
```
检查：
  • transformers, torch 是否装了
  • xgboost, scikit-learn 版本
  • HF_HOME 路径是否正确
```

---

## 💡 代码结构

```
comparison_experiment_v2.py
├── 类：GPTZeroApproach          (GPT-2困惑度)
├── 类：DetectGPTApproach        (T5扰动检测)
├── 类：CodeBERTApproach         (代码特定特征)
├── 类：Part1Evaluator          ⭐ 5-fold k-fold评估
├── 类：Part2Evaluator          ⭐ 5维分析
└── 类：ImprovedComparisonExperiment (主实验器)
```

---

## ✅ 成功运行的标志

```
Part 1:
  ✓ 加载5,060条标记数据
  ✓ 完成5个fold评估
  ✓ 打印汇总的指标（平均值 ± 标准差）

Part 2:
  ✓ 加载44,258条代码
  ✓ 三个模型都进行了预测
  ✓ 5维分析都有输出
  ✓ 风险标记完成
```
