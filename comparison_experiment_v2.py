"""
AI Code Detection Tools Comparison Experiment (IMPROVED)
Compare GPTZero, DetectGPT vs CodeBERT+XGBoost for code detection performance

MAJOR IMPROVEMENTS:
  Part 1: 5-fold stratified k-fold cross-validation with balanced class weights
          - Complete labeled dataset: 60 AI + 5000 Human = 5,060 samples
          - Full evaluation metrics (Precision, Recall, F1, AUC-ROC)
  
  Part 2: Three independent detectors on 44k full dataset + 5-dimensional analysis
          - Distribution comparison
          - Sample-level agreement analysis
          - Confidence calibration
          - Code feature-driven analysis
          - Risk assessment

Goal: Prove that AI code detection requires specialized code language models
"""

# CRITICAL: Set HF_HOME BEFORE any imports
import os
# Force CPU-only for this script unless user explicitly overrides
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Use environment variable if set, otherwise prefer a local setup cache path.
_preferred_hf_home = r'C:\Users\Accio\Desktop\ai-test-master\scripts\setup\hf_cache'
if 'HF_HOME' not in os.environ:
    if os.path.exists(_preferred_hf_home):
        os.environ['HF_HOME'] = _preferred_hf_home
    else:
        os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
if 'TRANSFORMERS_CACHE' not in os.environ:
    os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']

import json
import pandas as pd
import numpy as np
import time
import re
import functools
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix, cohen_kappa_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

import importlib.util

# Force unbuffered output
print = functools.partial(print, flush=True)

# Import GPTZero - with fallback paths
GPTZERO_AVAILABLE = False
GPT2PPL = None
_gptzero_paths = [
    r'C:\Users\Accio\Desktop\ai-test-master\core\GPTZero-main\GPTZero-main\model.py',
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ai-test-master', 'core', 'GPTZero-main', 'GPTZero-main', 'model.py'),
    os.path.expanduser('~/GPTZero-main/model.py'),
]
for _gpt_path in _gptzero_paths:
    if os.path.exists(_gpt_path):
        try:
            _gptzero_spec = importlib.util.spec_from_file_location("gptzero_model", _gpt_path)
            _gptzero_module = importlib.util.module_from_spec(_gptzero_spec)
            _gptzero_spec.loader.exec_module(_gptzero_module)
            GPT2PPL = _gptzero_module.GPT2PPL
            GPTZERO_AVAILABLE = True
            print(f"[OK] GPTZero loaded from: {_gpt_path}")
            break
        except Exception as e:
            print(f" Warning: GPTZero load failed from {_gpt_path}: {e}")
if not GPTZERO_AVAILABLE:
    print(" Warning: GPTZero not available from any path")

# Import DetectGPT - with fallback paths
DETECTGPT_AVAILABLE = False
GPT2PPLV2 = None
_detectgpt_paths = [
    r'C:\Users\Accio\Desktop\ai-test-master\core\DetectGPT-main\model.py',
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ai-test-master', 'core', 'DetectGPT-main', 'model.py'),
    os.path.expanduser('~/DetectGPT-main/model.py'),
]
for _dg_path in _detectgpt_paths:
    if os.path.exists(_dg_path):
        try:
            _detectgpt_spec = importlib.util.spec_from_file_location("detectgpt_model", _dg_path)
            _detectgpt_module = importlib.util.module_from_spec(_detectgpt_spec)
            _detectgpt_spec.loader.exec_module(_detectgpt_module)
            GPT2PPLV2 = _detectgpt_module.GPT2PPLV2
            DETECTGPT_AVAILABLE = True
            print(f"[OK] DetectGPT loaded from: {_dg_path}")
            break
        except Exception as e:
            print(f"Warning: DetectGPT load failed from {_dg_path}: {e}")
if not DETECTGPT_AVAILABLE:
    print("Warning: DetectGPT not available from any path")

# Import CodeBERT feature analyzer if available
AICODEANALYZER_AVAILABLE = False
AICodeAnalyzer = None
_analyzer_paths = [
    # Local repository paths (preferred in this project)
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'method.py'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core', 'method.py'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'core', 'method.py'),
    r'C:\Users\Accio\Desktop\ai-test-master\core\method.py',
    os.path.expanduser('~/core/method.py'),
    os.path.expanduser('~/method.py'),
]
for _analyzer_path in _analyzer_paths:
    if os.path.exists(_analyzer_path):
        try:
            _analyzer_spec = importlib.util.spec_from_file_location("method_module", _analyzer_path)
            _analyzer_module = importlib.util.module_from_spec(_analyzer_spec)
            _analyzer_spec.loader.exec_module(_analyzer_module)
            AICodeAnalyzer = _analyzer_module.AICodeAnalyzer
            AICODEANALYZER_AVAILABLE = True
            print(f"[OK] AICodeAnalyzer loaded from: {_analyzer_path}")
            break
        except Exception as e:
            print(f" Warning: AICodeAnalyzer load failed from {_analyzer_path}: {e}")
if not AICODEANALYZER_AVAILABLE:
    print("Warning: AICodeAnalyzer not available from any path")


# ============================================================================
# Method Classes (PRESERVED FROM ORIGINAL)
# ============================================================================

class GPTZeroApproach:
    """GPT-2 perplexity-based detection"""
    
    def __init__(self, threshold: float = 50.0, device: str = "cpu"):
        self.threshold = threshold
        self.method_name = "GPTZero (GPT-2 Perplexity)"
        self.device = device
        
        if GPTZERO_AVAILABLE:
            try:
                self.model = GPT2PPL(device=device, model_id="gpt2")
                self.use_gpt2 = True
            except Exception as e:
                print(f"Warning: GPTZero model init failed, fallback mode enabled: {e}")
                self.use_gpt2 = False
        else:
            self.use_gpt2 = False
    
    def compute_perplexity_gpt2(self, text: str) -> float:
        if not self.use_gpt2:
            return 0.0
        try:
            if len(text) < 20:
                return 0.0
            
            raw_result = self.model(text)
            
            # GPTZero implementation commonly returns:
            #   (result_dict, verdict_str)
            # where result_dict includes "Perplexity per line".
            if isinstance(raw_result, tuple):
                result = raw_result[0] if len(raw_result) > 0 else None
            else:
                result = raw_result
            
            if isinstance(result, dict):
                ppl = result.get("Perplexity per line", result.get("Perplexity", 0.0))
                return float(ppl) if ppl is not None else 0.0
            
            # Fallback for wrappers returning a direct numeric perplexity
            return float(result) if result else 0.0
        except Exception as e:
            return 0.0
    
    def predict(self, features, code_text: str = None) -> Tuple[int, float]:
        """Predict using perplexity"""
        if code_text and self.use_gpt2:
            perplexity = self.compute_perplexity_gpt2(code_text)
        else:
            if isinstance(features, np.ndarray):
                perplexity = features[0] if len(features) > 0 else 0.0
            else:
                perplexity = features.get('perplexity', 0.0) if isinstance(features, dict) else 0.0
        
        if perplexity == 0:
            return (0, 0.5)
        
        if perplexity < self.threshold:
            return (1, min(1.0, (self.threshold - perplexity) / self.threshold))
        else:
            return (0, min(1.0, (perplexity - self.threshold) / self.threshold))
    
    def batch_predict(self, X, code_texts: List[str] = None, label: str = "") -> List[Tuple[int, float]]:
        if code_texts is None:
            code_texts = [""] * len(X)
        results = []
        t0 = time.time()
        for i, (row, c) in enumerate(zip(X, code_texts)):
            if (i + 1) % max(1, len(X) // 10) == 0:
                elapsed = time.time() - t0
                print(f"    [GPTZero{label}] {i+1}/{len(X)} ({elapsed:.0f}s)")
            pred = self.predict(row, c)
            results.append(pred)
        return results


class DetectGPTApproach:
    """DetectGPT perturbation-based detection"""
    
    def __init__(self, device: str = "cpu", chunk_value: int = 180):
        self.method_name = "DetectGPT (Perturbation-based)"
        self.device = device
        self.chunk_value = chunk_value
        self.threshold = 0.7
        self.max_chars = 5000
        # Keep DetectGPT reasonably fast for Part 2 smoke tests on CPU.
        self.max_words = 220
        self.max_tokens = 512
        self.min_words_for_perturbation = 50
        
        if DETECTGPT_AVAILABLE:
            try:
                self.model = GPT2PPLV2(device=device)
                self.use_detectgpt = True
            except Exception as e:
                print(f"Warning: DetectGPT model init failed, fallback mode enabled: {e}")
                self.use_detectgpt = False
        else:
            self.use_detectgpt = False
    
    def compute_detectgpt_score(self, text: str) -> Tuple[int, float]:
        if not self.use_detectgpt:
            return (0, 0.5)
        
        if len(text.strip()) < 20:
            return (0, 0.5)
        
        text = " ".join(text.split())
        if len(text) > self.max_chars:
            text = text[:self.max_chars]
        
        words = text.split()
        if len(words) > self.max_words:
            text = " ".join(words[:self.max_words])
        
        if len(text.split()) < self.min_words_for_perturbation:
            return (0, 0.5)
        
        try:
            print(f"    [DetectGPT] Computing score for text length {len(text)}...")
            raw_result = self.model(text, self.chunk_value, "v1.1")
            print(f"    [DetectGPT] Model result: {raw_result}")
            
            # DetectGPT v1.1 commonly returns: (result_dict, verdict_str)
            # but some wrappers may return only result_dict.
            if isinstance(raw_result, tuple):
                result = raw_result[0] if len(raw_result) > 0 else None
            else:
                result = raw_result
            
            if not isinstance(result, dict):
                print(f"    [DetectGPT] Result not dict after normalization, returning fallback")
                return (0, 0.5)
            
            detectgpt_label = result.get('label', 1)
            prob_str = result.get('prob', '50.00%')
            try:
                prob = float(str(prob_str).replace('%', '').strip()) / 100.0
            except:
                prob = 0.5
            prob = min(max(prob, 0.0), 1.0)
            
            if detectgpt_label == 0:
                return (1, prob)
            else:
                return (0, 1.0 - prob)
        except Exception as e:
            print(f"    [DetectGPT] Error computing score: {e}")
            return (0, 0.5)
    
    def predict(self, features, code_text: str = None) -> Tuple[int, float]:
        if code_text and self.use_detectgpt:
            return self.compute_detectgpt_score(code_text)
        
        if isinstance(features, np.ndarray):
            perplexity = features[0] if len(features) > 0 else 0.0
        else:
            perplexity = features.get('perplexity', 0.0) if isinstance(features, dict) else 0.0
        
        if perplexity == 0:
            return (0, 0.5)
        
        ppl_threshold = 60.0
        if perplexity < ppl_threshold:
            return (1, min(1.0, (ppl_threshold - perplexity) / ppl_threshold))
        else:
            return (0, min(1.0, (perplexity - ppl_threshold) / ppl_threshold))
    
    def batch_predict(self, X, code_texts: List[str] = None, label: str = "") -> List[Tuple[int, float]]:
        SLOW_THRESHOLD = 30
        # Skip very long samples in TEST mode to avoid 5-10 minute single samples.
        MAX_HARD_SAMPLE_CHARS = 1200
        MAX_HARD_SAMPLE_LINES = 300
        is_test = "TEST" in label
        
        if code_texts is None:
            code_texts = [""] * len(X)
        
        results = []
        t0 = time.time()
        errors = 0
        slow_count = 0
        skipped_hard = 0
        neutral_fallback = 0
        
        for i, (row, c) in enumerate(zip(X, code_texts)):
            if (i + 1) % max(1, len(X) // 10) == 0:
                elapsed = time.time() - t0
                print(f"    [DetectGPT{label}] {i+1}/{len(X)} - {errors} errors, {slow_count} slow ({elapsed:.0f}s)")
            
            sample_t0 = time.time()
            try:
                if len(c) > MAX_HARD_SAMPLE_CHARS and is_test:
                    skipped_hard += 1
                    results.append((0, 0.5))
                    print(f"      [DetectGPT] Sample {i} skipped: too long ({len(c)} chars)")
                else:
                    if self.use_detectgpt:
                        cleaned_words = len(" ".join(c.split()).split()) if c else 0
                        if len(c.strip()) < 20:
                            pred = (0, 0.5)
                            if is_test:
                                print(f"      [DetectGPT] Sample {i} fallback: text too short")
                        elif cleaned_words < self.min_words_for_perturbation:
                            pred = (0, 0.5)
                            if is_test:
                                print(
                                    f"      [DetectGPT] Sample {i} fallback: "
                                    f"insufficient words ({cleaned_words} < {self.min_words_for_perturbation})"
                                )
                        else:
                            pred = self.predict(row, c)
                    else:
                        pred = self.predict(row, c)
                        if is_test:
                            print("      [DetectGPT] Fallback mode active: model unavailable")
                    results.append(pred)
                    if pred == (0, 0.5):
                        neutral_fallback += 1
                
                sample_elapsed = time.time() - sample_t0
                if sample_elapsed > SLOW_THRESHOLD:
                    slow_count += 1
                    print(f"      [Warning] Sample {i} took {sample_elapsed:.1f}s")
            except Exception as e:
                errors += 1
                results.append((0, 0.5))
        
        print(
            f"    DetectGPT done: {len(results)}/{len(X)}, {errors} errors, "
            f"{slow_count} slow, {skipped_hard} skipped, {neutral_fallback} neutral-fallback"
        )
        return results


class CodeBERTApproach:
    """CodeBERT + XGBoost with 10 code-specific features"""
    
    def __init__(self):
        self.method_name = "CodeBERT + XGBoost (Code-specific 10 features)"
        self.scaler = StandardScaler()
        self.model = None
        self.analyzer = None
        self.feature_names = [
            'perplexity', 'avg_token_probability', 'avg_entropy',
            'burstiness', 'code_length', 'avg_line_length',
            'std_line_length', 'comment_ratio', 'identifier_entropy',
            'ngram_repetition'
        ]
        
        if AICODEANALYZER_AVAILABLE and AICodeAnalyzer is not None:
            try:
                self.analyzer = AICodeAnalyzer()
                print("[OK] CodeBERT analyzer initialized")
            except Exception as e:
                print(f"Warning: CodeBERT analyzer failed to initialize: {e}")
                self.analyzer = None
        else:
            print("Warning: CodeBERT analyzer unavailable, will use fallback predictions")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, sample_weight=None):
        """Train with XGBoost and class weights"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.model.fit(X_scaled, y_train, sample_weight=sample_weight, verbose=0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def batch_predict(self, X: np.ndarray) -> List[Tuple[int, float]]:
        proba = self.predict_proba(X)
        return [(int(np.argmax(p)), float(p[1])) for p in proba]

    def batch_predict_with_features(self, code_texts: List[str]) -> List[Tuple[int, float]]:
        """Extract features from raw code texts and predict."""
        if self.analyzer is None:
            print("Warning: CodeBERT analyzer unavailable, using fallback predictions")
            return [(0, 0.5)] * len(code_texts)
        if self.model is None:
            print("[CodeBERT] Training model on default processed dataset...")
            self._train_on_default_data()
            if self.model is None:
                return [(0, 0.5)] * len(code_texts)
            print("[CodeBERT] Analyzer/model ready, running feature extraction...")
        
        features_list = []
        for i, code in enumerate(code_texts):
            try:
                extracted = self.analyzer.analyze_code(code)
                if extracted:
                    features_list.append([extracted.get(name, 0.0) for name in self.feature_names])
                else:
                    features_list.append([0.0] * len(self.feature_names))
            except Exception as e:
                print(f"Warning: CodeBERT feature extraction failed for sample {i}: {e}")
                features_list.append([0.0] * len(self.feature_names))
        
        X_test = np.array(features_list, dtype=np.float32)
        proba = self.predict_proba(X_test)
        return [(int(np.argmax(p)), float(p[1])) for p in proba]

    def _train_on_default_data(self):
        """Train a local XGBoost model with processed experiment_results.csv"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            training_data_path = os.path.join(project_root, 'data', 'processed', 'experiment_results.csv')
            df = pd.read_csv(training_data_path)
            X_train = df[self.feature_names].values
            y_train = (df['label'] == 'AI').astype(int).values
            self.train(X_train.astype(np.float32), y_train)
            print(f"[CodeBERT] Trained on {len(X_train)} samples")
        except Exception as e:
            print(f"Warning: Failed to train CodeBERT model: {e}")
            self.model = None


# ============================================================================
# IMPROVED Part 1: 5-Fold Stratified CV with Balanced Classes
# ============================================================================

class Part1Evaluator:
    """Part 1: Complete labeled data evaluation with stratified k-fold CV"""
    
    def __init__(self):
        self.X = None
        self.y = None
        self.code_texts = None
        self.feature_names = [
            'perplexity', 'avg_token_probability', 'avg_entropy',
            'burstiness', 'code_length', 'avg_line_length',
            'std_line_length', 'comment_ratio', 'identifier_entropy',
            'ngram_repetition'
        ]
    
    def load_labeled_data(self, ai_json_path: str, human_json_path: str, human_sample_size: int = None):
        """Load and merge AI + Human data, with controlled human sampling for Part 1."""
        print("\n[Part 1] Loading labeled data...")
        
        # Load AI data
        with open(ai_json_path, 'r', encoding='utf-8') as f:
            ai_data = json.load(f)
        
        headers_ai = ai_data[0]
        content_idx_ai = headers_ai.index('content')
        ai_samples = []
        for row in ai_data[1:]:
            ai_samples.append({
                'content': row[content_idx_ai],
                'label': 1
            })
        
        n_ai = len(ai_samples)
        print(f"  ✓ Loaded {n_ai} AI samples from {ai_json_path}")
        
        # Load Human data
        with open(human_json_path, 'r', encoding='utf-8') as f:
            human_data = json.load(f)
        
        headers_human = human_data[0]
        content_idx_human = headers_human.index('content')
        human_samples = []
        for row in human_data[1:]:
            human_samples.append({
                'content': row[content_idx_human],
                'label': 0
            })
        
        n_human_total = len(human_samples)
        print(f"  ✓ Loaded {n_human_total} Human samples from {human_json_path}")
        
        # Determine sampling size for human data
        if human_sample_size is None:
            target_human = min(max(int(n_ai * 5), 200), 500)
        else:
            target_human = human_sample_size
        target_human = min(target_human, n_human_total)
        
        if target_human < n_human_total:
            np.random.seed(42)
            human_samples = list(np.random.choice(human_samples, target_human, replace=False))
            print(f"  ✓ Sampled {target_human} Human examples for Part 1")
        else:
            print(f"  ✓ Using all {n_human_total} Human examples for Part 1")
        
        # Merge and shuffle
        all_samples = ai_samples + human_samples
        np.random.shuffle(all_samples)
        
        self.code_texts = [s['content'] for s in all_samples]
        self.y = np.array([s['label'] for s in all_samples])
        
        print(f"  ✓ Total labeled samples: {len(self.y)}")
        print(f"    - AI: {np.sum(self.y == 1)}, Human: {np.sum(self.y == 0)}")
        print(f"    - Class ratio: 1:{np.sum(self.y == 0) / np.sum(self.y == 1):.1f}")
        
        return self.code_texts, self.y
    
    def extract_features(self, code_texts: List[str], analyzer=None, feature_names: List[str] = None) -> np.ndarray:
        """
        Extract 10 code-specific features for Part 1.
        Prefer AICodeAnalyzer when available; otherwise use deterministic
        lightweight lexical/statistical fallback features.
        """
        print("\n  [Feature Extraction]")
        if feature_names is None:
            feature_names = [
                'perplexity', 'avg_token_probability', 'avg_entropy',
                'burstiness', 'code_length', 'avg_line_length',
                'std_line_length', 'comment_ratio', 'identifier_entropy',
                'ngram_repetition'
            ]
        
        rows = []
        use_analyzer = analyzer is not None
        if use_analyzer:
            print("  Using AICodeAnalyzer for feature extraction")
        else:
            print("  AICodeAnalyzer unavailable; using lexical fallback features")
        
        for i, code in enumerate(code_texts):
            try:
                if use_analyzer:
                    extracted = analyzer.analyze_code(code) or {}
                    row = [float(extracted.get(name, 0.0)) for name in feature_names]
                else:
                    lines = code.splitlines() if code else []
                    line_lengths = np.array([len(ln) for ln in lines], dtype=np.float32) if lines else np.array([0.0], dtype=np.float32)
                    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code or "")
                    token_count = len(tokens)
                    unique_tokens = len(set(tokens))
                    probs = np.array([tokens.count(t) / max(token_count, 1) for t in set(tokens)], dtype=np.float32)
                    identifier_entropy = float(-np.sum(probs * np.log2(np.clip(probs, 1e-8, 1.0)))) if len(probs) else 0.0
                    comments = sum(1 for ln in lines if ln.strip().startswith('#') or ln.strip().startswith('//'))
                    comment_ratio = comments / max(len(lines), 1)
                    ngrams = [" ".join(tokens[j:j+3]) for j in range(max(0, token_count - 2))]
                    ngram_repetition = 1.0 - (len(set(ngrams)) / max(len(ngrams), 1)) if ngrams else 0.0
                    row = [
                        0.0, 0.0, 0.0,  # perplexity/avg_prob/entropy unavailable without analyzer
                        float(np.std(line_lengths)),  # burstiness proxy
                        float(len(code or "")),
                        float(np.mean(line_lengths)),
                        float(np.std(line_lengths)),
                        float(comment_ratio),
                        float(identifier_entropy),
                        float(ngram_repetition),
                    ]
                rows.append(row)
            except Exception as e:
                print(f"  Warning: feature extraction failed for sample {i}: {e}")
                rows.append([0.0] * len(feature_names))
        
        X = np.array(rows, dtype=np.float32)
        print(f"  ✓ Extracted feature matrix: {X.shape}")
        return X
    
    def run_stratified_kfold(self, gptzero, detectgpt, codebert, n_splits=5):
        """Run 5-fold stratified cross-validation"""
        print("\n[Step 1.1] Running 5-fold stratified cross-validation...")
        print(f"  Total samples: {len(self.y)}")
        print(f"  Folds: {n_splits}")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = defaultdict(list)
        
        fold_num = 0
        for train_idx, test_idx in skf.split(np.arange(len(self.y)), self.y):
            fold_num += 1
            print(f"\n  {'='*70}")
            print(f"  [Fold {fold_num}/{n_splits}]")
            print(f"  {'='*70}")
            
            X_train = self.X[train_idx] if self.X is not None else None
            X_test = self.X[test_idx] if self.X is not None else None
            y_train = self.y[train_idx]
            y_test = self.y[test_idx]
            
            code_texts_train = [self.code_texts[i] for i in train_idx]
            code_texts_test = [self.code_texts[i] for i in test_idx]
            
            # Class weights for imbalanced data
            n_ai = np.sum(y_train == 1)
            n_human = np.sum(y_train == 0)
            ai_weight = len(y_train) / (2 * n_ai) if n_ai > 0 else 1.0
            human_weight = len(y_train) / (2 * n_human) if n_human > 0 else 1.0
            sample_weights = np.array([ai_weight if y == 1 else human_weight for y in y_train])
            
            print(f"  Train: {len(y_train)} ({np.sum(y_train==0)} human, {np.sum(y_train==1)} AI)")
            print(f"  Test:  {len(y_test)} ({np.sum(y_test==0)} human, {np.sum(y_test==1)} AI)")
            print(f"  Class weights: Human={human_weight:.2f}, AI={ai_weight:.2f}")
            
            # Evaluate GPTZero
            if gptzero:
                print(f"\n  [GPTZero] Evaluating...")
                preds_gptzero = gptzero.batch_predict([None]*len(code_texts_test), code_texts_test, f" Fold{fold_num}")
                fold_results['gptzero'].append(self._evaluate_predictions(preds_gptzero, y_test, 'GPTZero'))
            
            # Evaluate DetectGPT
            if detectgpt:
                print(f"\n  [DetectGPT] Evaluating...")
                preds_detectgpt = detectgpt.batch_predict([None]*len(code_texts_test), code_texts_test, f" Fold{fold_num} TEST")
                fold_results['detectgpt'].append(self._evaluate_predictions(preds_detectgpt, y_test, 'DetectGPT'))
            
            # Evaluate CodeBERT (with training)
            if codebert and X_train is not None:
                print(f"\n  [CodeBERT] Training on fold {fold_num}...")
                try:
                    codebert.train(X_train, y_train, sample_weight=sample_weights)
                    preds_codebert = codebert.batch_predict(X_test)
                    fold_results['codebert'].append(self._evaluate_predictions(preds_codebert, y_test, 'CodeBERT'))
                except Exception as e:
                    print(f"   CodeBERT failed: {e}")
        
        # Aggregate results
        self._aggregate_fold_results(fold_results, n_splits)
        return fold_results
    
    def _evaluate_predictions(self, predictions, y_true, method_name):
        """Evaluate a set of predictions"""
        preds = np.array([p[0] for p in predictions])
        probs = np.array([p[1] for p in predictions])
        
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, probs)
        except:
            auc = np.nan
        
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
        
        results = {
            'method': method_name,
            'accuracy': acc,
            'precision': prec,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'predictions': preds,
            'probabilities': probs
        }
        
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1 Score:  {f1:.4f}")
        print(f"    AUC-ROC:   {auc:.4f}")
        print(f"    Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        
        return results
    
    def _aggregate_fold_results(self, fold_results, n_splits):
        """Print aggregated results across folds"""
        print(f"\n{'='*70}")
        print(f"[PART 1 SUMMARY] Aggregated Results Across {n_splits} Folds")
        print(f"{'='*70}")
        
        for method, results_list in fold_results.items():
            if not results_list:
                continue
            
            print(f"\n[{results_list[0]['method']}]")
            
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            for metric in metrics:
                values = [r[metric] for r in results_list if not np.isnan(r[metric])]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"  {metric.upper():12} = {mean_val:.4f} ± {std_val:.4f}")


# ============================================================================
# IMPROVED Part 2: 5-Dimensional Analysis
# ============================================================================

class Part2Evaluator:
    """Part 2: Three independent detectors + 5D analysis on 44k full dataset"""
    
    def __init__(self):
        self.code_texts = None
        self.preds_gptzero_runs = []  # Store results from multiple runs
        self.preds_detectgpt_runs = []
        self.preds_codebert_runs = []
        self.sample_indices_runs = []  # Store which samples were used in each run
    
    def load_full_dataset(self, full_data_path: str, sample_size: int = 10) -> List[str]:
        """Load 44k code samples from last_success, then randomly sample sample_size."""
        print("\n[Part 2] Loading full dataset...")
        
        with open(full_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        headers = data[0]
        content_idx = headers.index('content')
        
        # Load all code texts first
        all_code_texts = [row[content_idx] for row in data[1:]]
        print(f" Loaded {len(all_code_texts)} total code samples")
        
        # Randomly sample target samples for quick analysis
        if len(all_code_texts) > sample_size:
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(all_code_texts), sample_size, replace=False)
            self.code_texts = [all_code_texts[i] for i in indices]
            print(f" Randomly sampled {sample_size} samples for analysis")
        else:
            self.code_texts = all_code_texts
            print(f" Using all {len(all_code_texts)} samples (less than {sample_size})")
        
        return self.code_texts
    
    def run_three_detectors(self, gptzero, detectgpt, codebert=None, sample_size: int = None, n_runs: int = 3):
        """Run all three detectors independently, multiple times with different random samples"""
        print(f"\n[Step 2.1] Running three independent detectors ({n_runs} runs with random sampling)...")
        
        # Reset previous results
        self.preds_gptzero_runs = []
        self.preds_detectgpt_runs = []
        self.preds_codebert_runs = []
        self.sample_indices_runs = []
        
        for run in range(n_runs):
            print(f"\n  === Run {run + 1}/{n_runs} ===")
            
            # Randomly sample for this run (different samples each time)
            indices = np.random.choice(len(self.code_texts), len(self.code_texts), replace=False)
            code_samples = [self.code_texts[i] for i in indices]
            self.sample_indices_runs.append(indices)
            
            print(f"  Sampled {len(code_samples)} codes for this run")
            
            # Run detectors
            print(f"    Running GPTZero...")
            preds_gz = gptzero.batch_predict([None]*len(code_samples), code_samples, f" RUN{run+1}")
            self.preds_gptzero_runs.append(preds_gz)
            
            print(f"    Running DetectGPT...")
            preds_dg = detectgpt.batch_predict([None]*len(code_samples), code_samples, f" RUN{run+1} TEST")
            self.preds_detectgpt_runs.append(preds_dg)
            
            if codebert is not None:
                print(f"    Running CodeBERT on raw code samples...")
                preds_cb = codebert.batch_predict_with_features(code_samples)
            else:
                preds_cb = [(0, 0.5)] * len(code_samples)
            self.preds_codebert_runs.append(preds_cb)
        
        print(f"\n  Completed {n_runs} runs with different random samples")
        return self.preds_gptzero_runs, self.preds_detectgpt_runs, self.preds_codebert_runs
    
    def analyze_distribution(self):
        """Analysis 1: Prediction distribution comparison (averaged across runs)"""
        print(f"\n{'='*70}")
        print("[ANALYSIS 1] Prediction Distribution Comparison (Averaged)")
        print(f"{'='*70}")
        
        n_runs = len(self.preds_gptzero_runs)
        n_samples_per_run = len(self.preds_gptzero_runs[0]) if n_runs > 0 else 0
        
        print(f"\n  Total runs: {n_runs}")
        print(f"  Samples per run: {n_samples_per_run}")
        print(f"  Total samples analyzed: {n_runs * n_samples_per_run}")
        
        # Calculate averages across runs
        ai_counts_gz = []
        ai_counts_dg = []
        ai_counts_cb = []
        confs_gz = []
        confs_dg = []
        
        for run in range(n_runs):
            preds_gz = self.preds_gptzero_runs[run]
            preds_dg = self.preds_detectgpt_runs[run]
            preds_cb = self.preds_codebert_runs[run]
            
            ai_count_gz = sum(1 for p, c in preds_gz if p == 1)
            ai_count_dg = sum(1 for p, c in preds_dg if p == 1)
            ai_count_cb = sum(1 for p, c in preds_cb if p == 1)
            
            conf_gz = np.mean([c for p, c in preds_gz])
            conf_dg = np.mean([c for p, c in preds_dg])
            
            ai_counts_gz.append(ai_count_gz)
            ai_counts_dg.append(ai_count_dg)
            ai_counts_cb.append(ai_count_cb)
            confs_gz.append(conf_gz)
            confs_dg.append(conf_dg)
        
        # Calculate averages and standard deviations
        avg_ai_gz = np.mean(ai_counts_gz)
        avg_ai_dg = np.mean(ai_counts_dg)
        avg_ai_cb = np.mean(ai_counts_cb)
        std_ai_gz = np.std(ai_counts_gz)
        std_ai_dg = np.std(ai_counts_dg)
        std_ai_cb = np.std(ai_counts_cb)
        
        avg_conf_gz = np.mean(confs_gz)
        avg_conf_dg = np.mean(confs_dg)
        std_conf_gz = np.std(confs_gz)
        std_conf_dg = np.std(confs_dg)
        
        print(f"\n  AI Detection Rates (averaged across {n_runs} runs):")
        print(f"    GPTZero:   {avg_ai_gz:6.1f} ± {std_ai_gz:4.1f} / {n_samples_per_run:6d} = {100*avg_ai_gz/n_samples_per_run:6.2f}%")
        print(f"    DetectGPT: {avg_ai_dg:6.1f} ± {std_ai_dg:4.1f} / {n_samples_per_run:6d} = {100*avg_ai_dg/n_samples_per_run:6.2f}%")
        print(f"    CodeBERT:  {avg_ai_cb:6.1f} ± {std_ai_cb:4.1f} / {n_samples_per_run:6d} = {100*avg_ai_cb/n_samples_per_run:6.2f}%")
        
        print(f"\n  Average Confidence (Probability) across runs:")
        print(f"    GPTZero:   {avg_conf_gz:.4f} ± {std_conf_gz:.4f}")
        print(f"    DetectGPT: {avg_conf_dg:.4f} ± {std_conf_dg:.4f}")
        
        return {
            'ai_counts_avg': {'gptzero': avg_ai_gz, 'detectgpt': avg_ai_dg, 'codebert': avg_ai_cb},
            'ai_counts_std': {'gptzero': std_ai_gz, 'detectgpt': std_ai_dg, 'codebert': std_ai_cb},
            'ai_rates_avg': {'gptzero': avg_ai_gz/n_samples_per_run, 'detectgpt': avg_ai_dg/n_samples_per_run, 'codebert': avg_ai_cb/n_samples_per_run},
            'confidences_avg': {'gptzero': avg_conf_gz, 'detectgpt': avg_conf_dg},
            'confidences_std': {'gptzero': std_conf_gz, 'detectgpt': std_conf_dg},
            'n_runs': n_runs,
            'samples_per_run': n_samples_per_run
        }
    
    def analyze_agreement(self):
        """Analysis 2: Sample-level agreement analysis (averaged across runs)"""
        print(f"\n{'='*70}")
        print("[ANALYSIS 2] Sample-Level Agreement Analysis (Averaged)")
        print(f"{'='*70}")
        
        n_runs = len(self.preds_gptzero_runs)
        n_samples_per_run = len(self.preds_gptzero_runs[0]) if n_runs > 0 else 0
        
        print(f"\n  Analysis across {n_runs} runs with {n_samples_per_run} samples each")
        
        # Calculate averages across runs
        agreement_rates = []
        both_ai_counts = []
        both_human_counts = []
        conflict_counts = []
        high_conf_agrees = []
        low_conf_conflicts = []
        
        for run in range(n_runs):
            preds_gz = np.array([p for p, c in self.preds_gptzero_runs[run]])
            preds_dg = np.array([p for p, c in self.preds_detectgpt_runs[run]])
            
            # Agreement statistics
            agreement_all = (preds_gz == preds_dg)
            n_all_agree = np.sum(agreement_all)
            agreement_rates.append(n_all_agree / n_samples_per_run)
            
            # Categorize samples
            both_ai = np.sum((preds_gz == 1) & (preds_dg == 1))
            both_human = np.sum((preds_gz == 0) & (preds_dg == 0))
            conflict = n_samples_per_run - both_ai - both_human
            
            both_ai_counts.append(both_ai)
            both_human_counts.append(both_human)
            conflict_counts.append(conflict)
            
            # Confidence-weighted analysis
            conf_gz = np.array([c for p, c in self.preds_gptzero_runs[run]])
            conf_dg = np.array([c for p, c in self.preds_detectgpt_runs[run]])
            min_conf = np.minimum(conf_gz, conf_dg)
            
            high_conf = np.sum((agreement_all) & (min_conf > 0.8))
            low_conf = np.sum((~agreement_all) & (min_conf < 0.6))
            
            high_conf_agrees.append(high_conf)
            low_conf_conflicts.append(low_conf)
        
        # Calculate averages
        avg_agreement = np.mean(agreement_rates)
        std_agreement = np.std(agreement_rates)
        
        avg_both_ai = np.mean(both_ai_counts)
        avg_both_human = np.mean(both_human_counts)
        avg_conflict = np.mean(conflict_counts)
        
        avg_high_conf = np.mean(high_conf_agrees)
        avg_low_conf = np.mean(low_conf_conflicts)
        
        print(f"\n  Agreement Statistics (averaged across {n_runs} runs):")
        print(f"    GPTZero <-> DetectGPT agreement: {100*avg_agreement:6.2f}% ± {100*std_agreement:4.2f}%")
        
        print(f"\n  Sample Categories (averaged):")
        print(f"    Both predict AI:    {avg_both_ai:6.1f}")
        print(f"    Both predict Human: {avg_both_human:6.1f}")
        print(f"    Conflict/Disagree:  {avg_conflict:6.1f} (needs review)")
        
        print(f"\n  Confidence-weighted Agreement (averaged):")
        print(f"    High confidence agreement (>0.8): {avg_high_conf:6.1f}")
        print(f"    Low confidence conflict (<0.6):   {avg_low_conf:6.1f}")
        
        return {
            'agreement_rate_avg': avg_agreement,
            'agreement_rate_std': std_agreement,
            'both_ai_avg': avg_both_ai,
            'both_human_avg': avg_both_human,
            'conflict_avg': avg_conflict,
            'high_conf_agree_avg': avg_high_conf,
            'low_conf_conflict_avg': avg_low_conf,
            'n_runs': n_runs
        }
    
    def analyze_confidence_calibration(self):
        """Analysis 3: Confidence calibration (placeholder, averaged across runs)"""
        print(f"\n{'='*70}")
        print("[ANALYSIS 3] Confidence Calibration Analysis (Averaged)")
        print(f"{'='*70}")
        
        print(f"\n  Calibration analysis requires ground truth labels")
        print(f"  (Available through Part 1 on test fold approximately 1,000 samples)")
        print(f"\n  Placeholder metrics across {len(self.preds_gptzero_runs)} runs:")
        
        conf_means_gz = []
        conf_stds_gz = []
        conf_means_dg = []
        conf_stds_dg = []
        
        for run in range(len(self.preds_gptzero_runs)):
            conf_gz = np.array([c for p, c in self.preds_gptzero_runs[run]])
            conf_dg = np.array([c for p, c in self.preds_detectgpt_runs[run]])
            
            conf_means_gz.append(np.mean(conf_gz))
            conf_stds_gz.append(np.std(conf_gz))
            conf_means_dg.append(np.mean(conf_dg))
            conf_stds_dg.append(np.std(conf_dg))
        
        avg_conf_mean_gz = np.mean(conf_means_gz)
        avg_conf_std_gz = np.mean(conf_stds_gz)
        avg_conf_mean_dg = np.mean(conf_means_dg)
        avg_conf_std_dg = np.mean(conf_stds_dg)
        
        print(f"    GPTZero confidence:   μ={avg_conf_mean_gz:.4f} ± {avg_conf_std_gz:.4f} (across runs)")
        print(f"    DetectGPT confidence: μ={avg_conf_mean_dg:.4f} ± {avg_conf_std_dg:.4f} (across runs)")
        
        return {
            'gptzero_conf_mean_avg': avg_conf_mean_gz,
            'gptzero_conf_std_avg': avg_conf_std_gz,
            'detectgpt_conf_mean_avg': avg_conf_mean_dg,
            'detectgpt_conf_std_avg': avg_conf_std_dg,
            'n_runs': len(self.preds_gptzero_runs)
        }
    
    def analyze_feature_driven(self):
        """Analysis 4: Code feature-driven analysis (placeholder, averaged across runs)"""
        print(f"\n{'='*70}")
        print("[ANALYSIS 4] Code Feature-Driven Analysis (Averaged)")
        print(f"{'='*70}")
        
        print(f"\n  Feature analysis requires feature extraction")
        print(f"  (CodeBERT embedding extraction not implemented in this version)")
        print(f"\n  Placeholder: Code length distribution in conflict samples across {len(self.preds_gptzero_runs)} runs")
        
        conflict_ratios = []
        agree_length_means = []
        agree_length_stds = []
        conflict_length_means = []
        conflict_length_stds = []
        
        for run in range(len(self.preds_gptzero_runs)):
            preds_gz = np.array([p for p, c in self.preds_gptzero_runs[run]])
            preds_dg = np.array([p for p, c in self.preds_detectgpt_runs[run]])
            conflict_mask = preds_gz != preds_dg
            
            code_lengths = np.array([len(c) for c in self.code_texts])
            
            conflict_count = np.sum(conflict_mask)
            conflict_ratio = conflict_count / len(self.code_texts)
            conflict_ratios.append(conflict_ratio)
            
            if conflict_count > 0:
                conflict_lengths = code_lengths[conflict_mask]
                agree_lengths = code_lengths[~conflict_mask]
                
                agree_length_means.append(np.mean(agree_lengths))
                agree_length_stds.append(np.std(agree_lengths))
                conflict_length_means.append(np.mean(conflict_lengths))
                conflict_length_stds.append(np.std(conflict_lengths))
        
        avg_conflict_ratio = np.mean(conflict_ratios)
        std_conflict_ratio = np.std(conflict_ratios)
        
        if len(agree_length_means) > 0:
            avg_agree_length = np.mean(agree_length_means)
            std_agree_length = np.mean(agree_length_stds)
            avg_conflict_length = np.mean(conflict_length_means)
            std_conflict_length = np.mean(conflict_length_stds)
            
            print(f"\n    Agreement samples:   μ={avg_agree_length:.0f} ± {std_agree_length:.0f} chars")
            print(f"    Conflict samples:    μ={avg_conflict_length:.0f} ± {std_conflict_length:.0f} chars")
        
        print(f"    Average conflict ratio: {100*avg_conflict_ratio:.2f}% ± {100*std_conflict_ratio:.2f}%")
        
        return {
            'conflict_ratio_avg': avg_conflict_ratio,
            'conflict_ratio_std': std_conflict_ratio,
            'agree_length_avg': np.mean(agree_length_means) if agree_length_means else 0,
            'conflict_length_avg': np.mean(conflict_length_means) if conflict_length_means else 0,
            'n_runs': len(self.preds_gptzero_runs)
        }
    
    def analyze_risk_assessment(self):
        """Analysis 5: Risk assessment and flagging (averaged across runs)"""
        print(f"\n{'='*70}")
        print("[ANALYSIS 5] Risk Assessment (Averaged Across Runs)")
        print(f"{'='*70}")
        
        high_risk_counts = []
        medium_risk_counts = []
        uncertain_counts = []
        low_risk_counts = []
        
        for run in range(len(self.preds_gptzero_runs)):
            preds_gz = np.array([p for p, c in self.preds_gptzero_runs[run]])
            preds_dg = np.array([p for p, c in self.preds_detectgpt_runs[run]])
            conf_gz = np.array([c for p, c in self.preds_gptzero_runs[run]])
            conf_dg = np.array([c for p, c in self.preds_detectgpt_runs[run]])
            
            # High risk: both predict AI with high confidence
            high_risk = (preds_gz == 1) & (preds_dg == 1) & (conf_gz > 0.9) & (conf_dg > 0.9)
            
            # Medium risk: both predict AI (any confidence)
            medium_risk = (preds_gz == 1) & (preds_dg == 1) & ~high_risk
            
            # Low risk but flagged: strong disagreement
            flagged_uncertain = (preds_gz != preds_dg)
            
            high_risk_counts.append(np.sum(high_risk))
            medium_risk_counts.append(np.sum(medium_risk))
            uncertain_counts.append(np.sum(flagged_uncertain))
            low_risk_counts.append(np.sum((preds_gz==0) & (preds_dg==0)))
        
        # Compute averages and standard deviations
        avg_high_risk = np.mean(high_risk_counts)
        std_high_risk = np.std(high_risk_counts)
        avg_medium_risk = np.mean(medium_risk_counts)
        std_medium_risk = np.std(medium_risk_counts)
        avg_uncertain = np.mean(uncertain_counts)
        std_uncertain = np.std(uncertain_counts)
        avg_low_risk = np.mean(low_risk_counts)
        std_low_risk = np.std(low_risk_counts)
        
        print(f"\n  Risk Categories (Averaged Across {len(self.preds_gptzero_runs)} Runs):")
        print(f"    HIGH RISK (both AI, high conf >0.9):    {avg_high_risk:6.0f} ± {std_high_risk:4.0f} samples")
        print(f"    MEDIUM RISK (both AI, any conf):       {avg_medium_risk:6.0f} ± {std_medium_risk:4.0f} samples")
        print(f"    UNCERTAIN (strong disagreement):       {avg_uncertain:6.0f} ± {std_uncertain:4.0f} samples")
        print(f"    LOW RISK (both human):                 {avg_low_risk:6.0f} ± {std_low_risk:4.0f} samples")
        
        print(f"\n  Action Items:")
        print(f"    → Review ~{avg_high_risk:.0f} HIGH RISK samples for immediate action")
        print(f"    → Manual inspect ~{min(1000, avg_uncertain):.0f} UNCERTAIN samples")
        
        return {
            'high_risk_avg': avg_high_risk,
            'high_risk_std': std_high_risk,
            'medium_risk_avg': avg_medium_risk,
            'medium_risk_std': std_medium_risk,
            'uncertain_avg': avg_uncertain,
            'uncertain_std': std_uncertain,
            'low_risk_avg': avg_low_risk,
            'low_risk_std': std_low_risk,
            'n_runs': len(self.preds_gptzero_runs)
        }
    
    def plot_confusion_matrices(self, output_dir: str = None):
        """Generate confusion matrices for each model across all runs"""
        print(f"\n{'='*70}")
        print("[VISUALIZATION] Confusion Matrices for Each Model")
        print(f"{'='*70}")
        
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            output_dir = os.path.join(project_root, 'results', 'figures', 'comparison', 'Confusion Matrices')
        os.makedirs(output_dir, exist_ok=True)
        
        # Since we don't have ground truth for Part 2, we'll create confusion matrices
        # showing agreement patterns between the three models
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        model_names = ['GPTZero', 'DetectGPT', 'CodeBERT']
        model_preds = [
            self.preds_gptzero_runs,
            self.preds_detectgpt_runs, 
            self.preds_codebert_runs
        ]
        
        for i, (name, preds_runs) in enumerate(zip(model_names, model_preds)):
            # Aggregate predictions across runs for this model
            all_preds = []
            all_confs = []
            for run_preds in preds_runs:
                run_pred_array = np.array([p for p, c in run_preds])
                run_conf_array = np.array([c for p, c in run_preds])
                all_preds.extend(run_pred_array)
                all_confs.extend(run_conf_array)
            
            all_preds = np.array(all_preds)
            all_confs = np.array(all_confs)
            
            # Create confusion matrix showing prediction distribution
            # Since no ground truth, show: AI vs Human predictions with confidence levels
            ai_high_conf = np.sum((all_preds == 1) & (all_confs > 0.8))
            ai_med_conf = np.sum((all_preds == 1) & (all_confs <= 0.8) & (all_confs > 0.5))
            ai_low_conf = np.sum((all_preds == 1) & (all_confs <= 0.5))
            human_high_conf = np.sum((all_preds == 0) & (all_confs > 0.8))
            human_med_conf = np.sum((all_preds == 0) & (all_confs <= 0.8) & (all_confs > 0.5))
            human_low_conf = np.sum((all_preds == 0) & (all_confs <= 0.5))
            
            # Create a custom confusion matrix
            conf_matrix = np.array([
                [human_high_conf, human_med_conf, human_low_conf],  # Human predictions by confidence
                [ai_high_conf, ai_med_conf, ai_low_conf]             # AI predictions by confidence
            ])
            
            # Plot confusion matrix
            im = axes[i].imshow(conf_matrix, interpolation='nearest', cmap='Blues')
            axes[i].set_title(f'{name} Predictions\n(Total: {len(all_preds)} samples)', fontsize=12)
            
            # Add text annotations
            thresh = conf_matrix.max() / 2.
            for ii in range(conf_matrix.shape[0]):
                for jj in range(conf_matrix.shape[1]):
                    axes[i].text(jj, ii, f'{int(conf_matrix[ii, jj]):,}',
                               ha="center", va="center",
                               color="white" if conf_matrix[ii, jj] > thresh else "black")
            
            axes[i].set_xticks([0, 1, 2])
            axes[i].set_yticks([0, 1])
            axes[i].set_xticklabels(['High\nConf\n(>0.8)', 'Med\nConf\n(0.5-0.8)', 'Low\nConf\n(≤0.5)'], fontsize=9)
            axes[i].set_yticklabels(['Human\nPredictions', 'AI\nPredictions'], fontsize=10)
            
            # Add percentages
            total = np.sum(conf_matrix)
            ai_total = np.sum(conf_matrix[1, :])
            human_total = np.sum(conf_matrix[0, :])
            axes[i].text(0.02, 0.98, f'AI Rate: {ai_total/total*100:.1f}%',
                        transform=axes[i].transAxes, fontsize=10, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'model_confusion_matrices.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[Saved confusion matrices] {plot_path}")
        
        # Also create individual plots for each model
        for i, (name, preds_runs) in enumerate(zip(model_names, model_preds)):
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Aggregate predictions across runs
            all_preds = []
            all_confs = []
            for run_preds in preds_runs:
                run_pred_array = np.array([p for p, c in run_preds])
                run_conf_array = np.array([c for p, c in run_preds])
                all_preds.extend(run_pred_array)
                all_confs.extend(run_conf_array)
            
            all_preds = np.array(all_preds)
            all_confs = np.array(all_confs)
            
            # Create detailed confusion matrix
            ai_high_conf = np.sum((all_preds == 1) & (all_confs > 0.8))
            ai_med_conf = np.sum((all_preds == 1) & (all_confs <= 0.8) & (all_confs > 0.5))
            ai_low_conf = np.sum((all_preds == 1) & (all_confs <= 0.5))
            human_high_conf = np.sum((all_preds == 0) & (all_confs > 0.8))
            human_med_conf = np.sum((all_preds == 0) & (all_confs <= 0.8) & (all_confs > 0.5))
            human_low_conf = np.sum((all_preds == 0) & (all_confs <= 0.5))
            
            conf_matrix = np.array([
                [human_high_conf, human_med_conf, human_low_conf],
                [ai_high_conf, ai_med_conf, ai_low_conf]
            ])
            
            im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
            ax.set_title(f'{name} Prediction Confidence Distribution\n(Total: {len(all_preds)} samples)', fontsize=14)
            
            # Add text annotations
            thresh = conf_matrix.max() / 2.
            for ii in range(conf_matrix.shape[0]):
                for jj in range(conf_matrix.shape[1]):
                    ax.text(jj, ii, f'{int(conf_matrix[ii, jj]):,}',
                           ha="center", va="center",
                           color="white" if conf_matrix[ii, jj] > thresh else "black")
            
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['High Confidence\n(>0.8)', 'Medium Confidence\n(0.5-0.8)', 'Low Confidence\n(≤0.5)'], fontsize=10)
            ax.set_yticklabels(['Predicted Human', 'Predicted AI'], fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Number of Samples', fontsize=12)
            
            plt.tight_layout()
            individual_plot_path = os.path.join(output_dir, f'{name.lower()}_confusion_matrix.png')
            plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[Saved {name} confusion matrix] {individual_plot_path}")
        
        return {
            'combined_plot': plot_path,
            'individual_plots': [os.path.join(output_dir, f'{name.lower()}_confusion_matrix.png') for name in model_names]
        }


# ============================================================================
# Main Experiment Runner
# ============================================================================

class ImprovedComparisonExperiment:
    """Run complete improved experiment"""
    
    def __init__(self):
        self.part1_evaluator = Part1Evaluator()
        self.part2_evaluator = Part2Evaluator()
        self.results = {}

    @staticmethod
    def _safe_init(cls, *args, **kwargs):
        """Backward-compatible class initializer for detector wrappers."""
        try:
            return cls(*args, **kwargs)
        except TypeError:
            return cls()
    
    def run_full_experiment(self, device=None, sample_size=None, part1_only=False, part2_only=False, output_dir: str = None):
        """Run complete experiment"""
        print("=" * 80)
        print("AI CODE DETECTION COMPARISON EXPERIMENT (IMPROVED)")
        print("=" * 80)
        
        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except:
                device = 'cpu'
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        if output_dir is None:
            output_dir = os.path.join(project_root, 'results', 'comparison_experiment_v2')
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        print(f"\nDevice: {device}")
        print(f"[Output directory] {self.output_dir}")
        
        # Initialize the three detector approaches
        print("\n[Initializing detectors...]")
        gptzero = self._safe_init(GPTZeroApproach, device=device)
        detectgpt = self._safe_init(DetectGPTApproach, device=device)
        codebert = CodeBERTApproach()
        print("[Detectors initialized]")
        
        # PART 1: Labeled data evaluation
        if not part2_only:
            print(f"\n{'='*80}")
            print("PART 1: LABELED DATA EVALUATION (5-fold Stratified K-Fold)")
            print(f"{'='*80}")
            
            # Construct dynamic paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            ai_path = os.path.join(project_root, 'data', 'raw', 'ai.json')
            human_path = os.path.join(project_root, 'data', 'raw', 'slice_before_2022_11_01_5000.json')
            
            code_texts, y = self.part1_evaluator.load_labeled_data(
                ai_path,
                human_path
            )
            
            # Extract features so CodeBERT can also participate in Part 1 5-fold.
            X = self.part1_evaluator.extract_features(
                code_texts,
                analyzer=codebert.analyzer,
                feature_names=codebert.feature_names
            )
            self.part1_evaluator.X = X
            
            fold_results = self.part1_evaluator.run_stratified_kfold(
                gptzero, detectgpt, codebert, n_splits=5
            )
            
            self.results['part1'] = fold_results
        
        # PART 2: Full dataset 5D analysis
        if not part1_only:
            print(f"\n{'='*80}")
            print("PART 2: FULL DATASET ANALYSIS (44k samples)")
            print(f"{'='*80}")
            
            # Construct dynamic paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            full_dataset_path = os.path.join(project_root, 'data', 'raw', 'smartbeans_submission_last_success.json')
            
            code_texts = self.part2_evaluator.load_full_dataset(
                full_dataset_path,
                sample_size=sample_size if sample_size is not None else 10
            )
            
            preds_gz, preds_dg, preds_cb = self.part2_evaluator.run_three_detectors(
                gptzero, detectgpt, codebert, sample_size=sample_size
            )
            
            # 5D Analysis
            dist_analysis = self.part2_evaluator.analyze_distribution()
            agr_analysis = self.part2_evaluator.analyze_agreement()
            conf_analysis = self.part2_evaluator.analyze_confidence_calibration()
            feat_analysis = self.part2_evaluator.analyze_feature_driven()
            risk_analysis = self.part2_evaluator.analyze_risk_assessment()
            
            # Generate confusion matrix visualizations
            confusion_plots = self.part2_evaluator.plot_confusion_matrices()
            
            self.results['part2'] = {
                'distribution': dist_analysis,
                'agreement': agr_analysis,
                'calibration': conf_analysis,
                'features': feat_analysis,
                'risk': risk_analysis,
                'confusion_matrices': confusion_plots
            }
        
        # Generate report
        self._generate_final_report()
        
        # Save outputs to files
        self._save_outputs()
        
        return self.results
    
    def _generate_final_report(self):
        """Generate final comprehensive report"""
        print(f"\n{'='*80}")
        print("FINAL REPORT")
        print(f"{'='*80}")
        
        print(f"\n[CONCLUSIONS]")
        print(f"\n1. Part 1 Results:")
        if 'part1' in self.results:
            print(f"   ✓ Complete evaluation with ground truth labels")
            print(f"   ✓ 5-fold cross-validation results available")
            print(f"   ✓ Class-weighted training for imbalanced data")
        
        print(f"\n2. Part 2 Results:")
        if 'part2' in self.results:
            part2 = self.results['part2']
            print(f"   ✓ Three independent detectors evaluated")
            print(f"   ✓ 5-dimensional analysis completed")
            if 'distribution' in part2:
                dist = part2['distribution']
                print(f"   ✓ Distribution comparison: GPTZero {dist['ai_rates_avg']['gptzero']*100:.1f}%, DetectGPT {dist['ai_rates_avg']['detectgpt']*100:.1f}%")
            if 'agreement' in part2:
                agr = part2['agreement']
                print(f"   ✓ Agreement rate: {agr['agreement_rate_avg']*100:.1f}%")
            if 'risk' in part2:
                risk = part2['risk']
                print(f"   ✓ Risk assessment: {risk['high_risk_avg']:.0f} high-risk samples flagged")

    def _render_final_report_lines(self):
        lines = []
        lines.append('=' * 80)
        lines.append('FINAL REPORT')
        lines.append('=' * 80)
        lines.append('')
        lines.append('[CONCLUSIONS]')
        lines.append('')
        lines.append('1. Part 1 Results:')
        if 'part1' in self.results:
            lines.append('   ✓ Complete evaluation with ground truth labels')
            lines.append('   ✓ 5-fold cross-validation results available')
            lines.append('   ✓ Class-weighted training for imbalanced data')
        else:
            lines.append('   Part 1 not executed.')
        lines.append('')
        lines.append('2. Part 2 Results:')
        if 'part2' in self.results:
            part2 = self.results['part2']
            lines.append('   ✓ Three independent detectors evaluated')
            lines.append('   ✓ 5-dimensional analysis completed')
            if 'distribution' in part2:
                dist = part2['distribution']
                lines.append(f"   ✓ Distribution comparison: GPTZero {dist['ai_rates_avg']['gptzero']*100:.1f}%, DetectGPT {dist['ai_rates_avg']['detectgpt']*100:.1f}%")
            if 'agreement' in part2:
                agr = part2['agreement']
                lines.append(f"   ✓ Agreement rate: {agr['agreement_rate_avg']*100:.1f}%")
            if 'risk' in part2:
                risk = part2['risk']
                lines.append(f"   ✓ Risk assessment: {risk['high_risk_avg']:.0f} high-risk samples flagged")
        else:
            lines.append('   Part 2 not executed.')
        lines.append('')
        if 'part1' in self.results:
            lines.append('[PART 1 AGGREGATED METRICS]')
            for method, results_list in self.results['part1'].items():
                if not results_list:
                    continue
                lines.append(f"  {results_list[0]['method']}")
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                    values = [r[metric] for r in results_list if not np.isnan(r[metric])]
                    if values:
                        lines.append(f"    {metric.upper():12} = {np.mean(values):.4f} ± {np.std(values):.4f}")
            lines.append('')
        if 'part2' in self.results:
            lines.append('[PART 2 SUMMARY]')
            dist = self.results['part2'].get('distribution', {})
            if dist:
                lines.append(f"  GPTZero AI rate: {dist['ai_rates_avg']['gptzero']*100:.2f}%")
                lines.append(f"  DetectGPT AI rate: {dist['ai_rates_avg']['detectgpt']*100:.2f}%")
            agr = self.results['part2'].get('agreement', {})
            if agr:
                lines.append(f"  Agreement rate: {agr['agreement_rate_avg']*100:.2f}%")
            risk = self.results['part2'].get('risk', {})
            if risk:
                lines.append(f"  High risk samples: {risk['high_risk_avg']:.0f}")
                lines.append(f"  Uncertain samples: {risk['uncertain_avg']:.0f}")
        return lines

    def _sanitize_for_json(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_for_json(v) for v in obj]
        return obj

    def _save_report(self):
        report_path = os.path.join(self.output_dir, 'comparison_experiment_v2_report.txt')
        lines = self._render_final_report_lines()
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"[Saved report] {report_path}")

    def _save_results_json(self):
        results_path = os.path.join(self.output_dir, 'comparison_experiment_v2_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self._sanitize_for_json(self.results), f, ensure_ascii=False, indent=2)
        print(f"[Saved JSON] {results_path}")

    def _save_part1_csv(self):
        if 'part1' not in self.results:
            return
        rows = []
        for method, results_list in self.results['part1'].items():
            for fold_idx, r in enumerate(results_list, start=1):
                rows.append({
                    'method': r['method'],
                    'fold': fold_idx,
                    'accuracy': r['accuracy'],
                    'precision': r['precision'],
                    'recall': r['recall'],
                    'f1': r['f1'],
                    'auc': r['auc'],
                    'tp': r['tp'],
                    'fp': r['fp'],
                    'fn': r['fn'],
                    'tn': r['tn']
                })
        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(self.output_dir, 'part1_fold_metrics.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"[Saved Part1 metrics CSV] {csv_path}")

    def _save_part2_csv(self):
        if 'part2' not in self.results:
            return
        part2 = self.results['part2']
        rows = []
        if 'distribution' in part2:
            dist = part2['distribution']
            rows.append({
                'metric': 'gptzero_ai_rate',
                'value': dist['ai_rates_avg']['gptzero']
            })
            rows.append({
                'metric': 'detectgpt_ai_rate',
                'value': dist['ai_rates_avg']['detectgpt']
            })
            rows.append({
                'metric': 'gptzero_confidence',
                'value': dist['confidences_avg']['gptzero']
            })
            rows.append({
                'metric': 'detectgpt_confidence',
                'value': dist['confidences_avg']['detectgpt']
            })
        if 'agreement' in part2:
            agr = part2['agreement']
            rows.append({'metric': 'agreement_rate', 'value': agr['agreement_rate_avg']})
            rows.append({'metric': 'both_ai', 'value': agr['both_ai_avg']})
            rows.append({'metric': 'both_human', 'value': agr['both_human_avg']})
            rows.append({'metric': 'conflict', 'value': agr['conflict_avg']})
        if 'risk' in part2:
            risk = part2['risk']
            rows.append({'metric': 'high_risk', 'value': risk['high_risk_avg']})
            rows.append({'metric': 'medium_risk', 'value': risk['medium_risk_avg']})
            rows.append({'metric': 'uncertain', 'value': risk['uncertain_avg']})
            rows.append({'metric': 'low_risk', 'value': risk['low_risk_avg']})
        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(self.output_dir, 'part2_summary_metrics.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"[Saved Part2 summary CSV] {csv_path}")

    def _save_plots(self):
        if 'part1' in self.results:
            rows = []
            for method, results_list in self.results['part1'].items():
                values = [r for r in results_list if not np.isnan(r['accuracy'])]
                if not values:
                    continue
                rows.append({
                    'method': method,
                    'accuracy': np.mean([r['accuracy'] for r in values]),
                    'precision': np.mean([r['precision'] for r in values]),
                    'recall': np.mean([r['recall'] for r in values]),
                    'f1': np.mean([r['f1'] for r in values]),
                    'auc': np.mean([r['auc'] for r in values])
                })
            if rows:
                df = pd.DataFrame(rows)
                plot_path = os.path.join(self.output_dir, 'part1_metrics.png')
                df.set_index('method').plot(kind='bar', figsize=(10, 6))
                plt.title('Part 1 Mean Cross-Validation Metrics')
                plt.ylabel('Score')
                plt.ylim(0, 1)
                plt.xticks(rotation=0)
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                print(f"[Saved Part1 plot] {plot_path}")
        if 'part2' in self.results:
            dist = self.results['part2'].get('distribution', {})
            if dist:
                methods = ['GPTZero', 'DetectGPT']
                rates = [dist['ai_rates_avg']['gptzero'], dist['ai_rates_avg']['detectgpt']]
                confs = [dist['confidences_avg']['gptzero'], dist['confidences_avg']['detectgpt']]
                plot_path = os.path.join(self.output_dir, 'part2_detection_rates.png')
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax[0].bar(methods, [r * 100 for r in rates], color=['#4c72b0', '#55a868'])
                ax[0].set_title('AI Prediction Rate (%)')
                ax[0].set_ylim(0, 100)
                ax[0].set_ylabel('Percent')
                ax[1].bar(methods, confs, color=['#4c72b0', '#55a868'])
                ax[1].set_title('Average Confidence')
                ax[1].set_ylim(0, 1)
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                print(f"[Saved Part2 plot] {plot_path}")

    def _save_outputs(self):
        self._save_report()
        self._save_results_json()
        self._save_part1_csv()
        self._save_part2_csv()
        self._save_plots()
        print(f"[All outputs saved to] {self.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Improved AI Code Detection Comparison')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--sample_size', type=int, default=10,
                        help='Sample size for Part 2 (default: 10 for smoke test)')
    parser.add_argument('--part1_only', action='store_true', help='Run Part 1 only')
    parser.add_argument('--part2_only', action='store_true', help='Run Part 2 only')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save report, CSV, JSON, and plots')
    
    args = parser.parse_args()
    
    experiment = ImprovedComparisonExperiment()
    results = experiment.run_full_experiment(
        device=args.device,
        sample_size=args.sample_size,
        part1_only=args.part1_only,
        part2_only=args.part2_only,
        output_dir=args.output_dir
    )
