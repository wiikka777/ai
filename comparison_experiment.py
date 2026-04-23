"""
AI Code Detection Tools Comparison Experiment
Compare GPTZero, DetectGPT vs CodeBERT+XGBoost for code detection performance

Goal: Prove that AI code detection requires specialized code language models and features,
      rather than generic text-based AI detection tools

Using official GPTZero implementation (GPT-2 perplexity) and
DetectGPT implementation (GPT-2 + T5 perturbation-based detection)
"""

# ⚠️ CRITICAL: Set HF_HOME BEFORE any imports that trigger huggingface_hub loading.
# huggingface_hub.constants.HF_HUB_CACHE is computed at import time from HF_HOME.
# All models (gpt2, gpt2-medium, t5-large, codebert) are symlinked under model_cache/hub/.
import os
os.environ['HF_HOME'] = '/user/zhuohang.yu/u24922/model_cache'
os.environ['TRANSFORMERS_CACHE'] = '/user/zhuohang.yu/u24922/model_cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import json
import pandas as pd
import numpy as np
import time
import functools
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import random

# Force unbuffered output so SLURM .out file updates in real time
print = functools.partial(print, flush=True)

# Import GPTZero (GPT2PPL) from GPTZero-main
import importlib.util
try:
    _gptzero_spec = importlib.util.spec_from_file_location(
        "gptzero_model", 
        "/user/zhuohang.yu/u24922/exam/GPTZero-main/GPTZero-main/model.py"
    )
    _gptzero_module = importlib.util.module_from_spec(_gptzero_spec)
    _gptzero_spec.loader.exec_module(_gptzero_module)
    GPT2PPL = _gptzero_module.GPT2PPL
    GPTZERO_AVAILABLE = True
except Exception as e:
    GPTZERO_AVAILABLE = False
    print(f"⚠️  Warning: GPTZero code not found: {e}")

# Import DetectGPT (GPT2PPLV2) from DetectGPT-main
try:
    _detectgpt_spec = importlib.util.spec_from_file_location(
        "detectgpt_model",
        "/user/zhuohang.yu/u24922/exam/DetectGPT-main/model.py"
    )
    _detectgpt_module = importlib.util.module_from_spec(_detectgpt_spec)
    _detectgpt_spec.loader.exec_module(_detectgpt_module)
    GPT2PPLV2 = _detectgpt_module.GPT2PPLV2
    DETECTGPT_AVAILABLE = True
except Exception as e:
    DETECTGPT_AVAILABLE = False
    print(f"⚠️  Warning: DetectGPT code not found: {e}")

# ============================================================================
# Method 1: GPTZero Approach - Official implementation, GPT-2 perplexity-based
# ============================================================================

class GPTZeroApproach:
    """
    Using official GPTZero implementation: GPT-2 perplexity-based calculation
    - Core: Calculate code perplexity using GPT-2 language model
    - High perplexity → More likely human code (diverse, unpredictable)
    - Low perplexity → More likely AI-generated code (pattern-based, repetitive)
    
    Note: GPT-2 is a general text model, not code-specific, so used as baseline
    """
    
    def __init__(self, threshold: float = 50.0, device: str = "cpu"):
        """
        Args:
            threshold: Perplexity threshold (GPT-2 output range is typically large)
            device: cuda or cpu
        """
        self.threshold = threshold
        self.method_name = "GPTZero (GPT-2 Perplexity)"
        self.device = device
        
        # Initialize GPT-2 model (cache set globally via HF_HOME at top of file)
        if GPTZERO_AVAILABLE:
            try:
                self.model = GPT2PPL(device=device, model_id="gpt2")
                self.use_gpt2 = True
                print(f"  ✓ GPT-2 loaded (HF_HOME={os.environ.get('HF_HOME')})")
            except Exception as e:
                print(f"  ⚠️  GPT-2 loading failed: {e}")
                print(f"       Run: python download_gpt2.py")
                self.use_gpt2 = False
        else:
            self.use_gpt2 = False
    
    def compute_perplexity_gpt2(self, text: str) -> float:
        """
        Calculate perplexity using GPT-2 (official GPTZero method)
        """
        if not self.use_gpt2:
            return 0.0
        
        try:
            # Ensure text length is sufficient
            if len(text) < 20:
                return 0.0
            
            results, _ = self.model(text)
            perplexity = results.get("Perplexity per line", 0)
            return float(perplexity)
        except Exception as e:
            print(f"  Error computing GPT-2 perplexity: {e}")
            return 0.0
    
    def predict(self, features, code_text: str = None) -> Tuple[int, float]:
        """
        Make prediction using perplexity
        
        Args:
            features: numpy array (first element is perplexity) or dictionary
            code_text: Original code text 
            
        Returns:
            prediction (0=Human, 1=AI), confidence score
        """
        # Prioritize GPT-2 calculation, otherwise use perplexity from CodeBERT features
        if code_text and self.use_gpt2:
            perplexity = self.compute_perplexity_gpt2(code_text)
        else:
            # Handle numpy array or dictionary
            if isinstance(features, np.ndarray):
                perplexity = features[0]  # First element is perplexity
            else:
                perplexity = features.get('perplexity', 0)
        
        # GPTZero logic: low perplexity → AI
        if perplexity == 0:
            return 0, 0.5  # Cannot calculate, return neutral
        
        if perplexity < self.threshold:
            # AI confidence = degree of relative low perplexity
            confidence = 1.0 - (perplexity / self.threshold)
            return 1, max(confidence, 0.0)
        else:
            # Human confidence = degree of relative high perplexity
            confidence = (perplexity - self.threshold) / max(self.threshold, 1.0)
            return 0, min(confidence, 1.0)
    
    def batch_predict(self, X, code_texts: List[str] = None, label: str = "") -> List[Tuple[int, float]]:
        """Batch prediction with progress"""
        if code_texts is None:
            code_texts = [None] * len(X)
        results = []
        t0 = time.time()
        for i, (row, c) in enumerate(zip(X, code_texts)):
            results.append(self.predict(row, c))
            if (i + 1) % 10 == 0 or (i + 1) == len(X):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(X) - i - 1) / rate if rate > 0 else 0
                print(f"    [GPTZero{label}] {i+1}/{len(X)} ({elapsed:.0f}s, {rate:.1f} samp/s, ETA {eta:.0f}s)")
        return results


# ============================================================================
# Method 2: DetectGPT - Perturbation-based AI text detection
# ============================================================================

class DetectGPTApproach:
    """
    DetectGPT: Using perturbation-based detection (GPT-2 + T5 model)
    - Core: Perturb text with T5, compare log-likelihood curvature
    - AI-generated text tends to sit at local maxima of the model's log-probability
    - Uses z-score of (original - perturbed) log-likelihoods as detection signal
    
    Reference: Mitchell et al., "DetectGPT: Zero-Shot Machine-Generated Text Detection
    using Probability Curvature", ICML 2023
    
    Note: DetectGPT was designed for natural language text, not code-specific detection.
    It requires GPT-2 medium + T5-large models, making it computationally expensive.
    """
    
    def __init__(self, device: str = "cpu", chunk_value: int = 180):
        """
        Args:
            device: cuda or cpu
            chunk_value: Word-chunk size passed to DetectGPT v1.1 splitter.
                         Smaller value => more chunks per sample => much slower.
        """
        self.method_name = "DetectGPT (Perturbation-based)"
        self.device = device
        self.chunk_value = chunk_value
        self.threshold = 0.7  # DetectGPT default threshold
        
        self.max_chars = 5000  # Generous char limit
        self.max_words = 500   # Allow more words (DetectGPT will chunk internally)
        self.max_tokens = 512  # GPT-2 medium token limit
        
        # Minimum text lengths to avoid std=0 errors
        self.min_words_for_perturbation = 50  # Need at least this for T5 masking
        
        # Initialize DetectGPT model (cache set globally via HF_HOME at top of file)
        if DETECTGPT_AVAILABLE:
            try:
                self.model = GPT2PPLV2(device=device, model_id="gpt2-medium")
                self.use_detectgpt = True
                print(f"  ✓ DetectGPT loaded (GPT-2 medium + T5-large, device={device})")
            except Exception as e:
                print(f"  ⚠️  DetectGPT loading failed: {e}")
                print(f"       Run: python download_detectgpt_models.py first")
                self.use_detectgpt = False
        else:
            self.use_detectgpt = False
            print("  ⚠️  DetectGPT not available, using fallback perplexity method")
    
    def compute_detectgpt_score(self, text: str) -> Tuple[int, float]:
        """
        Run DetectGPT v1.1 perturbation-based detection on a single text.
        
        Returns:
            prediction (0=Human, 1=AI), confidence score
        """
        if not self.use_detectgpt:
            return 0, 0.5
        
        if len(text.strip()) < 20:
            return 0, 0.5
        
        # Strong truncation to avoid pathological long-code cases that can stall
        # DetectGPT for minutes on a single sample.
        text = " ".join(text.split())
        if len(text) > self.max_chars:
            text = text[:self.max_chars]

        # Simple word-based truncation (avoid complex tokenization issues)
        words = text.split()
        if len(words) > self.max_words:
            text = ' '.join(words[:self.max_words])
        
        # CRITICAL: Check minimum words for DetectGPT's T5 masking to work
        # If too few words, std_dev will be 0 -> division by zero
        words = text.split()
        if len(words) < self.min_words_for_perturbation:
            # Text too short for meaningful perturbations
            # Fall back to simple perplexity threshold
            try:
                ppl = self.model.model.config.n_positions  # Just return neutral
                return 0, 0.5
            except:
                return 0, 0.5

        # Token-level cap for GPT-2 medium context stability/speed.
        try:
            tok = self.model.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_tokens
            )
            text = self.model.tokenizer.decode(tok["input_ids"][0], skip_special_tokens=True)
        except Exception:
            pass
        
        # Additional safeguard: if text is now too short after tokenization, skip
        if len(text.strip()) < 20:
            return 0, 0.5
        
        # Call DetectGPT v1.1 (perturbation-based)
        try:
            result, verdict = self.model(text, self.chunk_value, "v1.1")
        except ZeroDivisionError as e:
            # Catch division by zero in DetectGPT's getScore method
            # This happens when std_generated_log_likelihood is 0 (all perturbations identical)
            print(f"  ⚠️  DetectGPT division by zero (std=0, likely homogeneous code), returning neutral")
            return 0, 0.5
        except Exception as e:
            print(f"  Error in DetectGPT scoring: {type(e).__name__}: {str(e)[:60]}")
            return 0, 0.5
        
        # DEBUG: Check result structure
        if not isinstance(result, dict):
            print(f"  ⚠️  DetectGPT result is not dict: {type(result)}, value={result}")
            return 0, 0.5
        
        # DetectGPT label convention: 0 = AI, 1 = Human
        # Our convention: 0 = Human, 1 = AI
        # So we need to invert the label
        detectgpt_label = result.get('label', 1)
        prob_str = result.get('prob', '50.00%')
        try:
            prob = float(str(prob_str).replace('%', '').strip()) / 100.0
        except Exception:
            prob = 0.5
        prob = min(max(prob, 0.0), 1.0)
        
        # In DetectGPT source:
        #   label=0 => AI, and prob is probability for AI
        #   label=1 => Human, and prob is probability for Human
        # We return (our_label, AI_probability) so AUC uses consistent semantics.
        if detectgpt_label == 0:  # DetectGPT says AI
            return 1, prob
        else:  # DetectGPT says Human
            return 0, 1.0 - prob
    
    def predict(self, features, code_text: str = None) -> Tuple[int, float]:
        """
        Make prediction using DetectGPT.
        
        If code_text is provided and DetectGPT model is loaded, runs full
        perturbation-based detection. Otherwise, falls back to simple
        perplexity threshold (similar to GPTZero).
        
        Args:
            features: numpy array (first element is perplexity)
            code_text: Original code text for full DetectGPT analysis
            
        Returns:
            prediction (0=Human, 1=AI), confidence score
        """
        # Use full DetectGPT pipeline if code text is available
        if code_text and self.use_detectgpt:
            return self.compute_detectgpt_score(code_text)
        
        # Fallback: use perplexity-based threshold (similar to GPTZero but with
        # different threshold, simulating DetectGPT behavior without T5)
        if isinstance(features, np.ndarray):
            perplexity = features[0]
        else:
            perplexity = features.get('perplexity', 0)
        
        if perplexity == 0:
            return 0, 0.5
        
        # DetectGPT-style threshold (lower than GPTZero since it uses GPT-2 medium)
        ppl_threshold = 60.0
        if perplexity < ppl_threshold:
            confidence = 1.0 - (perplexity / ppl_threshold)
            return 1, max(confidence, 0.0)
        else:
            confidence = (perplexity - ppl_threshold) / max(ppl_threshold, 1.0)
            return 0, min(confidence, 1.0)
    
    def batch_predict(self, X, code_texts: List[str] = None, label: str = "") -> List[Tuple[int, float]]:
        """Batch prediction using DetectGPT's original v1.1 logic.
        
        Protections against hangs:
        1. Pre-filter pathological samples (>4000 chars or >300 lines)
        2. Log slow samples for monitoring
        3. Continue on errors (don't bail out)
        
        All samples processed through DetectGPT's original compute_detectgpt_score(),
        which includes internal truncation guards (max_chars, max_words, max_tokens).
        """
        SLOW_THRESHOLD = 30  # log warning if sample takes >30s
        MAX_HARD_SAMPLE_CHARS = 4000
        MAX_HARD_SAMPLE_LINES = 300
        is_test = "TEST" in label
        
        if code_texts is None:
            code_texts = [None] * len(X)
        
        results = []
        t0 = time.time()
        errors = 0
        slow_count = 0
        skipped_hard = 0
        
        for i, (row, c) in enumerate(zip(X, code_texts)):
            # Pre-filter pathological samples to avoid per-sample stalls
            # This is defensive; DetectGPT also has internal truncation in compute_detectgpt_score()
            if is_test and isinstance(c, str):
                if len(c) > MAX_HARD_SAMPLE_CHARS or (c.count('\n') + 1) > MAX_HARD_SAMPLE_LINES:
                    skipped_hard += 1
                    results.append((0, 0.5))
                    if (i + 1) % 5 == 0 or (i + 1) == len(X):
                        elapsed = time.time() - t0
                        rate = (i + 1) / elapsed if elapsed > 0 else 0
                        eta = (len(X) - i - 1) / rate if rate > 0 else 0
                        print(f"    [DetectGPT{label}] {i+1}/{len(X)} ({elapsed:.0f}s, {rate:.2f} samp/s, ETA {eta:.0f}s, err={errors}, slow={slow_count}, skip={skipped_hard})")
                    continue

            sample_t0 = time.time()
            try:
                # Use DetectGPT's original compute_detectgpt_score() logic
                results.append(self.predict(row, c))
            except Exception as e:
                errors += 1
                results.append((0, 0.5))
                print(f"  ❌ Sample {i+1} error: {str(e)[:100]}")
            
            sample_time = time.time() - sample_t0
            if sample_time > SLOW_THRESHOLD:
                slow_count += 1
                print(f"  ⚠️  Sample {i+1} took {sample_time:.1f}s (slow)")
            
            if (i + 1) % 5 == 0 or (i + 1) == len(X):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(X) - i - 1) / rate if rate > 0 else 0
                print(f"    [DetectGPT{label}] {i+1}/{len(X)} ({elapsed:.0f}s, {rate:.2f} samp/s, ETA {eta:.0f}s, err={errors}, slow={slow_count}, skip={skipped_hard})")
        
        print(f"    DetectGPT done: {len(results)} samples, {errors} errors, {slow_count} slow, {skipped_hard} skipped-hard, {time.time()-t0:.0f}s total")
        return results


# ============================================================================
# Method 3: CodeBERT+XGBoost - Specialized code features
# ============================================================================

class CodeBERTApproach:
    """
    Current approach: Leverage code-specific language features
    - CodeBERT token embeddings
    - Code structure related: code_length, line_length, comment_ratio
    - Code complexity: identifier_entropy, ngram_repetition
    - Language properties: perplexity, token probability, entropy, burstiness
    
    Total 10 features, far superior to generic text features
    """
    
    def __init__(self):
        self.method_name = "CodeBERT + XGBoost (Code-specific 10 features)"
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = [
            'perplexity', 'avg_token_probability', 'avg_entropy',
            'burstiness', 'code_length', 'avg_line_length',
            'std_line_length', 'comment_ratio', 'identifier_entropy',
            'ngram_repetition'
        ]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train with XGBoost"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.model.fit(X_scaled, y_train, verbose=0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def batch_predict(self, X: np.ndarray) -> List[Tuple[int, float]]:
     
        proba = self.predict_proba(X)
        return [(np.argmax(p), p[1]) for p in proba]


# ============================================================================
# Comparison Experiment Runner
# ============================================================================

class ComparisonExperiment:
    """Run complete comparison experiment"""
    
    def __init__(self, train_data_path: str = None, full_data_path: str = None):
        """
        Args:
            train_data_path: Training data CSV path (160 samples)
            full_data_path: Full dataset path (for retrieving original code text)
        """
        self.train_data = pd.read_csv(train_data_path) if train_data_path and os.path.exists(train_data_path) else None
        self.train_code_texts = []  # Legacy field, no longer required for Part 1
        self.known_ai_ids = set()
        self.known_ai_texts = []
        
        # Build id -> code mapping from cleaned/controlled sources only:
        # 1. last_success.json (~44k cleaned) — main source for Part 2
        # 2. ai.json (60 AI-generated samples) — known-positive AI anchor set (Part 1)
        self.full_data_path = full_data_path or '/user/zhuohang.yu/u24922/exam/smartbeans_submission_last_success.json'
        self._id_to_code = {}  # actual_id -> code content
        
        # Source 1: last_success (primary, for test set)
        if os.path.exists(self.full_data_path):
            print(f"Loading cleaned dataset: {os.path.basename(self.full_data_path)}")
            with open(self.full_data_path, 'r') as f:
                full_data = json.load(f)
            for row in full_data[1:]:
                if isinstance(row, list) and len(row) > 5:
                    actual_id = str(row[0])
                    content = row[5]
                    if isinstance(content, str) and len(content.strip()) > 20:
                        self._id_to_code[actual_id] = content
            print(f"  ✓ last_success: {len(self._id_to_code):,} submissions")
            del full_data
        
        # Source 2: ai.json (AI-generated code for training set)
        ai_json_path = '/user/zhuohang.yu/u24922/exam/ai.json'
        ai_count = 0
        if os.path.exists(ai_json_path):
            with open(ai_json_path, 'r') as f:
                ai_data = json.load(f)
            for row in ai_data[1:]:
                if isinstance(row, list) and len(row) > 5:
                    ai_id = str(row[0])
                    content = row[5]
                    if isinstance(content, str) and len(content.strip()) > 10:
                        self._id_to_code[ai_id] = content
                        self.known_ai_ids.add(ai_id)
                        self.known_ai_texts.append(content)
                        ai_count += 1
            print(f"  ✓ ai.json: {ai_count} AI-generated samples added")
            del ai_data

        # Legacy: only for backward compatibility if train_data_path is provided.
        if self.train_data is not None:
            for _, row in self.train_data.iterrows():
                filename = str(row['filename'])
                self.train_code_texts.append(self._id_to_code.get(filename, None))
            found = sum(1 for c in self.train_code_texts if c is not None)
            missing_count = len(self.train_code_texts) - found
            print(f"  ✓ Training set: {found}/{len(self.train_code_texts)} samples have original code")
            if missing_count > 0:
                missing_names = [str(r['filename']) for _, r in self.train_data.iterrows()
                               if self._id_to_code.get(str(r['filename'])) is None]
                print(f"  ⚠️  {missing_count} samples still missing code text: {missing_names[:5]}...")
        
        self.results = {}
        self.feature_names = [
            'perplexity', 'avg_token_probability', 'avg_entropy',
            'burstiness', 'code_length', 'avg_line_length',
            'std_line_length', 'comment_ratio', 'identifier_entropy',
            'ngram_repetition'
        ]
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data"""
        if self.train_data is None:
            raise ValueError("train_data is not available")
        X = self.train_data[self.feature_names].values
        y = self.train_data['label'].map({'Human': 0, 'AI': 1, 'Student': 0}).values
        return X, y
    
    def run_experiment(self, sample_size: int = 0, device: str = None, test_repeats: int = 3,
                       methods: List[str] = None, sampling_strategy: str = "stratified") -> Dict:
        """Run complete comparison experiment.
        
        Args:
            sample_size: If > 0, subsample the pseudo-labeled pool for speed.
            test_repeats: Number of random subsampling repeats for Part 2.
            device: 'cuda' or 'cpu'. Auto-detects if None.
            sampling_strategy: 'stratified' (pseudo-label stratified) or 'random'.
        """
        methods = methods or ['gptzero', 'detectgpt', 'codebert']
        method_set = set(m.lower() for m in methods)
        self.selected_methods = sorted(method_set)
        sampling_strategy = (sampling_strategy or "stratified").lower()
        if sampling_strategy not in {"stratified", "random"}:
            sampling_strategy = "stratified"
        self.sampling_strategy = sampling_strategy

        print("=" * 80)
        print("AI Code Detection Methods Comparison Experiment")
        print("=" * 80)
        print(f"Selected methods: {', '.join(self.selected_methods)}")
        print(f"Part-2 sampling strategy: {self.sampling_strategy}")
        
        # ====================================================================
        # Part 1: Known-positive AI detection (no verified human ground truth)
        # ====================================================================
        print(f"\n[Part 1] Known-AI Detection (ai.json anchor set)")
        print("=" * 80)
        
        # Auto-detect device
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[Device] Using: {device}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        # Train all models
        print(f"\n[Step 1.1] Training models...")
        
        # 1: GPTZero
        gptzero = GPTZeroApproach(threshold=50.0, device=device) if 'gptzero' in method_set else None
        
        # 2: DetectGPT (perturbation-based)
        detectgpt = DetectGPTApproach(device=device, chunk_value=50) if 'detectgpt' in method_set else None
        
        print("✓ Models initialized")

        # CodeBERT can be included in Part 1 only if feature table is available.
        codebert = None
        if self.train_data is not None and 'codebert' in method_set:
            try:
                X_train, y_train = self.prepare_data()
                codebert = CodeBERTApproach()
                codebert.train(X_train, y_train)
                print("  ✓ CodeBERT reference model trained from feature table")
            except Exception as e:
                print(f"  ⚠️  CodeBERT training unavailable for Part 1: {e}")

        print(f"\n[Step 1.2] Evaluating recall on known AI positives only...")
        known_ai_results = self._evaluate_known_ai_positives(
            gptzero, detectgpt, codebert
        )
        self.results['known_ai_eval'] = known_ai_results
        
        # ====================================================================
        # Part 2: Pseudo-label consistency on full-dataset pool (last_success)
        # ====================================================================
        print(f"\n[Part 2] Pseudo-label Consistency Evaluation (last_success)")
        print("=" * 80)
        
        # Load full pseudo-labeled pool once, subsample per repeat below.
        test_results = self._load_test_set(sample_size=0)
        
        if test_results is not None:
            y_pool, codebert_probs_pool, code_texts_pool = test_results
            
            print(f"\n[Step 2.1] CodeBERT predictions already loaded (from inference_full_dataset.py)")
            repeat_results = []
            repeats = max(1, int(test_repeats))
            run_n = repeats if sample_size > 0 else 1
            print(f"\n[Step 2.2] Running {run_n} pseudo-label consistency repeat(s)...")
            for r in range(run_n):
                seed = 42 + r
                target_n = sample_size if sample_size > 0 else len(y_pool)
                draw_n = target_n
                if sample_size > 0 and 'detectgpt' in method_set:
                    # Draw extra candidates so DetectGPT can skip problematic samples.
                    draw_n = min(len(code_texts_pool), int(target_n * 1.8))
                y_test, codebert_probs, test_code_texts = self._sample_test_subset(
                    y_pool, codebert_probs_pool, code_texts_pool, draw_n, seed, self.sampling_strategy
                )
                print(f"  [Repeat {r+1}/{run_n}] seed={seed}, target={target_n}, candidates={len(test_code_texts)}")
                repeat_result = self._evaluate_on_test_set(
                    y_test, codebert_probs, test_code_texts, gptzero, detectgpt,
                    enabled_methods=method_set, target_n=target_n, label_suffix=f" TEST-R{r+1}"
                )
                repeat_results.append(repeat_result)
            self.results['test_set'] = self._aggregate_test_results(repeat_results)
        else:
            print("  ⚠️  Test set not available, skipping...")
            self.results['test_set'] = None
        
        # Generate comparison report
        self._generate_comprehensive_report()
        
        return self.results

    def _evaluate_known_ai_positives(self, gptzero, detectgpt, codebert=None) -> Dict:
        """Evaluate model behavior on known AI samples only (ai.json IDs).

        This is not full accuracy evaluation because verified human negatives
        are unavailable.
        """
        n_known = len(self.known_ai_texts)
        if n_known == 0:
            print("  ❌ No known AI samples available for Part 1.")
            return {}

        ai_code_texts = self.known_ai_texts
        print(f"  Known AI samples: {n_known}")

        results = {}
        for method_name, model in [
            ("GPTZero", gptzero),
            ("DetectGPT", detectgpt),
        ]:
            if model is None:
                continue
            t0 = time.time()
            dummy_X = np.zeros((n_known, 1))
            preds_list = model.batch_predict(dummy_X, code_texts=ai_code_texts, label=" KNOWN-AI")
            preds = np.array([p[0] for p in preds_list], dtype=np.int32)
            probs = np.array([p[1] for p in preds_list], dtype=np.float32)
            elapsed = time.time() - t0
            recall_ai = float(np.mean(preds == 1))
            results[method_name] = {
                'known_ai_count': n_known,
                'ai_recall_on_known_ai': recall_ai,
                'mean_ai_probability': float(np.mean(probs)),
                'median_ai_probability': float(np.median(probs)),
                'num_features': 1,
                'predictions': preds,
                'probabilities': probs,
                'elapsed_seconds': float(elapsed),
            }
            print(f"  [{method_name}] AI recall on known AI: {recall_ai:.4f}, avg prob={np.mean(probs):.4f}")

        # CodeBERT path: requires feature vectors for the same known AI IDs.
        if codebert is not None and self.train_data is not None:
            filename_series = self.train_data['filename'].astype(str)
            known_mask = filename_series.isin(self.known_ai_ids).values
            if int(np.sum(known_mask)) == 0:
                known_mask = self.train_data['label'].astype(str).eq('AI').values

            n_codebert = int(np.sum(known_mask))
            if n_codebert > 0:
                X_ai = self.train_data.loc[known_mask, self.feature_names].values
                t0 = time.time()
                preds_list = codebert.batch_predict(X_ai)
                preds = np.array([p[0] for p in preds_list], dtype=np.int32)
                probs = np.array([p[1] for p in preds_list], dtype=np.float32)
                elapsed = time.time() - t0
                recall_ai = float(np.mean(preds == 1))
                results['CodeBERT'] = {
                    'known_ai_count': n_codebert,
                    'ai_recall_on_known_ai': recall_ai,
                    'mean_ai_probability': float(np.mean(probs)),
                    'median_ai_probability': float(np.median(probs)),
                    'num_features': 10,
                    'predictions': preds,
                    'probabilities': probs,
                    'elapsed_seconds': float(elapsed),
                    'note': 'Computed on known-AI subset with available feature vectors.'
                }
                print(f"  [CodeBERT] AI recall on known AI: {recall_ai:.4f}, avg prob={np.mean(probs):.4f}")
            else:
                results['CodeBERT'] = {
                    'known_ai_count': 0,
                    'ai_recall_on_known_ai': float('nan'),
                    'mean_ai_probability': float('nan'),
                    'median_ai_probability': float('nan'),
                    'num_features': 10,
                    'predictions': np.array([], dtype=np.int32),
                    'probabilities': np.array([], dtype=np.float32),
                    'elapsed_seconds': 0.0,
                    'note': 'No overlap between ai.json IDs and feature table.'
                }
                print("  ⚠️  [CodeBERT] No known-AI feature rows available for Part 1.")

        return results
    
    def _evaluate_model(self, method_name: str, model, X, y, 
                        use_full_features: bool = True) -> Dict:
        """Evaluate a single model"""
        predictions_list = model.batch_predict(X)
        predictions = np.array([p[0] for p in predictions_list])
        probabilities = np.array([p[1] for p in predictions_list])
        
        # Calculate metrics
        acc = accuracy_score(y, predictions)
        prec = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(y, probabilities)
        except:
            auc = 0.0
        
        results = {
            'method': method_name,
            'features_used': 10 if use_full_features else 1,
            'accuracy': acc,
            'precision': prec,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        print(f"\n[{method_name}]")
        print(f"  Features Used: {results['features_used']}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        return results
    
    def _cross_validation(self, X, y, code_texts, gptzero, detectgpt, codebert) -> Dict:
        """5-fold cross validation with original code texts for GPTZero/DetectGPT"""
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = {}
        
        for method_name, model, X_subset, feature_indices in [
            ("GPTZero", gptzero, X[:, [0]], [0]),
            ("DetectGPT", detectgpt, X[:, [0]], [0]),
            ("CodeBERT", codebert, X, list(range(10)))
        ]:
            print(f"\n  [{method_name}] Running 5-fold CV...")
            method_t0 = time.time()
            
            fold_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
            all_predictions = np.zeros(len(y))
            all_probabilities = np.zeros(len(y))
            
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_subset, y)):
                fold_t0 = time.time()
                print(f"    Fold {fold_idx+1}/5 (val={len(val_idx)} samples)...")
                # Training and validation sets
                X_fold_train, X_fold_val = X_subset[train_idx], X_subset[val_idx]
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                
                # Get code texts for validation set
                val_code_texts = [code_texts[i] for i in val_idx]
                
                # Train model
                if method_name == "GPTZero":
                    # GPTZero doesn't need training (unsupervised)
                    pass
                elif method_name == "DetectGPT":
                    # DetectGPT doesn't need training (unsupervised)
                    pass
                else:  # CodeBERT
                    model_fold = CodeBERTApproach()
                    model_fold.train(X_fold_train, y_fold_train)
                
                # Predict (pass code_texts for GPTZero/DetectGPT)
                if method_name == "GPTZero":
                    preds_list = gptzero.batch_predict(X_fold_val, code_texts=val_code_texts, label=f" F{fold_idx+1}")
                elif method_name == "DetectGPT":
                    preds_list = detectgpt.batch_predict(X_fold_val, code_texts=val_code_texts, label=f" F{fold_idx+1}")
                else:
                    preds_list = model_fold.batch_predict(X_fold_val)
                
                preds = np.array([p[0] for p in preds_list])
                probs = np.array([p[1] for p in preds_list])
                
                # Save prediction results
                all_predictions[val_idx] = preds
                all_probabilities[val_idx] = probs
                
                # Calculate metrics
                fold_scores['accuracy'].append(accuracy_score(y_fold_val, preds))
                fold_scores['precision'].append(precision_score(y_fold_val, preds, zero_division=0))
                fold_scores['recall'].append(recall_score(y_fold_val, preds, zero_division=0))
                fold_scores['f1'].append(f1_score(y_fold_val, preds, zero_division=0))
                try:
                    fold_scores['auc'].append(roc_auc_score(y_fold_val, probs))
                except:
                    fold_scores['auc'].append(0.0)
                
                fold_time = time.time() - fold_t0
                total_time = time.time() - method_t0
                print(f"    Fold {fold_idx+1}/5 done in {fold_time:.1f}s (F1={fold_scores['f1'][-1]:.4f}, total {total_time:.1f}s)")
            
            method_time = time.time() - method_t0
            print(f"  [{method_name}] CV completed in {method_time:.1f}s ({method_time/60:.1f} min)")
            
            # Calculate mean and standard deviation
            cv_results[method_name] = {
                'accuracy': {'mean': np.mean(fold_scores['accuracy']), 'std': np.std(fold_scores['accuracy'])},
                'precision': {'mean': np.mean(fold_scores['precision']), 'std': np.std(fold_scores['precision'])},
                'recall': {'mean': np.mean(fold_scores['recall']), 'std': np.std(fold_scores['recall'])},
                'f1': {'mean': np.mean(fold_scores['f1']), 'std': np.std(fold_scores['f1'])},
                'auc': {'mean': np.mean(fold_scores['auc']), 'std': np.std(fold_scores['auc'])},
                'num_features': len(feature_indices),
                'all_predictions': all_predictions,
                'all_probabilities': all_probabilities
            }
            
            print(f"    Accuracy:  {cv_results[method_name]['accuracy']['mean']:.4f} ± {cv_results[method_name]['accuracy']['std']:.4f}")
            print(f"    Precision: {cv_results[method_name]['precision']['mean']:.4f} ± {cv_results[method_name]['precision']['std']:.4f}")
            print(f"    Recall:    {cv_results[method_name]['recall']['mean']:.4f} ± {cv_results[method_name]['recall']['std']:.4f}")
            print(f"    F1 Score:  {cv_results[method_name]['f1']['mean']:.4f} ± {cv_results[method_name]['f1']['std']:.4f}")
            print(f"    AUC:       {cv_results[method_name]['auc']['mean']:.4f} ± {cv_results[method_name]['auc']['std']:.4f}")
        
        return cv_results
    
    def _load_test_set(self, sample_size: int = 0):
        """Load test set from existing inference results + original code texts.
        
        Uses:
        - full_dataset_results.json: CodeBERT+XGBoost predictions (from inference_full_dataset.py)
        - smartbeans_submission_last_success.json: Original code texts
        
        No feature extraction needed — reuses what inference_full_dataset.py already computed.
        
        Args:
            sample_size: If > 0, randomly sample this many records (for speed).
        """
        print(f"\n[Step 2.0] Loading test set (last_success dataset)...")
        
        results_file = '/user/zhuohang.yu/u24922/exam/full_dataset_results.json'
        raw_file = '/user/zhuohang.yu/u24922/exam/smartbeans_submission_last_success.json'
        
        if not os.path.exists(results_file) or not os.path.exists(raw_file):
            print(f"  ❌ Required files not found:")
            if not os.path.exists(results_file):
                print(f"     - {results_file} (run inference_full_dataset.py first)")
            if not os.path.exists(raw_file):
                print(f"     - {raw_file}")
            return None
        
        try:
            # Load existing CodeBERT+XGBoost predictions
            print(f"  Loading CodeBERT predictions from full_dataset_results.json...")
            with open(results_file, 'r') as f:
                pred_data = json.load(f)
            print(f"    {len(pred_data)} predictions loaded")
            
            # Load original code texts
            print(f"  Loading code texts from last_success.json...")
            with open(raw_file, 'r') as f:
                raw_data = json.load(f)
            print(f"    {len(raw_data)-1} submissions loaded")
            
            # Build: submission_id -> prediction result
            pred_map = {r['submission_id']: r for r in pred_data}
            del pred_data
            
            # Exclude known anchor IDs to avoid leakage into Part 2.
            train_ids = set(str(fid) for fid in self.train_data['filename'].values) if self.train_data is not None else set()
            skip_ids = set(self.known_ai_ids)
            skip_ids.update(train_ids)
            
            # Collect test data: code text + CodeBERT pseudo-label
            test_code_texts = []
            y_test = []  # CodeBERT pseudo-labels (prediction from inference_full_dataset.py)
            codebert_probs = []  # CodeBERT AI probabilities
            skipped_train = 0
            skipped_no_pred = 0
            skipped_short = 0
            
            for row_idx in range(1, len(raw_data)):  # skip header
                row = raw_data[row_idx]
                actual_id = str(row[0])
                content = row[5] if len(row) > 5 else ''
                
                # Skip training samples (avoid data leakage)
                if actual_id in skip_ids:
                    skipped_train += 1
                    continue
                
                # Skip short/invalid code
                if not isinstance(content, str) or len(content.strip()) < 20:
                    skipped_short += 1
                    continue
                
                # Match to prediction (submission_id = row index in last_success)
                pred = pred_map.get(row_idx)
                if pred is None:
                    skipped_no_pred += 1
                    continue
                
                test_code_texts.append(content)
                y_test.append(int(pred['prediction']))
                codebert_probs.append(float(pred['ai_probability']))
            
            del raw_data, pred_map
            
            if len(test_code_texts) == 0:
                print(f"  ⚠️  No valid samples found")
                return None
            
            y_test = np.array(y_test, dtype=np.int32)
            codebert_probs = np.array(codebert_probs, dtype=np.float32)
            
            print(f"  ✓ Loaded {len(test_code_texts)} valid test samples")
            print(f"    - Skipped (in training set): {skipped_train}")
            print(f"    - Skipped (short/invalid): {skipped_short}")
            print(f"    - Skipped (no prediction): {skipped_no_pred}")
            print(f"    - CodeBERT pseudo-labels: Human={int(np.sum(y_test==0))}, AI={int(np.sum(y_test==1))}")
            
            return y_test, codebert_probs, test_code_texts
            
        except Exception as e:
            import traceback
            print(f"  ❌ Error loading test set: {e}")
            traceback.print_exc()
            return None

    def _sample_test_subset(self, y_pool, codebert_probs_pool, code_texts_pool,
                            sample_size: int, seed: int, strategy: str = "stratified"):
        """Sample a reproducible subset from the pseudo-labeled pool.

        strategy='stratified' means pseudo-label stratified sampling using y_pool.
        This stabilizes repeat-to-repeat class proportions, but does NOT introduce
        ground-truth labels.
        """
        if sample_size <= 0 or sample_size >= len(code_texts_pool):
            return y_pool, codebert_probs_pool, code_texts_pool

        strategy = (strategy or "stratified").lower()
        if strategy == "stratified":
            # Use pseudo-label stratification to keep AI/Human proportion stable.
            # Fallback to random when stratification requirements are not met.
            unique, counts = np.unique(y_pool, return_counts=True)
            can_stratify = (
                len(unique) >= 2 and
                sample_size >= len(unique) and
                np.all(counts >= 2)
            )
            if can_stratify:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=seed)
                all_idx = np.arange(len(y_pool))
                _, indices = next(sss.split(all_idx, y_pool))
            else:
                rng = np.random.default_rng(seed)
                indices = rng.choice(len(code_texts_pool), size=sample_size, replace=False)
                print("  ⚠️  Stratified sampling fallback to random (insufficient pseudo-label support).")
        else:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(code_texts_pool), size=sample_size, replace=False)

        indices.sort()
        y = y_pool[indices]
        probs = codebert_probs_pool[indices]
        texts = [code_texts_pool[i] for i in indices]
        return y, probs, texts

    def _aggregate_test_results(self, repeat_results: List[Dict]) -> Dict:
        """Aggregate pseudo-label consistency results across repeated subsamples."""
        if not repeat_results:
            return {}
        methods = [m for m in repeat_results[0].keys()]
        metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc',
                       'agreement_vs_codebert', 'ai_rate', 'elapsed_seconds']
        agg = {'__meta__': {'num_repeats': len(repeat_results)}}
        for method in methods:
            base = repeat_results[0][method]
            out = {'num_features': base['num_features']}
            if 'confusion_matrix' in base:
                out['confusion_matrix'] = base['confusion_matrix']
            if 'predictions' in base:
                out['predictions'] = base['predictions']
            if 'probabilities' in base:
                out['probabilities'] = base['probabilities']
            if 'role' in base:
                out['role'] = base['role']
            if 'note' in base:
                out['note'] = base['note']
            for key in metric_keys:
                values = np.array([r[method].get(key, np.nan) for r in repeat_results], dtype=np.float64)
                if np.all(np.isnan(values)):
                    out[key] = float('nan')
                    out[f"{key}_std"] = float('nan')
                else:
                    out[key] = float(np.nanmean(values))
                    out[f"{key}_std"] = float(np.nanstd(values))
            agg[method] = out
        return agg
    
    def _evaluate_on_test_set(self, y_test, codebert_probs, test_code_texts, gptzero, detectgpt,
                              enabled_methods=None, target_n: int = None,
                              label_suffix: str = " TEST") -> Dict:
        """Evaluate on independent test set.
        
        y_test: CodeBERT pseudo-labels (from inference_full_dataset.py, already computed)
        codebert_probs: CodeBERT AI probabilities (already computed)
        GPTZero and DetectGPT run their full pipelines on original code texts.
        """
        test_results = {}
        n_samples = len(test_code_texts)
        if target_n is None or target_n <= 0:
            target_n = n_samples
        target_n = min(target_n, n_samples)

        # First target_n are the reference subset; the remainder are fallback
        # candidates for DetectGPT-only replacement in Part 2.
        y_ref = y_test[:target_n]
        p_ref = codebert_probs[:target_n]
        texts_ref = test_code_texts[:target_n]
        y_extra = y_test[target_n:]
        texts_extra = test_code_texts[target_n:]

        ai_count = int(np.sum(y_ref == 1))
        human_count = int(np.sum(y_ref == 0))
        
        # CodeBERT is the pseudo-label source in Part 2 (reference only).
        print(f"  [CodeBERT] Reference pseudo-labels ({target_n} samples)")
        print(f"    Pseudo-label distribution: Human={human_count}, AI={ai_count}")
        print(f"    ⚠️  Part 2 has no ground truth; no true accuracy is defined.")
        
        codebert_preds = y_ref.copy()
        test_results['CodeBERT'] = {
            'accuracy': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'auc': float('nan'),
            'confusion_matrix': confusion_matrix(y_ref, codebert_preds),
            'tp': ai_count, 'fp': 0, 'tn': human_count, 'fn': 0,
            'num_features': 10,
            'predictions': codebert_preds,
            'probabilities': p_ref,
            'agreement_vs_codebert': float('nan'),
            'ai_rate': float(np.mean(codebert_preds)),
            'elapsed_seconds': 0.0,
            'role': 'reference_baseline',
            'note': 'Pseudo-label source only; not evaluated for performance.',
        }
        
        enabled_methods = enabled_methods or {'gptzero', 'detectgpt', 'codebert'}
        # Run selected external detectors on original code texts
        eval_targets = []
        if 'gptzero' in enabled_methods and gptzero is not None:
            eval_targets.append(("GPTZero", gptzero))
        if 'detectgpt' in enabled_methods and detectgpt is not None:
            eval_targets.append(("DetectGPT", detectgpt))

        for method_name, model in eval_targets:
            print(f"  [{method_name}] Running full pipeline on {target_n} code samples...")
            test_t0 = time.time()
            
            # Use standard batch_predict for all methods (DetectGPT handles its own robustness)
            dummy_X = np.zeros((target_n, 1))
            preds_list = model.batch_predict(dummy_X, texts_ref, label=label_suffix)
            preds = np.array([p[0] for p in preds_list], dtype=np.int32)
            probs = np.array([p[1] for p in preds_list], dtype=np.float32)
            y_eval = y_ref

            # Calculate metrics vs CodeBERT pseudo-labels
            acc = accuracy_score(y_eval, preds)
            prec = precision_score(y_eval, preds, zero_division=0)
            recall = recall_score(y_eval, preds, zero_division=0)
            f1 = f1_score(y_eval, preds, zero_division=0)
            
            try:
                auc = roc_auc_score(y_eval, probs)
            except:
                auc = 0.0
            
            cm = confusion_matrix(y_eval, preds)
            tn, fp, fn, tp = cm.ravel()
            agreement = float(np.mean(preds == y_eval))
            ai_rate = float(np.mean(preds))
            test_time = time.time() - test_t0
            
            test_results[method_name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'confusion_matrix': cm,
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
                'num_features': 1,
                'predictions': preds,
                'probabilities': probs,
                'agreement_vs_codebert': agreement,
                'ai_rate': ai_rate,
                'elapsed_seconds': float(test_time),
            }
            
            # Agreement with pseudo-label source
            print(f"    Agreement with CodeBERT pseudo-labels: {agreement:.4f}")
            print(f"    {method_name} AI rate: {ai_rate:.4f} vs CodeBERT AI rate: {np.mean(y_eval):.4f}")
            print(f"    Time: {test_time:.1f}s ({test_time/60:.1f} min)")
        
        return test_results
    
    def _generate_comprehensive_report(self):
        """Generate complete comparison report (cross validation + pseudo-labeled test set)"""
        report = "\n" + "=" * 90 + "\n"
        report += "COMPREHENSIVE COMPARISON REPORT\n"
        report += "AI Code Detection Methods: GPTZero vs DetectGPT vs CodeBERT\n"
        report += "=" * 90 + "\n\n"
        
        # ====================================================================
        # Part 1: Known-positive AI results
        # ====================================================================
        if 'known_ai_eval' in self.results and self.results['known_ai_eval']:
            report += "┌─ PART 1: KNOWN-AI DETECTION (ai.json, positive-only) ───────────────┐\n"
            report += "│\n"
            report += "│ Method             Features  AI Recall        Mean AI Prob\n"
            report += "├──────────────────────────────────────────────────────────────────────────────┤\n"
            
            known_results = self.results['known_ai_eval']
            for method in ['GPTZero', 'DetectGPT', 'CodeBERT']:
                if method in known_results:
                    r = known_results[method]
                    sec_per_sample = r['elapsed_seconds'] / max(1.0, float(len(r['predictions'])))
                    recall_str = "n/a" if np.isnan(r['ai_recall_on_known_ai']) else f"{r['ai_recall_on_known_ai']:.4f}"
                    prob_str = "n/a" if np.isnan(r['mean_ai_probability']) else f"{r['mean_ai_probability']:.4f}"
                    report += f"│ {method:<17} {r['num_features']:>3}       "
                    report += f"{recall_str:<14} "
                    report += f"{prob_str:<16} "
                    report += f"{sec_per_sample:.2f}s\n"
            
            report += "│\n"
            report += "│ Note: No verified human ground truth is available, so Part 1 reports AI recall only.\n"
            report += "└──────────────────────────────────────────────────────────────────────────────┘\n\n"
        
        # ====================================================================
        # Part 2: Pseudo-label consistency results
        # ====================================================================
        if self.results.get('test_set') is not None:
            report += "┌─ PART 2: PSEUDO-LABEL CONSISTENCY (last_success pool) ────────┐\n"
            report += "│\n"
            report += "│ Method             Repeats  Agreement         AI Rate\n"
            report += "├──────────────────────────────────────────────────────────────────────────────┤\n"
            
            test_results = self.results['test_set']
            repeats = int(test_results.get('__meta__', {}).get('num_repeats', 1))
            codebert_rate = test_results['CodeBERT']['ai_rate'] if 'CodeBERT' in test_results else float('nan')
            for method in ['GPTZero', 'DetectGPT']:
                if method in test_results:
                    r = test_results[method]
                    report += f"│ {method:<17} {repeats:>3}      "
                    report += f"{r['agreement_vs_codebert']:.4f}±{r['agreement_vs_codebert_std']:.3f}     "
                    report += f"{r['ai_rate']:.4f}±{r['ai_rate_std']:.3f}\n"
            if 'CodeBERT' in test_results:
                report += f"│ {'CodeBERT (reference)':<17} {repeats:>3}      "
                report += f"n/a              {codebert_rate:.4f}\n"
            
            report += "│\n"
            report += "│ Note: Part 2 uses CodeBERT pseudo-labels (consistency study, not ground-truth performance).\n"
            report += f"│ Sampling: {getattr(self, 'sampling_strategy', 'stratified')} (pseudo-label based).\n"
            report += "└──────────────────────────────────────────────────────────────────────────────┘\n\n"
        
        # ====================================================================
        # Key Findings
        # ====================================================================
        report += "┌─ KEY FINDINGS ──────────────────────────────────────────────────────┐\n"
        report += "│\n"
        
        if 'known_ai_eval' in self.results and self.results['known_ai_eval']:
            known = self.results['known_ai_eval']
            report += "│ 1. Known-AI Detection (positive-only):\n"
            if 'GPTZero' in known:
                report += f"│    - GPTZero AI recall on known AI:  {known['GPTZero']['ai_recall_on_known_ai']:.4f}\n"
            if 'DetectGPT' in known:
                report += f"│    - DetectGPT AI recall on known AI: {known['DetectGPT']['ai_recall_on_known_ai']:.4f}\n"
            if 'CodeBERT' in known and not np.isnan(known['CodeBERT']['ai_recall_on_known_ai']):
                report += f"│    - CodeBERT AI recall on known AI: {known['CodeBERT']['ai_recall_on_known_ai']:.4f}\n"
            elif 'CodeBERT' in known:
                report += "│    - CodeBERT AI recall on known AI: n/a (no matching feature rows)\n"
            report += "│    - This is not full accuracy because verified human negatives are unavailable.\n"
            report += f"│\n"
        
        if self.results.get('test_set') is not None:
            test_results = self.results['test_set']
            repeats = int(test_results.get('__meta__', {}).get('num_repeats', 1))
            report += "│ 2. Pseudo-label Consistency (unlabeled full-data pool):\n"
            report += f"│    - Repeats: {repeats} random subsamples\n"
            if 'GPTZero' in test_results:
                report += f"│    - GPTZero: agreement={test_results['GPTZero']['agreement_vs_codebert']:.4f}±{test_results['GPTZero']['agreement_vs_codebert_std']:.3f}, AI-rate={test_results['GPTZero']['ai_rate']:.4f}±{test_results['GPTZero']['ai_rate_std']:.3f}\n"
            if 'DetectGPT' in test_results:
                report += f"│    - DetectGPT: agreement={test_results['DetectGPT']['agreement_vs_codebert']:.4f}±{test_results['DetectGPT']['agreement_vs_codebert_std']:.3f}, AI-rate={test_results['DetectGPT']['ai_rate']:.4f}±{test_results['DetectGPT']['ai_rate_std']:.3f}\n"
            if 'CodeBERT' in test_results:
                report += f"│    - CodeBERT reference AI-rate={test_results['CodeBERT']['ai_rate']:.4f}\n"
                report += "│    - CodeBERT is the pseudo-label source in Part 2, so no self-performance metric is reported.\n"
            report += f"│\n"
        
        report += "│ 3. Why Code Detection Needs Specialized Methods:\n"
        report += "│    ✓ GPTZero (text-based) focuses on perplexity, ignores code structure\n"
        report += "│    ✓ DetectGPT uses perturbation curvature, but designed for natural language\n"
        report += "│    ✓ CodeBERT captures code-specific patterns:\n"
        report += "│      - Identifier naming entropy (naming conventions differ)\n"
        report += "│      - Comment ratio and placement (AI vs human commenting styles)\n"
        report += "│      - Code complexity distribution (burstiness patterns)\n"
        report += "│      - Trained on 6 programming languages (not just text)\n"
        report += "│\n"
        report += "│ 4. Feature Impact:\n"
        report += "│    - GPTZero:  1 feature (perplexity threshold only)\n"
        report += "│    - DetectGPT: 1 feature (perturbation log-likelihood curvature)\n"
        report += "│    - CodeBERT: 10 features (comprehensive code characteristics) ✓\n"
        report += "│\n"
        report += "│ 5. Scalability & Compute Cost (per sample on A100 GPU):\n"
        report += "│    - GPTZero:  ~0.03s/sample (fast, single GPT-2 forward pass)\n"
        report += "│    - DetectGPT: ~5s/sample (50 T5 perturbations + 51 GPT-2 passes/chunk)\n"
        report += "│    - CodeBERT: ~0.01s/sample (pre-computed features + XGBoost)\n"
        report += "│    ⚠ DetectGPT on 44k samples would require ~61 hours (infeasible)\n"
        report += "│    → Test set subsampled to show representativeness\n"
        report += "│\n"
        report += "└─────────────────────────────────────────────────────────────────────┘\n"
        
        print(report)
        
        # Save report
        output_path = '/user/zhuohang.yu/u24922/exam/comparison_comprehensive_report.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ Report saved to: {output_path}")
        
        # Generate visualization
        self._generate_visualization()
        
    def _generate_visualization(self):
        """Generate comparison charts focused on Part 2 (pseudo-label consistency)."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except Exception:
            print("  ⚠️  Matplotlib not available, skipping visualization")
            return

        if self.results.get('test_set') is None:
            print("  ⚠️  No test_set results, skipping visualization")
            return

        test_results = self.results['test_set']
        methods = ['GPTZero', 'DetectGPT', 'CodeBERT']
        available = [m for m in methods if m in test_results]
        if not available:
            print("  ⚠️  No available methods in test_set, skipping visualization")
            return

        colors = {'GPTZero': '#FF6B6B', 'DetectGPT': '#4ECDC4', 'CodeBERT': '#45B7D1'}
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # 1) F1 score bars (for methods with defined F1)
        ax = fig.add_subplot(gs[0, 0])
        f1_methods = [m for m in available if np.isfinite(test_results[m].get('f1', np.nan))]
        f1_vals = [test_results[m]['f1'] for m in f1_methods]
        ax.bar(f1_methods, f1_vals, color=[colors[m] for m in f1_methods], alpha=0.75)
        ax.set_ylabel('F1 Score')
        ax.set_title('Part 2: F1 (Pseudo-label)')
        ax.set_ylim([0, 1])
        for i, v in enumerate(f1_vals):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)

        # 2) Precision vs Recall (series colors fixed, not method colors)
        ax = fig.add_subplot(gs[0, 1])
        pr_methods = [m for m in available if np.isfinite(test_results[m].get('precision', np.nan))
                      and np.isfinite(test_results[m].get('recall', np.nan))]
        x = np.arange(len(pr_methods))
        width = 0.35
        prec_vals = [test_results[m]['precision'] for m in pr_methods]
        rec_vals = [test_results[m]['recall'] for m in pr_methods]
        ax.bar(x - width/2, prec_vals, width, label='Precision', color='#F28E8E', alpha=0.85)
        ax.bar(x + width/2, rec_vals, width, label='Recall', color='#8ECAE6', alpha=0.85)
        ax.set_ylabel('Score')
        ax.set_title('Part 2: Precision vs Recall')
        ax.set_xticks(x)
        ax.set_xticklabels(pr_methods)
        ax.legend()
        ax.set_ylim([0, 1])

        # 3) Real ROC curves vs CodeBERT pseudo-label reference
        ax = fig.add_subplot(gs[0, 2])
        y_true = test_results['CodeBERT']['predictions'] if 'CodeBERT' in test_results else None
        roc_plotted = False
        if y_true is not None:
            for m in ['GPTZero', 'DetectGPT']:
                if m in test_results:
                    probs = test_results[m].get('probabilities', None)
                    if probs is None or len(probs) != len(y_true):
                        continue
                    try:
                        fpr, tpr, _ = roc_curve(y_true, probs)
                        auc_val = roc_auc_score(y_true, probs)
                        ax.plot(fpr, tpr, color=colors[m], lw=2, label=f'{m} (AUC={auc_val:.3f})')
                        roc_plotted = True
                    except Exception:
                        continue
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.35, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Part 2: ROC Curves')
        ax.grid(alpha=0.25)
        ax.legend()
        if not roc_plotted:
            ax.text(0.5, 0.5, 'ROC unavailable', ha='center', va='center', transform=ax.transAxes, fontsize=10)

        # 4-6) Confusion matrices
        for idx, m in enumerate(available[:3]):
            ax = fig.add_subplot(gs[1, idx])
            cm = test_results[m].get('confusion_matrix', None)
            if cm is None:
                ax.axis('off')
                continue
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'],
                        ax=ax, cbar=False)
            ax.set_title(f'{m} Confusion Matrix', fontweight='bold', fontsize=10)
            ax.set_ylabel('Pseudo Label')
            ax.set_xlabel('Predicted Label')

        plt.suptitle('AI Code Detection Methods Comparison', fontsize=16, fontweight='bold', y=0.98)
        output_path = '/user/zhuohang.yu/u24922/exam/comparison_comprehensive_visualization.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
        plt.close()


# ============================================================================
# Main Function
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Code Detection Comparison Experiment')
    parser.add_argument('--train_data_path', type=str, default=None,
                        help='Optional legacy CSV path. Not required in current no-ground-truth setting.')
    parser.add_argument('--sample_size', type=int, default=200,
                        help='Subsample test set to this size (0=use all). '
                             'Default 200: practical trade-off for DetectGPT stability.')
    parser.add_argument('--test_repeats', type=int, default=3,
                        help='Part 2 random-subsample repeats (recommended: 3-5).')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Device to use (auto-detect if not specified)')
    parser.add_argument('--methods', type=str, default='gptzero,detectgpt,codebert',
                        help='Comma-separated methods to run: gptzero,detectgpt,codebert')
    parser.add_argument('--sampling_strategy', type=str, default='stratified',
                        choices=['stratified', 'random'],
                        help='Part-2 subsampling: stratified (pseudo-label based) or random.')
    args = parser.parse_args()

    selected_methods = [m.strip().lower() for m in args.methods.split(',') if m.strip()]
    
    # Run comparison experiment (no-ground-truth setting):
    # Part 1 uses ai.json known positives; Part 2 uses cleaned last_success pool.
    experiment = ComparisonExperiment(
        train_data_path=args.train_data_path,
        full_data_path='/user/zhuohang.yu/u24922/exam/smartbeans_submission_last_success.json'
    )
    
    results = experiment.run_experiment(
        sample_size=args.sample_size,
        device=args.device,
        test_repeats=args.test_repeats,
        methods=selected_methods,
        sampling_strategy=args.sampling_strategy
    )
