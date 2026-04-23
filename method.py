import torch
import numpy as np
import json
import os
import re
from collections import Counter
from transformers import AutoModelForMaskedLM, AutoTokenizer


os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['SAFETENSORS_FORCE_CONVERT'] = '0'

class AICodeAnalyzer:
    """
    AICodeAnalyzer Used to estimate the perplexity of code under a pre-trained code language model, 
    thereby measuring its similarity to AI-generated code.
    """
    def __init__(self, model_name="microsoft/codebert-base-mlm"):
        print(f"Loading model: {model_name} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
       
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
            trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=False,
            local_files_only=True
        ).to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")

    def compute_perplexity(self, code_snippet):
        """
        Calculate the perplexity of the entire code segment.

        A lower perplexity indicates that the code is statistically more predictable by the model.
      
        """
        # Tokenize with truncation to handle long sequences
        encodings = self.tokenizer(
            code_snippet, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        max_length = 512
        stride = 256 
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # The actual length of the calculated loss
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            
            # To avoid redundant calculations, the loss for the overlapping portion is not calculated.
            target_ids[:, :-trg_len] = -100 

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                # outputs.loss is the average loss, which needs to be converted to the total loss.
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        total_nll = torch.stack(nlls).sum()
        # Total loss divided by total number of tokens
        avg_loss = total_nll / seq_len
        perplexity = torch.exp(avg_loss)

        return perplexity.item(), avg_loss.item()

    def compute_average_token_probability(self, code_snippet):
        """
        Calculate the average probability of the true token under the model's predicted distribution.

        This metric is used to supplement the explanation of whether the code consists of high-frequency, 
        templated tokens. By calculating the model's prediction confidence for each word,

        it captures the characteristics of AI code being 'highly templated' and 'lacking in unexpectedness'. Typically, this metric for AI code is significantly higher than that for student-written code.

        Refer to the feature analysis ideas of EX-CODE.
        Bulla （2024) "EX-CODE: A Robust and Explainable Model to Detect AI-Generated Code"
        """
        inputs = self.tokenizer(
            code_snippet,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)

        token_probs = []
        for i, token_id in enumerate(input_ids[0]):
            if token_id == self.tokenizer.pad_token_id:
                continue
            token_probs.append(probs[0, i, token_id].item())

        return float(np.mean(token_probs))

    def compute_entropy(self, code_snippet):
        """
        Calculate the average entropy of the model's token distribution.

        Lower entropy indicates more confident and templated predictions,
        which is often observed in AI-generated code.
        """
        inputs = self.tokenizer(
            code_snippet,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-12)

        entropies = []
        for i, token_id in enumerate(input_ids[0]):
            if token_id == self.tokenizer.pad_token_id:
                continue
            entropy = -torch.sum(probs[0, i] * log_probs[0, i]).item()
            entropies.append(entropy)

        if not entropies:
            return 0.0
        return float(np.mean(entropies))

    def compute_code_length(self, code_snippet):
        """Calculate code length (number of characters)."""
        return float(len(code_snippet.strip()))

    def compute_line_length_stats(self, code_snippet):
        """Calculate average and standard deviation of line lengths."""
        lines = [line for line in code_snippet.split('\n') if line.strip()]
        if not lines:
            return 0.0, 0.0
        
        line_lengths = [len(line.strip()) for line in lines]
        avg_length = float(np.mean(line_lengths))
        std_length = float(np.std(line_lengths)) if len(line_lengths) > 1 else 0.0
        return avg_length, std_length

    def compute_comment_ratio(self, code_snippet):
        """Calculate the proportion of comments in the code."""
        lines = code_snippet.split('\n')
        comment_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('#') or stripped.startswith('*'):
                comment_lines += 1
        
        total_lines = len([l for l in lines if l.strip()])
        if total_lines == 0:
            return 0.0
        return float(comment_lines / total_lines)

    def compute_identifier_entropy(self, code_snippet):
        """
        Calculate the entropy of identifier (variable/function) names.
        AI-generated code tends to have more descriptive, less diverse names.
        Human code tends to have more diverse and varied naming patterns.
        """
        # Extract identifiers (variable/function names)
        # Match: letters, digits, underscore combinations
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code_snippet)
        
        if not identifiers:
            return 0.0
        
        # Count frequency of each identifier
        identifier_counts = Counter(identifiers)
        
        # Calculate entropy
        total = len(identifiers)
        entropy = 0.0
        for count in identifier_counts.values():
            prob = count / total
            entropy -= prob * np.log2(prob)
        
        return float(entropy)

    def compute_ngram_repetition(self, code_snippet, n=3):
        """
        Calculate the repetition rate of n-grams (e.g., 3-grams).
        AI code tends to have higher repetition (more templated patterns).
        """
        # Tokenize into words
        tokens = re.findall(r'\w+|[(){};,=\[\]]', code_snippet)
        
        if len(tokens) < n:
            return 0.0
        
        # Extract n-grams
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        
        if not ngrams:
            return 0.0
        
        # Count n-gram frequencies
        ngram_counts = Counter(ngrams)
        
        # Calculate repetition rate (how many n-grams appear more than once)
        repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
        repetition_rate = float(repeated_ngrams / len(set(ngrams))) if len(set(ngrams)) > 0 else 0.0
        
        return repetition_rate

    def analyze_code(self, code_snippet):
        """
        It analyzes a single code segment and returns multiple AI similarity metrics.
        Includes both language model features and code structural features.
        """
        # Language model features
        perplexity, loss = self.compute_perplexity(code_snippet)
        avg_token_prob = self.compute_average_token_probability(code_snippet)
        entropy = self.compute_entropy(code_snippet)
        burstiness = self.compute_burstiness(code_snippet)
        
        # Code structural features
        code_length = self.compute_code_length(code_snippet)
        avg_line_length, std_line_length = self.compute_line_length_stats(code_snippet)
        comment_ratio = self.compute_comment_ratio(code_snippet)
        identifier_entropy = self.compute_identifier_entropy(code_snippet)
        ngram_repetition = self.compute_ngram_repetition(code_snippet)

        analysis_result = {
            # Language model features
            "perplexity": round(perplexity, 4),
            "cross_entropy_loss": round(loss, 4),
            "avg_token_probability": round(avg_token_prob, 6),
            "avg_entropy": round(entropy, 6),
            "burstiness": round(burstiness, 4),
            
            # Code structural features
            "code_length": round(code_length, 1),
            "avg_line_length": round(avg_line_length, 2),
            "std_line_length": round(std_line_length, 2),
            "comment_ratio": round(comment_ratio, 4),
            "identifier_entropy": round(identifier_entropy, 4),
            "ngram_repetition": round(ngram_repetition, 4),
            
            "interpretation": (
                "Lower perplexity, higher token probability, lower entropy, lower burstiness, "
                "higher comment ratio, lower identifier entropy, and higher n-gram repetition "
                "indicate higher similarity to typical AI-generated code patterns."
            )
        }

        return analysis_result

    def extract_all_features(self, code_snippet):
        """
        Alias for analyze_code - extract all features from a code snippet.
        """
        try:
            return self.analyze_code(code_snippet)
        except Exception as e:
            return None

    def compute_burstiness(self, code_snippet):
        """
        Calculate the perplexity fluctuations between lines of code. (Burstiness)。
        
        Principle:
        AI-generated code typically maintains low and stable perplexity (small variance) on each line;
        Human code exhibits greater fluctuations in complexity (large variance).
        
        Reference: 
        - "variation of code line perplexity" (Xu & Sheng, 2024)
        - "standard deviation of perplexity across individual lines" 
        """
        # Filter out excessively short lines of code (such as those containing only parentheses '}' or simple 'else:').
        lines = [line.strip() for line in code_snippet.split('\n') if len(line.strip()) > 10]
        
        # If the number of valid rows is too small to calculate the variance, return 0.0 or the default value.
        if len(lines) < 2:
            return 0.0
            
        line_losses = []
        for line in lines:
            # Calculate the loss for each row separately.
            inputs = self.tokenizer(
                line, 
                return_tensors="pt", 
                truncation=True, 
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                line_losses.append(outputs.loss.item())
        
        # Calculate the standard deviation of the loss as the Burstiness.
        # AI code typically has a smaller standard deviation (due to consistent style), while human code has a larger standard deviation.
        return np.std(line_losses)


if __name__ == "__main__":
    import json
    import os
    
    json_path = 'parsed.json'
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found in current directory")
        exit(1)
    
    print(f"Loading dataset from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Dataset loaded. Total records: {len(data) - 1}")

    print("\nInitializing AICodeAnalyzer...")
    analyzer = AICodeAnalyzer()

    headers = data[0]
    try:
        content_idx = headers.index('content')
        id_idx = headers.index('id')
        user_idx = headers.index('user')
    except ValueError as e:
        print(f"Error: Required column not found - {e}")
        exit(1)
    
    results = []
    errors = []
    
    print("\nProcessing codes...\n")
    for i, row in enumerate(data[1:], 1):
        try:
            code_content = row[content_idx]
            record_id = row[id_idx]
            user_id = row[user_idx]
            
            if not code_content or not code_content.strip():
                continue
            
            analysis = analyzer.analyze_code(code_content)
            analysis['record_id'] = record_id
            analysis['user_id'] = user_id
            results.append(analysis)
            
            if i % 10 == 0:
                print(f"Processed {i} records...")
        except Exception as e:
            error_msg = f"Error processing row {i}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
    
    output_path = 'analysis_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Analysis complete!")
    print(f"Successfully processed: {len(results)} records")
    print(f"Errors encountered: {len(errors)} records")
    print(f"Results saved to: {output_path}")
    print(f"{'='*50}")