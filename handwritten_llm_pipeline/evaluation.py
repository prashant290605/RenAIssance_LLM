import evaluate
import pandas as pd
from collections import Counter
from tqdm import tqdm

class SpanishHTREvaluator:
    def __init__(self):
        self.cer_metric = evaluate.load("cer")
        self.wer_metric = evaluate.load("wer")

    def compute_metrics(self, predictions, references):
        cer = self.cer_metric.compute(predictions=predictions, references=references)
        wer = self.wer_metric.compute(predictions=predictions, references=references)
        return {"cer": cer, "wer": wer}

    def run_comparison(self, baseline_preds, rerank_preds, final_preds, references):
        """
        Stage-by-stage comparison.
        """
        results = []
        stages = [
            ("Baseline TrOCR", baseline_preds),
            ("+ Beam Rerank", rerank_preds),
            ("+ Paragraph Correction", final_preds)
        ]
        
        for stage_name, preds in stages:
            metrics = self.compute_metrics(preds, references)
            results.append({
                "Stage": stage_name,
                "CER": round(metrics["cer"], 4),
                "WER": round(metrics["wer"], 4)
            })
            
        return pd.DataFrame(results)

    def perform_error_analysis(self, predictions, references):
        """
        Detailed error analysis.
        Identifies character confusions and diacritics issues.
        """
        confusion_counter = Counter()
        diacritic_errors = 0
        word_split_errors = 0
        
        # Simple character-level analysis
        for pred, ref in zip(predictions, references):
            # Diacritics check (e.g., 'á' vs 'a')
            if any(c in "áéíóúñ" for c in ref) and not any(c in "áéíóúñ" for c in pred):
                diacritic_errors += 1
            
            # Word split check (e.g., space count mismatch)
            if pred.count(" ") != ref.count(" "):
                word_split_errors += 1
                
            # Alignment analysis (short strings for example)
            min_len = min(len(pred), len(ref))
            for i in range(min_len):
                if pred[i] != ref[i]:
                    confusion_counter[(ref[i], pred[i])] += 1
                    
        common_confusions = confusion_counter.most_common(10)
        
        return {
            "Common Confusions": common_confusions,
            "Diacritic Error Freq": diacritic_errors / max(1, len(references)),
            "Word-split Error Freq": word_split_errors / max(1, len(references))
        }

if __name__ == "__main__":
    # Example usage
    evaluator = SpanishHTREvaluator()
    refs = ["En un lugar de la Mancha", "cuyo nombre no quiero"]
    p1 = ["En un lugat de la Mancha", "cuyo nombe no quiero"]
    p2 = ["En un lugar de la Mancha", "cuyo nombe no quiero"]
    p3 = ["En un lugar de la Mancha", "cuyo nombre no quiero"]
    
    df = evaluator.run_comparison(p1, p2, p3, refs)
    print(df)
    
    analysis = evaluator.perform_error_analysis(p1, refs)
    print("Error Analysis:", analysis)
