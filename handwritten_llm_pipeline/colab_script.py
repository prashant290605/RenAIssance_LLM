import os

# --- CELL 1: SETUP ---
def setup_colab():
    print("Setting up Colab environment...")
    # Clone repo - note: local environment is already in the repo, but for colab:
    # !git clone https://github.com/prashant290605/RenAIssance_LLM
    # %cd RenAIssance_LLM
    # !git checkout prashant_handwritten_llm_pipeline
    
    # In Colab, we would run:
    # !pip install -r requirements.txt
    print("Dependencies would be installed here.")

# --- CELL 2: TRAINING ---
def run_training(train_dir, val_dir):
    from handwritten_llm_pipeline.train import train
    train(train_dir, val_dir, epochs=5)

# --- CELL 3: INFERENCE & LLM PIPELINE ---
def run_pipeline(model_path, image_path):
    from handwritten_llm_pipeline.inference import TrOCRInference
    from handwritten_llm_pipeline.llm_rerank import SpanishLLMRefiner
    from handwritten_llm_pipeline.evaluation import SpanishHTREvaluator
    
    # 1. Baseline
    inf = TrOCRInference(model_path)
    baseline, beams = inf.predict(image_path)
    print(f"Baseline: {baseline}")
    
    # 2. LLM Rerank
    refiner = SpanishLLMRefiner()
    reranked = refiner.rerank_beams(beams)
    print(f"Reranked: {reranked}")
    
    # 3. Paragraph Correction
    final = refiner.correct_paragraph(reranked)
    print(f"Final: {final}")
    
    return baseline, reranked, final

# --- CELL 4: EVALUATION ---
def run_evaluation(baselines, reranks, finals, references):
    from handwritten_llm_pipeline.evaluation import SpanishHTREvaluator
    evaluator = SpanishHTREvaluator()
    results = evaluator.run_comparison(baselines, reranks, finals, references)
    print(results)
    
    analysis = evaluator.perform_error_analysis(baselines, references)
    print("Error Analysis (Baseline):", analysis)

if __name__ == "__main__":
    print("Colab Pipeline Script Ready.")
