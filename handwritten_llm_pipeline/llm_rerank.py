import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

class SpanishLLMRefiner:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def rerank_beams(self, beams):
        """
        Stage 2: Beam Re-ranking.
        Selects the most historically and linguistically plausible Spanish transcription.
        """
        prompt = (
            "Select the most plausible 17th-century Spanish transcription from the following candidates. "
            "Output only the number of the selected transcription.\n\n"
            "Candidates:\n"
        )
        for i, beam in enumerate(beams):
            prompt += f"{i+1}. {beam}\n"
        prompt += "\nSelected number:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=10)
        selection = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Extract number if LLM returns something like "4." or "The answer is 4"
        match = re.search(r'(\d+)', selection)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(beams):
                return beams[idx]
        
        # Fallback to the first beam if parsing fails
        return beams[0]

    def correct_paragraph(self, paragraph_text):
        """
        Stage 3: Paragraph-level contextual correction.
        """
        prompt = (
            "The following is an OCR transcription of 17th-century Spanish handwritten text. "
            "Correct obvious recognition errors while strictly preserving historical spelling and orthography. "
            "Do not modernize the language.\n\n"
            f"Text: {paragraph_text}\n\n"
            "Corrected text:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return corrected

if __name__ == "__main__":
    # Example usage
    refiner = SpanishLLMRefiner()
    test_beams = [
        "En un lugar de la Mancha",
        "En un lugat de la Mancha",
        "En un lugar de la Manha",
        "En un lugat de la Manha",
        "En un lugar de la Man-cha"
    ]
    print("Reranked:", refiner.rerank_beams(test_beams))
    
    test_para = "En un lugat de la Mancha de cujo nombre no quiero acordarme..."
    print("Corrected Paragraph:", refiner.correct_paragraph(test_para))
