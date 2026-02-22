from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Handle high-resolution historical scans
Image.MAX_IMAGE_PIXELS = None

class TrOCRInference:
    def __init__(self, model_path):
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, image_path, num_beams=5):
        """
        Baseline prediction with beam search.
        Returns top-1 prediction and all beam candidates.
        """
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        # Generate with beams
        generated_ids = self.model.generate(
            pixel_values,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            max_length=64,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        # Decode all beams
        candidates = self.processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)
        
        # Primary prediction is the first one
        primary_prediction = candidates[0]
        
        return primary_prediction, candidates

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 2:
        model_path = sys.argv[1]
        img_path = sys.argv[2]
        inf = TrOCRInference(model_path)
        baseline, beams = inf.predict(img_path)
        print(f"Baseline: {baseline}")
        print("Beams:")
        for i, b in enumerate(beams):
            print(f"{i+1}: {b}")
