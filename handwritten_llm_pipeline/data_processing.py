import os
import re
import pandas as pd
import unicodedata
import html
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor

# Handle high-resolution historical scans
Image.MAX_IMAGE_PIXELS = None

def robust_normalize(text):
    """Deep normalization: handles HTML entities, lowercase, no accents, no punctuation, no spaces."""
    if not text: return ""
    # Unescape HTML entities (e.g., &#x3a; -> :)
    text = html.unescape(text)
    # Normalize unicode (NFD) and filter out non-spacing marks (accents)
    text = unicodedata.normalize('NFD', text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    # Lowercase and remove all non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9]', '', text).lower()
    return text

def convert_docx_to_txt(docx_path):
    """Converts DOCX to TXT using python-docx."""
    from docx import Document
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def align_transcriptions(txt_content, pdf_base_name):
    """
    Splits text by PDF p\d+ markers and maps to expected image names.
    Captures page numbers using regex to be more robust.
    """
    # Split content by "PDF p1", "PDF p 1", "PDF P1", etc.
    chunks = re.split(r'PDF\s+[pP]\s*(\d+)', txt_content)
    map_list = []
    
    # re.split with capture group returns [preamble, p1_num, p1_text, p2_num, p2_text, ...]
    for i in range(1, len(chunks), 2):
        page_num = chunks[i]
        text = chunks[i+1].strip()
        if text:
            # We try standard underscore naming
            img_name = f"{pdf_base_name}_page{page_num}.jpg"
            map_list.append({"file_name": img_name, "text": text})
            
    return map_list

class SpanishHTRDataset(Dataset):
    """
    Dataset for Spanish Handwritten Text Recognition.
    """
    def __init__(self, root_dir, processor, max_target_length=128):
        self.root_dir = root_dir
        self.processor = processor
        self.max_target_length = max_target_length
        self.df = pd.read_csv(os.path.join(root_dir, "labels.csv"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        img_path = os.path.join(self.root_dir, "images", file_name)
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length,
            truncation=True
        ).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
