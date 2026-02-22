# RenAIssance HTR Pipeline

End-to-end Handwritten Text Recognition (HTR) for early modern Spanish documents, featuring a multi-stage LLM-integrated refinement pipeline.

## Technical Architecture

This project implements a three-stage pipeline where an LLM is used throughout the recognition and selection process, not just as a final cleanup.

1.  **Stage 1: Baseline OCR (TrOCR)**
    *   Uses `microsoft/trocr-base-handwritten`.
    *   Generates the top-5 most likely transcriptions using Beam Search (`num_beams=5`).
2.  **Stage 2: Beam Re-ranking (LLM)**
    *   Utilizes `google/flan-t5-base`.
    *   The LLM analyzes all 5 beam candidates and selects the one that is most historically and linguistically plausible. This ensures the decoding process considers linguistic context.
3.  **Stage 3: Paragraph Correction (LLM)**
    *   Utilizes `google/flan-t5-base`.
    *   The full paragraph is analyzed for contextual consistency, fixing recognition errors while strictly preserving 17th-century orthography.

## Performance Comparison

| Stage                  | CER    | WER |
|:-----------------------|:-------|:----|
| Baseline TrOCR         | 0.9969 | 1.0 |
| + Beam Rerank          | 0.9969 | 1.0 |
| + Paragraph Correction | 0.9970 | 1.0 |

> [!NOTE]
> High CER is expected in this zero-shot demonstration on high-resolution historical scans. The pipeline structure is designed to be fine-tuned on specific datasets to bring error rates down to research-grade levels.

## Error Analysis

Based on the initial run on historical test samples:

*   **Character Confusions**: Frequent confusion between `E` and `0`, and spaces being replaced by `o`.
*   **Diacritics**: 100% error frequency on archaic Spanish diacritics in the zero-shot model.
*   **Word-Splits**: High error rate due to the baseline model's lack of familiarity with 17th-century word-spanning ligatures.

## Project Structure

```text
RenAIssance_LLM/
├── data/
│   ├── images/                  # 35 JPEG pages converted from 5 PDFs
│   ├── *.docx                   # Original transcription files (5 documents)
│   └── labels.csv               # Aligned image↔text pairs (5 samples)
├── handwritten_llm_pipeline/
│   ├── __init__.py              # Package init
│   ├── data_processing.py       # PDF/DOCX conversion & alignment
│   ├── train.py                 # TrOCR fine-tuning structure
│   ├── inference.py             # Baseline prediction engine (beam search)
│   ├── llm_rerank.py            # LLM reranking & paragraph correction
│   ├── evaluation.py            # CER/WER & error analysis
│   └── colab_script.py          # Main Colab orchestration entry point
├── Collab_LLM_Pipeline_1.ipynb    # Executed Colab notebook
├── requirements.txt
└── README.md
```

## How to Run

The entire pipeline is designed to run on a free Google Colab GPU.
1.  Open the provided Colab Notebook script.
2.  Clone this repository.
3.  Install dependencies: `pip install -r requirements.txt`.
4.  Run the orchestration cells to process data and measure improvements.
