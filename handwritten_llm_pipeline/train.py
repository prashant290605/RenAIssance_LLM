import os
import torch
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from .data_processing import SpanishHTRDataset
import evaluate

def train(
    train_dir, 
    val_dir, 
    output_dir="./checkpoints", 
    epochs=10, 
    batch_size=4,
    freeze_encoder=True
):
    # 1. Load processor and model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # 2. Freeze encoder if requested
    if freeze_encoder:
        print("Freezing TrOCR Encoder...")
        for param in model.vit.parameters():
            param.requires_grad = False

    # 3. Set model configuration
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # 4. Prepare datasets
    train_dataset = SpanishHTRDataset(train_dir, processor)
    val_dataset = SpanishHTRDataset(val_dir, processor)

    # 5. Define evaluation metric
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    # 6. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True, # GPU support
        output_dir=output_dir,
        logging_steps=10,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="tensorboard"
    )

    # 7. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    # 8. Train
    print("Starting training...")
    trainer.train()
    
    # 9. Save
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    processor.save_pretrained(os.path.join(output_dir, "final_model"))
    print(f"Model saved to {output_dir}/final_model")

if __name__ == "__main__":
    # Placeholder for Colab execution
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train(args.train_dir, args.val_dir, epochs=args.epochs)
