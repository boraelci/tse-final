import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PLBartTokenizer, PLBartForConditionalGeneration
from tqdm import tqdm
import math
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from training.utg_dataset import UtgDataset

class UtgTrainer:
    def __init__(self, args):
        self.output_dir = args.output_dir
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.offset = args.offset
        self.limit = args.limit
        self.learning_rate = args.learning_rate
        self.checkpoint_dir = args.checkpoint_dir
        self.model_path = args.model_path

    def run(
        self, train_input_path, train_target_path, eval_input_path, eval_target_path
    ):
        # Set parameters
        model_path = self.model_path

        epochs = self.epochs
        batch_size = self.batch_size
        offset = self.offset
        limit = self.limit
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the tokenizer and model
        tokenizer = PLBartTokenizer.from_pretrained(model_path)
        model = PLBartForConditionalGeneration.from_pretrained(model_path).to(device)

        # Create the dataset and data loaders
        train_dataset = UtgDataset(
            tokenizer,
            train_input_path,
            train_target_path,
            max_length=1024,
            offset=offset,
            limit=limit,
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        eval_dataset = UtgDataset(
            tokenizer,
            eval_input_path,
            eval_target_path,
            max_length=1024,
            offset=offset,
            limit=limit,
        )
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        # Set up the optimizer and learning rate scheduler
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=math.ceil(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        # Use mixed-precision training if available
        scaler = GradScaler(enabled=torch.cuda.is_available())

        # Initialize variables for checkpointing
        best_eval_loss = float("inf")

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            model.train()
            train_loss = 0

            for inputs, targets in tqdm(train_dataloader, desc="Training"):
                optimizer.zero_grad()

                inputs = {
                    key: val.reshape(val.shape[0], -1).to(device)
                    for key, val in inputs.items()
                }
                targets = (
                    targets["input_ids"]
                    .reshape(-1, targets["input_ids"].shape[-1])
                    .to(device)
                )

                with autocast(enabled=torch.cuda.is_available()):
                    outputs = model(**inputs, labels=targets)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            scheduler.step()
            print(f"Train loss: {train_loss / len(train_dataloader)}")

            # Evaluation loop
            model.eval()
            eval_loss = 0
            bleu_score = 0
            smoothing = SmoothingFunction()

            with torch.no_grad():
                for inputs, targets in tqdm(eval_dataloader, desc="Evaluating"):
                    inputs = {
                        key: val.reshape(val.shape[0], -1).to(device)
                        for key, val in inputs.items()
                    }
                    targets = (
                        targets["input_ids"]
                        .reshape(-1, targets["input_ids"].shape[-1])
                        .to(device)
                    )

                    with autocast(enabled=torch.cuda.is_available()):
                        outputs = model(**inputs, labels=targets)
                        loss = outputs.loss

                    eval_loss += loss.item()

                    # Generate predictions
                    predictions = model.generate(inputs["input_ids"])
                    predictions = predictions.cpu().numpy().tolist()
                    targets = targets.cpu().numpy().tolist()

                    # Calculate BLEU score for each pair of prediction and target
                    for pred, tgt in zip(predictions, targets):
                        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
                        tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
                        bleu_score += sentence_bleu(
                            [tgt_text.split()],
                            pred_text.split(),
                            smoothing_function=smoothing.method1,
                        )

            print(f"Eval loss: {eval_loss / len(eval_dataloader)}")
            bleu_score /= len(eval_dataloader.dataset)
            print(f"BLEU score: {bleu_score}")

            best_eval_loss = self.checkpoint(
                model, tokenizer, eval_loss, best_eval_loss
            )
        self.save(model, tokenizer)

    def checkpoint(self, model, tokenizer, eval_loss, best_eval_loss):
        checkpoint_dir = self.checkpoint_dir
        model.save_pretrained(f"{checkpoint_dir}")
        tokenizer.save_pretrained(f"{checkpoint_dir}")

        # Checkpointing
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            model.save_pretrained(f"{self.output_dir}/best_model")
            tokenizer.save_pretrained(f"{self.output_dir}/best_model")

        return best_eval_loss

    def save(self, model, tokenizer):
        # Save the fine-tuned model
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
