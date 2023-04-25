import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PLBartTokenizer, PLBartForConditionalGeneration
from tqdm import tqdm
import math
from torch.cuda.amp import GradScaler, autocast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from training.utg_dataset import UtgDataset
from torch.optim.lr_scheduler import LambdaLR
# from datetime import datetime
import time
import json

def inverse_sqrt_schedule(step, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step ** 0.5) * warmup_steps ** -1.5
    else:
        return base_lr * (step ** -0.5)

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
        self.start_epoch = args.start_epoch

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

        # Custom learning rate schedule function
        total_steps = len(train_dataloader) * epochs
        warmup_steps = total_steps * 0.008
        learning_rate = self.learning_rate
        optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6)
        lr_scheduler = LambdaLR(optimizer, lambda step: inverse_sqrt_schedule(step, base_lr=learning_rate, warmup_steps=warmup_steps))

        # Use mixed-precision training if available
        scaler = GradScaler(enabled=torch.cuda.is_available())

        # Initialize variables for checkpointing
        best_eval_loss = float("inf")
        
        checkpoint_dir = self.checkpoint_dir
        start_epoch = self.start_epoch
        if start_epoch > 1:
            state_dics_path = f"{checkpoint_dir}/checkpoint.pt"
            state_dics = torch.load(state_dics_path)
            # model.load_state_dict(checkpoint['model_state_dict']) # no need
            optimizer.load_state_dict(state_dics['optimizer_state_dict'])
            lr_scheduler.load_state_dict(state_dics['lr_scheduler_state_dict'])
            scaler.load_state_dict(state_dics['scaler_state_dict'])
            print("Loaded state_dics")

        # Training loop
        accumulation_steps = 4
        for epoch in range(epochs):
            # print(f"Epoch {epoch + 1}/{epochs}")

            train_start_time = time.time()
            model.train()
            train_loss = 0

            for step, (inputs, targets) in enumerate(tqdm(train_dataloader, desc="Training")):
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

                if (step + 1) % accumulation_steps == 0:
                    # Update model parameters with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # Update learning rate
                    lr_scheduler.step()

                train_loss += loss.item()
            train_end_time = time.time()
            print(f"Train loss: {train_loss / len(train_dataloader)}")
            eval_start_time = time.time()
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
                    predictions = model.generate(inputs["input_ids"], max_length=1024, num_return_sequences=1)
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
            eval_end_time = time.time()
            eval_loss = eval_loss / len(eval_dataloader)
            print(f"Eval loss: {eval_loss}")
            bleu_score /= len(eval_dataloader.dataset)
            print(f"BLEU score: {round(bleu_score*100, 2)}%")

            # Checkpoint
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            torch.save({
                # 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
                }, f"{checkpoint_dir}/checkpoint.pt")

            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            real_epoch = start_epoch + epoch
            with open(f"{checkpoint_dir}/epochs/{real_epoch}.txt", "w") as file:
                file.write(json.dumps({
                'epoch': real_epoch,
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'eval_bleu_score': bleu_score,
                'train_time_taken': train_end_time - train_start_time,
                'eval_time_taken': eval_end_time - eval_start_time,
            }, indent=4))
            
            """
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                model.save_pretrained(f"{self.output_dir}/best_model")
                tokenizer.save_pretrained(f"{self.output_dir}/best_model")
            """

            return best_eval_loss
        # self.save(model, tokenizer)

    def save(self, model, tokenizer):
        # Save the fine-tuned model
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
