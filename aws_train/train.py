import argparse
import torch
import boto3
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PLBartTokenizer, PLBartForConditionalGeneration
from tqdm import tqdm
import math
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup

class Methods2TestDataset(Dataset):
    def __init__(self, tokenizer, input_file, target_file, max_length=1024):
        self.tokenizer = tokenizer
        self.input_file = input_file
        self.target_file = target_file
        self.max_length = max_length

        if input_file.startswith("s3://"):
            self.inputs = self.read_s3_file(input_file)
        else:
            with open(input_file, "r") as f:
                self.inputs = f.readlines()

        if target_file.startswith("s3://"):
            self.targets = self.read_s3_file(target_file)
        else:
            with open(target_file, "r") as f:
                self.targets = f.readlines()

    def read_s3_file(self, s3_path):
        s3 = boto3.client("s3")
        bucket, key = self.parse_s3_path(s3_path)
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read().decode("utf-8")
        return data.splitlines()[:10000]

    def parse_s3_path(self, s3_path):
        s3_path = s3_path.replace("s3://", "")
        bucket, key = s3_path.split("/", 1)
        return bucket, key

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx].strip()
        target = self.targets[idx].strip()

        tokenized_input = self.tokenizer(input, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        tokenized_target = self.tokenizer(target, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")

        return tokenized_input, tokenized_target

def main(args):
    # Set parameters
    model_name = "uclanlp/plbart-base"
    train_input_file = args.train_input_file
    train_target_file = args.train_target_file
    eval_input_file = args.eval_input_file
    eval_target_file = args.eval_target_file
    epochs = args.epochs
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer and model
    tokenizer = PLBartTokenizer.from_pretrained(model_name)
    model = PLBartForConditionalGeneration.from_pretrained(model_name).to(device)

    # Create the dataset and data loaders
    train_dataset = Methods2TestDataset(tokenizer, train_input_file, train_target_file)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    eval_dataset = Methods2TestDataset(tokenizer, eval_input_file, eval_target_file)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Set up the optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=math.ceil(total_steps*0.1), num_training_steps=total_steps)

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

            inputs = {key: val.reshape(val.shape[0], -1).to(device) for key, val in inputs.items()}
            targets = targets["input_ids"].reshape(-1, targets["input_ids"].shape[-1]).to(device)

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(**inputs, labels=targets)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            train_loss += loss.item()

        print(f"Train loss: {train_loss / len(train_dataloader)}")

        # Evaluation loop
        model.eval()
        eval_loss = 0

        with torch.no_grad():
            for inputs, targets in tqdm(eval_dataloader, desc="Evaluating"):
                inputs = {key: val.reshape(val.shape[0], -1).to(device) for key, val in inputs.items()}
                targets = targets["input_ids"].reshape(-1, targets["input_ids"].shape[-1]).to(device)

                with autocast(enabled=torch.cuda.is_available()):
                    outputs = model(**inputs, labels=targets)
                    loss = outputs.loss

                eval_loss += loss.item()

        print(f"Eval loss: {eval_loss / len(eval_dataloader)}")

        # Checkpointing
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            model.save_pretrained(f"{args.output_dir}/best_model")
            tokenizer.save_pretrained(f"{args.output_dir}/best_model")

    # Save the fine-tuned model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_input_file", type=str, required=True)
    parser.add_argument("--train_target_file", type=str, required=True)
    parser.add_argument("--eval_input_file", type=str, required=True)
    parser.add_argument("--eval_target_file", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    args = parser.parse_args()
    main(args)