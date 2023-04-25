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

import argparse

class UtgEvaluator:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.model_path = args.model_path

    def run(self, eval_input_path, eval_target_path):
        # Set parameters
        model_path = self.model_path
        batch_size = self.batch_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the tokenizer and model
        tokenizer = PLBartTokenizer.from_pretrained(model_path)
        model = PLBartForConditionalGeneration.from_pretrained(model_path).to(device)

        # Create the dataset and data loaders
        eval_dataset = UtgDataset(
            tokenizer,
            eval_input_path,
            eval_target_path,
            max_length=1024,
            offset=0,
            limit=1000, # scales to 1/8th
        )
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

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

        print(f"Eval loss: {eval_loss / len(eval_dataloader)}")
        bleu_score /= len(eval_dataloader.dataset)
        print(f"BLEU score: {bleu_score}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()  
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--batch_size", type=int, required=True)
  args, _ = parser.parse_known_args()
  utg_evaluator = UtgEvaluator(args)
  utg_evaluator.run(eval_input_path="../data/raw/eval/input.methods.txt", eval_target_path="../data/raw/eval/output.tests.txt")