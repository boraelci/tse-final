import argparse
from training.utg_trainer import UtgTrainer

class LocalRunner:
    def __init__(self):
        pass

    def put_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--train_input_path", type=str, required=True)
            parser.add_argument("--train_target_path", type=str, required=True)
            parser.add_argument("--eval_input_path", type=str, required=True)
            parser.add_argument("--eval_target_path", type=str, required=True)

        parser.add_argument("--epochs", type=int, required=True)  # 3, 5
        parser.add_argument("--batch_size", type=int, required=True)  # 2
        parser.add_argument(
            "--output_dir", type=str, required=True
        )  # output/model for local or opt/ml/model for aws
        parser.add_argument(
            "--learning_rate", type=float, required=True
        )  # 1e-3 (for scheduler)
        parser.add_argument(
            "--offset", type=int, required=True
        )  # Only specify for train, eval scales to 1/8th automatically
        parser.add_argument(
            "--limit", type=int, required=True
        )  # Only specify for train, eval scales to 1/8th automatically
        parser.add_argument(
            "--checkpoint_dir", type=str
        )  # output/checkpoints for local or /opt/ml/checkpoints for aws

        args, _ = parser.parse_known_args()
        return args

    def run(self, args):
        utg_trainer = UtgTrainer(args)
        utg_trainer.run(
            args.train_input_path, args.train_target_path, args.eval_input_path, args.eval_target_path
        )
if __name__ == "__main__":
    local_runner = LocalRunner()
    args = local_runner.put_args()
    local_runner.run(args)