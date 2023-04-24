import argparse
from runners.local_runner import LocalRunner

def get_file_paths(input_dir, corpus_type):
    if corpus_type == "raw":
        train_input_path = f"{input_dir}/{corpus_type}/train/input.methods.txt"
        train_target_path = f"{input_dir}/{corpus_type}/train/output.tests.txt"
        eval_input_path = f"{input_dir}/{corpus_type}/eval/input.methods.txt"
        eval_target_path = f"{input_dir}/{corpus_type}/eval/output.tests.txt"
    elif corpus_type == "tokenized":
        train_input_path = f"{input_dir}/{corpus_type}/train/train.input.methods"
        train_target_path = f"{input_dir}/{corpus_type}/train/train.output.tests"
        eval_input_path = f"{input_dir}/{corpus_type}/eval/eval.input.methods"
        eval_target_path = f"{input_dir}/{corpus_type}/eval/eval.output.tests"
    else:
        raise ValueError(f"Invalid corpus type: {corpus_type}")

    return train_input_path, train_target_path, eval_input_path, eval_target_path

def main(args):

    platform = args.platform
    input_dir = args.input_dir
    corpus_type = args.corpus_type
    train_input_path, train_target_path, eval_input_path, eval_target_path = get_file_paths(input_dir, corpus_type)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input_path", type=str, default=train_input_path)
    parser.add_argument("--train_target_path", type=str, default=train_target_path)
    parser.add_argument("--eval_input_path", type=str, default=eval_input_path)
    parser.add_argument("--eval_target_path", type=str, default=eval_target_path)
    local_runner = LocalRunner()
    args = local_runner.put_args(parser)
    if platform == "local":
        local_runner.run(args)
    elif platform == "aws":
        from runners.sagemaker_runner import SagemakerRunner
        s3_bucket_name = "sagemaker-bora-training"
        sagemaker_runner = SagemakerRunner(s3_bucket_name, args)
        sagemaker_runner.run(args.train_input_path, args.train_target_path, args.eval_input_path, args.eval_target_path)
    else:
        raise Exception("Platform invalid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", type=str, required=True)  # local, aws, jupyter
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--corpus_type", type=str, required=True)
    args, _ = parser.parse_known_args()
    main(args)
