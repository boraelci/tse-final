import sagemaker
from sagemaker.pytorch import PyTorch

# Set your S3 bucket and dataset paths
s3_bucket = "sagemaker-training-bora"
s3_prefix = "input"
s3_output_path = f"s3://{s3_bucket}/output"

# Set the input data locations in S3
use_tokenized = False
if use_tokenized is True:
  s3_train_input = f"s3://{s3_bucket}/{s3_prefix}/tokenized/train/train.input.methods"
  s3_train_target = f"s3://{s3_bucket}/{s3_prefix}/tokenized/train/train.output.tests"
  s3_eval_input = f"s3://{s3_bucket}/{s3_prefix}/tokenized/eval/eval.input.methods"
  s3_eval_target = f"s3://{s3_bucket}/{s3_prefix}/tokenized/eval/eval.output.tests"
else:
  s3_train_input = f"s3://{s3_bucket}/{s3_prefix}/raw/train/input.methods.txt"
  s3_train_target = f"s3://{s3_bucket}/{s3_prefix}/raw/train/output.tests.txt"
  s3_eval_input = f"s3://{s3_bucket}/{s3_prefix}/raw/eval/input.methods.txt"
  s3_eval_target = f"s3://{s3_bucket}/{s3_prefix}/raw/eval/output.tests.txt"

# role = sagemaker.get_execution_role()
role = "arn:aws:iam::977865569421:role/bora-tse-sagemaker"

# Create a PyTorch estimator
estimator = PyTorch(
    entry_point="train.py",
    source_dir=".",
    role=role,
    framework_version="1.8.1",
    py_version="py3",
    instance_count=1,
    instance_type="ml.p3.2xlarge",
    output_path=s3_output_path,
    hyperparameters={
        "train_input_file": s3_train_input,
        "train_target_file": s3_train_target,
        "eval_input_file": s3_eval_input,
        "eval_target_file": s3_eval_target,
        "epochs": 5,
        "batch_size": 2,
        "output_dir": "/opt/ml/model",
        "learning_rate": 1e-5,
    },
)

# Run the training job
estimator.fit()