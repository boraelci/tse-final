import sagemaker
from sagemaker.pytorch import PyTorch

# Set your S3 bucket and dataset paths
s3_bucket = "sagemaker-training-bora"
s3_prefix = "input"
s3_output_path = f"s3://{s3_bucket}/output"

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
    checkpoint_s3_uri=f"s3://{s3_bucket}/checkpoints",
    checkpoint_local_path="/opt/ml/checkpoints",
    hyperparameters={
        "input_dir": f"s3://{s3_bucket}/{s3_prefix}",
        "corpus_type": "raw",
        "epochs": 5,
        "batch_size": 2,
        "output_dir": "/opt/ml/model",
        "learning_rate": 1e-5,
        "read_limit": 15000,
        "checkpoint_dir": "/opt/ml/checkpoints",
    },
)

# Run the training job
estimator.fit(wait=True, logs=True)
