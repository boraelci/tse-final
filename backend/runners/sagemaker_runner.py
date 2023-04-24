import sagemaker
from sagemaker.pytorch import PyTorch

class SagemakerRunner:
    def __init__(self, s3_bucket_name, args):
        self.input_dir = f"s3://{s3_bucket_name}/input"
        self.output_path = f"s3://{s3_bucket_name}/output"
        self.checkpoints_s3_uri = f"s3://{s3_bucket_name}/checkpoints"
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.offset = args.offset
        self.limit = args.limit

    def run(self, train_input_path, train_target_path, eval_input_path, eval_target_path):

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
            output_path=self.output_path,
            checkpoint_s3_uri=self.checkpoints_s3_uri,
            checkpoint_local_path="/opt/ml/checkpoints",
            hyperparameters={
                "train_input_path": train_input_path,
                "train_target_path": train_target_path,
                "eval_input_path": eval_input_path,
                "eval_target_path": eval_target_path,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "output_dir": "/opt/ml/model",
                "learning_rate": self.learning_rate,
                "offset": self.offset,
                "limit": self.limit,
                "checkpoint_dir": "/opt/ml/checkpoints",
            },
        )

        # Run the training job
        estimator.fit(wait=True, logs=True)
