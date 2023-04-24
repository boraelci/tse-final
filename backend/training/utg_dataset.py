import boto3
from torch.utils.data import Dataset


class UtgDataset(Dataset):
    def __init__(
        self, tokenizer, input_path, target_path, max_length=1024, offset=0, limit=-1
    ):
        self.tokenizer = tokenizer
        self.input_path = input_path
        self.target_path = target_path
        self.max_length = max_length

        self.inputs = self.read_file(input_path, offset, limit)
        self.targets = self.read_file(target_path, offset, limit)

    def read_file(self, path, offset, limit):
        if path.startswith("s3://"):
            return self.read_s3_file(path, offset, limit)
        else:
            return self.read_local_file(path, offset, limit)

    def read_local_file(self, local_path, offset, limit):
        lines = None
        with open(local_path, "r") as f:
            lines = f.readlines()
        return self.cut_lines(local_path, lines, offset, limit)

    def read_s3_file(self, s3_path, offset, limit):
        s3 = boto3.client("s3")
        bucket, key = self.parse_s3_path(s3_path)
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read().decode("utf-8")
        lines = data.splitlines()
        return self.cut_lines(s3_path, lines, offset, limit)

    def parse_s3_path(self, s3_path):
        s3_path = s3_path.replace("s3://", "")
        bucket, key = s3_path.split("/", 1)
        return bucket, key

    def cut_lines(self, path, lines, offset, limit):
        if "eval" in path:  # Validation is only 1/8th of the size of the training set
            scale = 0.125
            if limit == -1:
                limit = len(lines)
            offset = int(offset * scale)
            limit = int(limit * scale)
            return lines[offset:limit]
        else:
            return lines[offset:limit]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx].strip()
        target = self.targets[idx].strip()

        tokenized_input = self.tokenizer(
            input,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        tokenized_target = self.tokenizer(
            target,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        return tokenized_input, tokenized_target
