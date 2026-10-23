
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "data" / "train.jsonl",
    ROOT / "data" / "validation.jsonl",
]


def validate_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    count = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(
                    f"{path.name}: empty line {line_number}"
                )

            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name}:{line_number}: invalid JSON: {error}"
                ) from error

            messages = example.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(
                    f"{path.name}:{line_number}: missing or invalid messages list"
                )

            if messages[0].get("role") != "user":
                raise ValueError(
                    f"{path.name}:{line_number}: first role must be user"
                )

            if messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"{path.name}:{line_number}: final role must be assistant"
                )

            for msg in messages:
                content = msg.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        f"{path.name}:{line_number}: message content must be non-empty string"
                    )

            count += 1

    if count == 0:
        raise ValueError(f"{path.name}: dataset is empty")

    print(f"PASS: {path.name} — {count} examples")


if __name__ == "__main__":
    for dataset_file in FILES:
        validate_file(dataset_file)

    print("Dataset structure checks passed.")
