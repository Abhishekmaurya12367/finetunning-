
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

            contents = example.get("contents")
            if not isinstance(contents, list) or len(contents) < 2:
                raise ValueError(
                    f"{path.name}:{line_number}: invalid contents"
                )

            if contents[0].get("role") != "user":
                raise ValueError(
                    f"{path.name}:{line_number}: first role must be user"
                )

            if contents[-1].get("role") != "model":
                raise ValueError(
                    f"{path.name}:{line_number}: final role must be model"
                )

            for message in contents:
                if not isinstance(message.get("parts"), list):
                    raise ValueError(
                        f"{path.name}:{line_number}: parts must be a list"
                    )

                if not any(
                    isinstance(part.get("text"), str)
                    and part["text"].strip()
                    for part in message["parts"]
                ):
                    raise ValueError(
                        f"{path.name}:{line_number}: missing text"
                    )

            count += 1

    if count == 0:
        raise ValueError(f"{path.name}: dataset is empty")

    print(f"PASS: {path.name} — {count} examples")


if __name__ == "__main__":
    for dataset_file in FILES:
        validate_file(dataset_file)

    print("Dataset structure checks passed.")
