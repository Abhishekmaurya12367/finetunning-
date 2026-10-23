import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_FILE = ROOT / "file_ids.json"


def main():
    client = OpenAI()
    ids = {}
    for key, name in (("train", "train.jsonl"), ("validation", "validation.jsonl")):
        with open(DATA_DIR / name, "rb") as f:
            uploaded = client.files.create(file=f, purpose="fine-tune")
        ids[key] = uploaded.id
        print(f"Uploaded {name} -> {uploaded.id}")

    OUT_FILE.write_text(json.dumps(ids, indent=2))


if __name__ == "__main__":
    main()