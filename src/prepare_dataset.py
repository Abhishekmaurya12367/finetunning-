
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

train_examples = [
    {
        "question": "Explain a linked list in simple words.",
        "answer": "A linked list is a data structure made of nodes. Each node stores data and a pointer to the next node."
    },
    {
        "question": "What is a stack in DSA?",
        "answer": "A stack follows LIFO: Last In, First Out. Its common operations are push, pop, and top."
    },
    {
        "question": "Explain binary search and its complexity.",
        "answer": "Binary search repeatedly halves a sorted search interval. Its time complexity is O(log n)."
    },
    {
        "question": "What is BFS?",
        "answer": "Breadth-First Search traverses a graph level by level, typically using a queue."
    },
    {
        "question": "Explain a hash map.",
        "answer": "A hash map stores key-value pairs and usually provides average O(1) lookup, insertion, and deletion."
    },
]

validation_examples = [
    {
        "question": "What is a queue?",
        "answer": "A queue follows FIFO: First In, First Out. Items are added at the rear and removed from the front."
    },
    {
        "question": "What is a binary tree?",
        "answer": "A binary tree is a tree in which each node has at most two children."
    },
]


def to_gemini_example(example):
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": example["question"]}],
            },
            {
                "role": "model",
                "parts": [{"text": example["answer"]}],
            },
        ]
    }


def save_jsonl(examples, filename):
    path = DATA_DIR / filename

    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            row = to_gemini_example(example)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Created: {path} ({len(examples)} examples)")


if __name__ == "__main__":
    save_jsonl(train_examples, "train.jsonl")
    save_jsonl(validation_examples, "validation.jsonl")
