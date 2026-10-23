import json
import math
from dataclasses import asdict

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from config import Config


class FineTuner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.trainer = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.use_fp16 = torch.cuda.is_available() and not self.use_bf16

        if self.device == "cpu":
            print("[INFO] No CUDA GPU detected. Running locally on CPU with FP32 precision.")
        else:
            print(f"[INFO] Running on GPU ({torch.cuda.get_device_name(0)}).")
        set_seed(cfg.train.seed)

    # ---------- components ----------
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self):
        m = self.cfg.model

        if self.device == "cuda":
            compute_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
            if m.use_4bit:
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    m.base_model,
                    quantization_config=bnb,
                    torch_dtype=compute_dtype,
                    device_map={"": 0},
                )
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=self.cfg.train.gradient_checkpointing
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    m.base_model, torch_dtype=compute_dtype, device_map="auto"
                )
        else:
            if m.use_4bit:
                print("[WARN] 4-bit quantization (QLoRA) requires an NVIDIA GPU with CUDA. Using FP32 on CPU.")
            model = AutoModelForCausalLM.from_pretrained(
                m.base_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        model.config.use_cache = False
        self.model = model

    def load_data(self):
        d = self.cfg.data
        self.dataset = load_dataset(
            "json",
            data_files={
                "train": str(self.cfg.path(d.train_file)),
                "validation": str(self.cfg.path(d.validation_file)),
            },
        )
        print(
            f"Train: {len(self.dataset['train'])} | "
            f"Validation: {len(self.dataset['validation'])}"
        )

    def build_lora(self) -> LoraConfig:
        l = self.cfg.lora
        return LoraConfig(
            r=l.r,
            lora_alpha=l.alpha,
            lora_dropout=l.dropout,
            target_modules=l.target_modules,
            task_type="CAUSAL_LM",
        )

    def build_args(self) -> SFTConfig:
        t = self.cfg.train

        # warmup_ratio was removed in newer transformers, so compute steps manually
        steps_per_epoch = math.ceil(
            len(self.dataset["train"]) / (t.batch_size * t.grad_accum)
        )
        total_steps = steps_per_epoch * t.epochs
        warmup_steps = max(1, int(total_steps * t.warmup_ratio))

        return SFTConfig(
            output_dir=str(self.cfg.output_dir / "checkpoints"),
            num_train_epochs=t.epochs,
            per_device_train_batch_size=t.batch_size,
            per_device_eval_batch_size=t.batch_size,
            gradient_accumulation_steps=t.grad_accum,
            learning_rate=t.learning_rate,
            lr_scheduler_type=t.lr_scheduler,
            warmup_steps=warmup_steps,
            logging_steps=t.logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            bf16=self.use_bf16,
            fp16=not self.use_bf16,
            gradient_checkpointing=t.gradient_checkpointing,
            max_length=t.max_length,
            seed=t.seed,
            report_to="none",
        )
    def build_trainer(self):
        self.trainer = SFTTrainer(
            model=self.model,
            args=self.build_args(),
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            processing_class=self.tokenizer,
            peft_config=self.build_lora(),
        )

    # ---------- run / save ----------
    def train(self):
        self.trainer.train()

    def evaluate(self) -> dict:
        metrics = self.trainer.evaluate()
        if "eval_loss" in metrics:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        print(f"Eval: {metrics}")
        return metrics

    def save(self, eval_metrics: dict):
        out = self.cfg.output_dir
        out.mkdir(parents=True, exist_ok=True)

        self.trainer.model.save_pretrained(str(self.cfg.adapter_dir))
        self.tokenizer.save_pretrained(str(self.cfg.adapter_dir))

        (out / "train_log.json").write_text(
            json.dumps(self.trainer.state.log_history, indent=2)
        )
        (out / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))
        (out / "config_used.json").write_text(json.dumps(asdict(self.cfg), indent=2))
        print(f"✅ Adapter saved to {self.cfg.adapter_dir}")

    def run(self):
        self.load_tokenizer()
        self.load_model()
        self.load_data()
        self.build_trainer()
        self.train()
        metrics = self.evaluate()
        self.save(metrics)