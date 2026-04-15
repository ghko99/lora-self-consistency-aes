import os, random, json, numpy as np, torch, datetime, re
import wandb
from transformers import AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from .aes_dataloader import LlamaAESCollator, LlamaAESCollatorMTL
from .number_tokenizer import AutoNumberTokenizer
from .custom_trainer import CustomTrainer
from transformers.trainer_utils import get_last_checkpoint
# -------------------------------------------------
# Utils
# -------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 완전 재현성 (성능 약간 저하)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]', '_', name)

def make_unique_output_dir(use_ntl: bool, use_emo: bool, use_cbfl: bool, ratio: float, loss_type: str, is_mtl: bool) -> str:
    parts = ["ce"]
    if use_ntl: parts.append("ntl")
    if use_emo: parts.append("emo")
    if use_cbfl: parts.append("cbfl")
    tag = "+".join(parts)
    ratio_tag = f"ratio_{ratio}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"./runs/{sanitize_name(tag)}_{sanitize_name(ratio_tag)}{('_mtl' if is_mtl else '')}_{timestamp}_{loss_type}"
    os.makedirs(dir_name, exist_ok=True)

    return dir_name


def compute_score_distribution(data_path: str) -> torch.Tensor:
    """Parse training data to count occurrences of each score (1-9).

    Returns:
        Tensor of shape [9] with counts for scores 1-9.
    """
    counts = torch.zeros(9, dtype=torch.float32)
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            output = item.get("output", "")
            # First line of output contains space-separated scores
            first_line = output.strip().split("\n")[0].strip()
            for token in first_line.split():
                try:
                    score = int(token)
                    if 1 <= score <= 9:
                        counts[score - 1] += 1
                except ValueError:
                    continue
    return counts

def init_wandb(use_ntl: bool, use_emo: bool, use_cbfl: bool, ratio: float, output_dir: str, is_mtl: bool, no_wandb: bool = False):
    if no_wandb:
        return

    parts = ["ce"]
    if use_ntl: parts.append("ntl")
    if use_emo: parts.append("emo")
    if use_cbfl: parts.append("cbfl")
    loss_tag = "+".join(parts)

    project_name = sanitize_name(f"2026-02-07-llama_aes_mtl_full_{loss_tag}")
    run_name = sanitize_name(f"{loss_tag}_r{ratio}{'_mtl' if is_mtl else ''}_{datetime.datetime.now().strftime('%m%d_%H%M')}")
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_RUN_NAME"] = run_name
    wandb.init(project=project_name, name=run_name, dir=output_dir, reinit=True)
    print(f"WandB initialized → Project: {project_name}, Run: {run_name}")

def sample_ratio(dataset, ratio):
    if ratio < 1.0:
        n = max(1, int(len(dataset) * ratio))
        dataset = dataset.select(range(n))
    return dataset


# -------------------------------------------------
# Train function
# -------------------------------------------------
def train_model(use_ntl: bool = False,
                use_emo: bool = False,
                use_cbfl: bool = False,
                ratio: float = 1.0,
                loss_type: str = "mse",
                is_mtl: bool = False,
                resume_checkpoint: str = None,
                no_wandb: bool = False,
                base_model_name: str = None,
                dry_run: bool = False):
    # NTL/EMO는 항상 dynamic 모드(-1), on/off 플래그로만 제어
    ntl_weight = -1 if use_ntl else 0
    emo_weight = -1 if use_emo else 0
    cb_weight  = 1.0 if use_cbfl else 0.0
    cb_beta    = 0.9999
    cb_gamma   = 2.0

    set_seed(42)
    model_name = os.path.expanduser(base_model_name)

    ckpt_to_resume = None
    if resume_checkpoint is not None:
        output_dir = resume_checkpoint
        ckpt_to_resume = get_last_checkpoint(output_dir)
        if ckpt_to_resume is None:
            raise ValueError(f"No checkpoint found under: {output_dir}")
    else:
        output_dir = make_unique_output_dir(use_ntl, use_emo, use_cbfl, ratio, loss_type, is_mtl)

    if is_mtl:
        max_seq_length = 2048
    else:
        max_seq_length = 1250

    use_bf16 = torch.cuda.is_bf16_supported()

    init_wandb(use_ntl, use_emo, use_cbfl, ratio, output_dir, is_mtl, no_wandb=no_wandb)

    # Dataset load (train, valid, test 동일한 비율)
    if is_mtl:
        train_ds = sample_ratio(load_dataset('json', data_files="./aes_dataset_mtl/train.jsonl")['train'], ratio)
        valid_ds = sample_ratio(load_dataset('json', data_files="./aes_dataset_mtl/valid.jsonl")['train'], ratio)
        test_ds  = sample_ratio(load_dataset('json', data_files="./aes_dataset_mtl/test.jsonl")['train'], ratio)
    else:
        train_ds = sample_ratio(load_dataset('json', data_files="./aes_dataset/train.jsonl")['train'], ratio)
        valid_ds = sample_ratio(load_dataset('json', data_files="./aes_dataset/valid.jsonl")['train'], ratio)
        test_ds  = sample_ratio(load_dataset('json', data_files="./aes_dataset/test.jsonl")['train'], ratio)
    
    tokenizer = AutoNumberTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if is_mtl:
        data_collator = LlamaAESCollatorMTL(tokenizer, max_seq_length=max_seq_length)
    else:
        data_collator = LlamaAESCollator(tokenizer, max_seq_length=max_seq_length)


    # Model & LoRA config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    ))

    # Trainer args
    trainer_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        logging_steps=10,
        num_train_epochs=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        bf16=use_bf16,
        fp16=(not use_bf16),
        save_total_limit=2,
        report_to="none",
        load_best_model_at_end=True,   # ✅ Best checkpoint 자동 로드
        greater_is_better=False,      # ✅ Loss 기준으로 best 모델 선택
        seed=42,
        remove_unused_columns=False,   # ✅ collator 문제 방지
    )


    # Compute score distribution for CBFL
    class_counts = None
    if use_cbfl:
        train_data_path = "./aes_dataset_mtl/train.jsonl" if is_mtl else "./aes_dataset/train.jsonl"
        class_counts = compute_score_distribution(train_data_path)
        print(f"Score distribution (1-9): {class_counts.tolist()}")

    trainer = CustomTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        num_tokenizer=tokenizer,
        order_numbers=True,
        ntl_weight=ntl_weight,
        emo_weight=emo_weight,
        loss_type=loss_type,
        cb_weight=cb_weight,
        cb_beta=cb_beta,
        cb_gamma=cb_gamma,
        class_counts=class_counts,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    if dry_run:
        print("\n✅ Dry run successful. Model, tokenizer, and data loaded without errors. Skipping training.")
        return None, None, None, None

    if ckpt_to_resume is not None:
        print(f"Resuming from checkpoint: {ckpt_to_resume}")
        trainer.train(resume_from_checkpoint=ckpt_to_resume)
    else:
        trainer.train()
    if not no_wandb:
        wandb.finish()

    # ✅ Best model already loaded (no reloading needed)
    print(f"Training complete. Best model already loaded into memory.")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer, test_ds, tokenizer, output_dir
