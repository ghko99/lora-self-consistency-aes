import os, random, numpy as np, torch, datetime, re
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

def make_unique_output_dir(baseline: bool, ratio: float, ntl_weight: float, emo_weight: float, loss_type: str, is_mtl: bool) -> str:
    tag = "baseline" if baseline else f"ntl_{ntl_weight}_emo_{emo_weight}"
    ratio_tag = f"ratio_{ratio}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"./runs/{sanitize_name(tag)}_{sanitize_name(ratio_tag)}{('_mtl' if is_mtl else '')}_{timestamp}_{loss_type}"
    os.makedirs(dir_name, exist_ok=True)
    
    return dir_name

def init_wandb(baseline: bool, ratio: float, ntl_weight: float, emo_weight: float, output_dir: str, is_mtl: bool, no_wandb: bool = False):
    if no_wandb:
        return

    project_name = sanitize_name(f"2026-02-07-llama_aes_mtl_full_{'baseline' if baseline else 'NTL'}")
    run_name = sanitize_name(f"{'baseline' if baseline else f'ntl_{ntl_weight}'}_emo_{emo_weight}_r{ratio}{'_mtl' if is_mtl else ''}_{datetime.datetime.now().strftime('%m%d_%H%M')}")
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
def train_model(baseline: bool = False, 
                ntl_weight: float = 2.0, 
                emo_weight: float = 0.1, 
                ratio: float = 1.0, 
                loss_type: str = "mse", 
                is_mtl: bool = False, 
                resume_checkpoint: str = None,
                no_wandb: bool = False,
                base_model_name: str = None,
                dry_run: bool = False):
    set_seed(42)
    model_name = os.path.expanduser(base_model_name)

    ckpt_to_resume = None
    if resume_checkpoint is not None:
        # ✅ run 폴더를 그대로 output_dir로 사용
        output_dir = resume_checkpoint

        # ✅ 그 안에서 가장 최신 checkpoint-xxxx 자동 탐색
        ckpt_to_resume = get_last_checkpoint(output_dir)
        if ckpt_to_resume is None:
            raise ValueError(f"No checkpoint found under: {output_dir}")
    else:
        output_dir = make_unique_output_dir(baseline, ratio, ntl_weight, emo_weight, loss_type, is_mtl)
    
    if is_mtl:
        max_seq_length = 2048
    else:
        max_seq_length = 1250

    use_bf16 = torch.cuda.is_bf16_supported()

    init_wandb(baseline, ratio, ntl_weight, emo_weight, output_dir, is_mtl, no_wandb=no_wandb)

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
