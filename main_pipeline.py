import argparse
from modules.train_module import train_model
from modules.inference_module import run_inference
from modules.evaluate_module import evaluate_results
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Baseline 학습 (NTL 미적용)")
    parser.add_argument("--ratio", type=float, default=1.0, help="데이터 비율 (0.1=10%)")
    parser.add_argument("--ntl_weights", type=float, default=2.0, help="NTL loss 가중치")
    parser.add_argument("--emo_weights", type=float, default=0.1, help="EMO loss 가중치")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "was"], help="NTL 손실 함수 유형")
    parser.add_argument("--device_id", type=int, help="사용할 GPU 장치 ID")
    parser.add_argument("--mtl", action="store_true", default=False, help="Multi-Task Learning 적용 여부")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="체크포인트에서 학습 재개 경로")
    parser.add_argument("--no_wandb", action="store_true", help="WandB 로깅 비활성화")
    parser.add_argument("--base_model_name", type=str, required=True, help="베이스 모델 경로")
    parser.add_argument("--dry_run", action="store_true", help="Dry run: 모델, 데이터 로딩까지만 확인하고 학습은 시작하지 않음")


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    # 1️⃣ Training (Trainer, test_ds, tokenizer, output_dir)
    trainer, test_ds, tokenizer, output_dir = train_model(
        baseline=args.baseline,
        ntl_weight=args.ntl_weights,
        emo_weight=args.emo_weights,
        ratio=args.ratio,
        loss_type = args.loss_type,
        is_mtl = args.mtl,
        resume_checkpoint=args.resume_checkpoint,
        no_wandb=args.no_wandb,
        base_model_name=args.base_model_name,
        dry_run=args.dry_run
    )

    if args.dry_run:
        print("Dry run finished. Exiting.")
        exit()

    # 2️⃣ Inference (best model already loaded)
    csv_path = run_inference(
        model=trainer.model,
        tokenizer=tokenizer,
        test_dataset=test_ds,
        out_dir=output_dir
    )

    # 3️⃣ Evaluation
    evaluate_results(str(csv_path), save_dir=output_dir)
