import pandas as pd, numpy as np, json, os
from sklearn.metrics import cohen_kappa_score

RUBRICS = [
    "task_1",
    "content_1",
    "content_2",
    "content_3",
    "organization_1",
    "organization_2",
    "expression_1",
    "expression_2",
]

def evaluate_results(csv_path: str, save_dir: str):
    df = pd.read_csv(csv_path, encoding='utf-8')
    labels, preds_token, preds_weighted = [], [], []
    total_test = len(df['sample_idx'].unique())

    for i in range(total_test):
        ce_results = df[df['sample_idx'] == i]
        label = [int(l) for l in ce_results['label'].values[0][:15].split(" ")]
        preds_t, preds_w = [], []
        for pos in range(1, 17, 2):
            try:
                tok_str = ce_results[ce_results['gen_pos'] == pos]['chosen_token'].values[0]
                tok = int(tok_str)
            except Exception:
                # chosen_token이 비정상일 경우 (공백, NaN 등) → 확률값 최대인 토큰 선택
                row = ce_results[ce_results['gen_pos'] == pos]
                probs = [row[f'prob_{k}'].values[0] for k in range(1, 10)]
                tok = int(np.argmax(probs) + 1)  # prob_1~prob_9 → 토큰 1~9
                print(f"⚠️ Warning: Invalid chosen_token '{tok_str}' at sample_idx {i}, gen_pos {pos}. Using max probability token instead.")
            # weighted 계산
            weighted = sum(
                ce_results[ce_results['gen_pos'] == pos][f'prob_{k}'].values[0] * k
                for k in range(1, 10)
            )
            preds_t.append(tok)
            preds_w.append(weighted)
        labels.append(label)
        preds_token.append(preds_t)
        preds_weighted.append(preds_w)

    labels = np.array(labels).flatten()
    preds_token = np.array(preds_token).flatten()
    preds_weighted = np.rint(preds_weighted).flatten()

    res_token, res_weighted = {}, {}
    res_token['overall'] = cohen_kappa_score(labels, preds_token, weights='quadratic')
    res_weighted['overall'] = cohen_kappa_score(labels, preds_weighted, weights='quadratic')

    labels = labels.reshape(-1, 8)
    preds_token = preds_token.reshape(-1, 8)
    preds_weighted = preds_weighted.reshape(-1, 8)

    for i, rubric in enumerate(RUBRICS):
        res_token[rubric] = cohen_kappa_score(labels[:, i], preds_token[:, i], weights='quadratic')
        res_weighted[rubric] = cohen_kappa_score(labels[:, i], preds_weighted[:, i], weights='quadratic')

    # 콘솔 출력
    print("\n=== CE Token Scores ===")
    for k, v in res_token.items():
        print(f"{k}: {v:.4f}")
    token_avg = np.mean(list(res_token.values()))
    print(f"Average: {token_avg:.4f}")
    res_token["average"] = token_avg


    print("\n=== CE Weighted Scores ===")
    for k, v in res_weighted.items():
        print(f"{k}: {v:.4f}")
    weighted_avg = np.mean(list(res_weighted.values()))
    print(f"Average: {weighted_avg:.4f}")
    res_weighted["average"] = weighted_avg

    # JSON 저장
    result_dict = {"token": res_token, "weighted": res_weighted}
    out_path = os.path.join(save_dir, "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
    print(f"\n📁 평가 결과 저장 완료: {out_path}")
