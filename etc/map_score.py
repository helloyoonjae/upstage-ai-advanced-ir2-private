import json

# 두 파일에서 topk 항목을 불러옴
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Average Precision (AP) 계산 (위치에 따라 감점)
def calculate_ap(true_topk, predicted_topk):
    if not predicted_topk or not true_topk:
        return 0.0
    true_item = true_topk[0]
    if predicted_topk[0] == true_item:
        return 1.0
    elif len(predicted_topk) > 1 and predicted_topk[1] == true_item:
        return 0.7  # 감점
    elif len(predicted_topk) > 2 and predicted_topk[2] == true_item:
        return 0.5  # 더 큰 감점
    else:
        return 0.0

# Mean Average Precision (MAP) 계산
def calculate_map(ground_truth_data, prediction_data):
    ap_scores = []
    for gt_item in ground_truth_data:
        pred_item = next((item for item in prediction_data if item['eval_id'] == gt_item['eval_id']), None)
        if pred_item:
            ap = calculate_ap(gt_item['topk'], pred_item['topk'])
            ap_scores.append(ap)
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0

# 파일 로드
ground_truth_file = '/data/ephemeral/home/data/highscore_hacking.csv'
prediction_file = '/data/ephemeral/home/sample_submission13_roberta_hybriddocid_synonyms.csv'

ground_truth_data = load_jsonl(ground_truth_file)
prediction_data = load_jsonl(prediction_file)

# MAP 계산
map_score = calculate_map(ground_truth_data, prediction_data)
print(f'MAP Score: {map_score}')
