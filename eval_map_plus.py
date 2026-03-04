import os
import json
import numpy as np
from mean_average_precision import MetricBuilder


def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def update_detection_stats(predictions, groundtruths, class_stats, iou_threshold=0.5):
    class_ids = set(list(class_stats.keys()))

    for class_id in class_ids:
        preds_cls = [p for p in predictions if int(p[4]) == class_id]
        gts_cls = [g for g in groundtruths if int(g[4]) == class_id]

        preds_cls = sorted(preds_cls, key=lambda x: x[5], reverse=True)
        matched_gt = set()

        tp = 0
        fp = 0
        for pred in preds_cls:
            pred_box = pred[:4]
            best_iou = -1.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts_cls):
                if gt_idx in matched_gt:
                    continue
                current_iou = iou_xyxy(pred_box, gt[:4])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(gts_cls) - len(matched_gt)

        class_stats[class_id]["tp"] += tp
        class_stats[class_id]["fp"] += fp
        class_stats[class_id]["fn"] += fn


def safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator

# Configurações
IMG_WIDTH = 1600  #2048 city # 1600 nusc # 1920 dair # 1914 sim10k
IMG_HEIGHT = 900  #1024 city # 900 nusc  # 1080 dair # 1052 sim10k
CLASSES = {"car": 0}  # Classes de interesse

# Caminhos de entrada
detections_json_path = "/home/andremedeiros/experiments/ConfMix/runs/detect/official_SAMPLE_dolphins2nuscenes_NOCROP_NOLOAD_NOLAMBDA_PLATEAU/detections.json"
#gt_folder = "/home/andremedeiros/datasets/Cityscapes/labels/val/all_val" #labels GTs 
gt_folder = "/home/andremedeiros/datasets/nuscenes/custom_train/labels/val" #labels GTs nuScenes

# Carregar detecções``
with open(detections_json_path, "r") as f:
    detections = json.load(f)

# Inicializa mAP
metric_fn = MetricBuilder.build_evaluation_metric("map_2d", num_classes=len(CLASSES))
valid_class_ids = set(CLASSES.values())
class_stats = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in valid_class_ids}

for filename, data in detections.items():
    label_file = filename.replace(".png", ".txt").replace(".jpg", ".txt")
    gt_path = os.path.join(gt_folder, label_file)

    if not os.path.exists(gt_path):
        continue

    # Previsões com confiança
    predictions = []
    for box, label, conf in zip(data["boxes"], data["labels"], data["confidences"]):
        if label not in CLASSES:
            continue
        xmin, ymin, xmax, ymax = box
        predictions.append([xmin, ymin, xmax, ymax, CLASSES[label], conf])

    # Ground Truths
    groundtruths = []
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id not in valid_class_ids:
                continue
            x_center, y_center, w, h = map(float, parts[1:])
            xmin = (x_center - w / 2) * IMG_WIDTH
            ymin = (y_center - h / 2) * IMG_HEIGHT
            xmax = (x_center + w / 2) * IMG_WIDTH
            ymax = (y_center + h / 2) * IMG_HEIGHT
            groundtruths.append([xmin, ymin, xmax, ymax, cls_id, 0, 0])

    predictions_np = np.array(predictions, dtype=np.float32) if predictions else np.empty((0, 6), dtype=np.float32)
    groundtruths_np = np.array(groundtruths, dtype=np.float32) if groundtruths else np.empty((0, 7), dtype=np.float32)

    metric_fn.add(predictions_np, groundtruths_np)
    update_detection_stats(predictions, groundtruths, class_stats, iou_threshold=0.5)

# ------------------------
# Cálculo VOC mAP@0.5
result_voc = metric_fn.value(
    iou_thresholds=[0.5],
    recall_thresholds=np.arange(0., 1.01, 0.01)
)

# Cálculo COCO mAP@[0.5:0.95]
result_coco = metric_fn.value(
    iou_thresholds=np.arange(0.5, 1.0, 0.05),
    recall_thresholds=np.arange(0., 1.01, 0.01),
    mpolicy="soft"
)

# Impressão
print("=== RESULTADO mAP ===")
print(f"mAP@0.5: {result_voc['mAP']:.6f}") #VOC
print(f"mAP@[0.5:0.95]: {result_coco['mAP']:.6f}") #COCO
print()

# Métricas globais (IoU=0.5)
tp_total = sum(stats["tp"] for stats in class_stats.values())
fp_total = sum(stats["fp"] for stats in class_stats.values())
fn_total = sum(stats["fn"] for stats in class_stats.values())

precision_global = safe_div(tp_total, tp_total + fp_total)
recall_global = safe_div(tp_total, tp_total + fn_total)
f1_global = safe_div(2 * precision_global * recall_global, precision_global + recall_global)

print("=== MÉTRICAS ADICIONAIS (IoU=0.5) ===")
print(f"TP: {tp_total}")
print(f"FP: {fp_total}")
print(f"FN: {fn_total}")
print(f"Precision: {precision_global:.6f}")
print(f"Recall: {recall_global:.6f}")
print(f"F1-score: {f1_global:.6f}")
print()

# AP por classe (VOC)
print("AP por classe (IoU=0.5):")
voc_iou_block = result_voc.get(0.5, {})
for class_name, class_id in CLASSES.items():
    ap = voc_iou_block.get(class_id, {}).get("ap", None)
    if ap is None:
        print(f"  {class_name}: sem AP disponível")
    else:
        print(f"  {class_name}: {ap:.6f}")

print()
print("Métricas por classe (IoU=0.5):")
for class_name, class_id in CLASSES.items():
    tp = class_stats[class_id]["tp"]
    fp = class_stats[class_id]["fp"]
    fn = class_stats[class_id]["fn"]

    precision_cls = safe_div(tp, tp + fp)
    recall_cls = safe_div(tp, tp + fn)
    f1_cls = safe_div(2 * precision_cls * recall_cls, precision_cls + recall_cls)

    print(f"  {class_name} -> TP: {tp}, FP: {fp}, FN: {fn}, Precision: {precision_cls:.6f}, Recall: {recall_cls:.6f}, F1: {f1_cls:.6f}")

print()
print("Best-F1 por classe a partir da curva PR (IoU=0.5):")
for class_name, class_id in CLASSES.items():
    class_curve = result_voc.get(0.5, {}).get(class_id, {})
    precision_curve = np.array(class_curve.get("precision", []), dtype=np.float32)
    recall_curve = np.array(class_curve.get("recall", []), dtype=np.float32)

    if len(precision_curve) == 0 or len(recall_curve) == 0:
        print(f"  {class_name}: sem pontos na curva PR")
        continue

    f1_curve = (2 * precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-16)
    best_idx = int(np.argmax(f1_curve))

    print(
        f"  {class_name}: best_F1={f1_curve[best_idx]:.6f}, "
        f"precision={precision_curve[best_idx]:.6f}, recall={recall_curve[best_idx]:.6f}"
    )
