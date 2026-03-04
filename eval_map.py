import os
import json
import numpy as np
from mean_average_precision import MetricBuilder

# Configurações
IMG_WIDTH = 2048  #2048 city # 1600 nusc # 1920 dair # 1914 sim10k
IMG_HEIGHT = 1024  #1024 city # 900 nusc  # 1080 dair # 1052 sim10k
CLASSES = {"car": 0}  # Classes de interesse

# Caminhos de entrada
detections_json_path = "/home/andremedeiros/experiments/ConfMix/runs/detect/official_SAMPLE_dolphins2cityscapes_PLATEAU_latest_A2B/detections.json"
gt_folder = "/home/andremedeiros/datasets/Cityscapes/labels/val/all_val" #labels GTs 

# Carregar detecções``
with open(detections_json_path, "r") as f:
    detections = json.load(f)

# Inicializa mAP
metric_fn = MetricBuilder.build_evaluation_metric("map_2d", num_classes=len(CLASSES))

for filename, data in detections.items():
    label_file = filename.replace(".png", ".txt").replace(".jpg", ".txt")
    gt_path = os.path.join(gt_folder, label_file)

    if not os.path.exists(gt_path):
        continue

    # Previsões com confiança
    predictions = []
    for box, label, conf in zip(data["boxes"], data["labels"], data["confidences"]):
        xmin, ymin, xmax, ymax = box
        predictions.append([xmin, ymin, xmax, ymax, CLASSES[label], conf])

    # Ground Truths
    groundtruths = []
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])
            xmin = (x_center - w / 2) * IMG_WIDTH
            ymin = (y_center - h / 2) * IMG_HEIGHT
            xmax = (x_center + w / 2) * IMG_WIDTH
            ymax = (y_center + h / 2) * IMG_HEIGHT
            groundtruths.append([xmin, ymin, xmax, ymax, cls_id, 0, 0])

    metric_fn.add(np.array(predictions), np.array(groundtruths))

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

# AP por classe (VOC)
print("AP por classe (IoU=0.5):")
for class_id, ap in result_voc.get("per_class", {}).items():
    class_name = list(CLASSES.keys())[list(CLASSES.values()).index(class_id)]
    print(f"  {class_name}: {ap:.6f}")
