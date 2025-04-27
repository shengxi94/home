from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n-seg.pt")  # 官方预训练模型

# 启动训练
model.train(
    data="/home/featurize/work/ultralytics-main/ultralytics/cfg/datasets/mydata.yaml",
    epochs=300,  # 适当增加训练轮数
    imgsz=640,
    batch=64,  # GPU可适当增大，CPU建议2 - 4
    workers=8,  # Windows建议≤2，Linux可增加
    device="cuda",  # 有GPU时改为 "cuda" 或 "0"
    lr0=0.01,  # 初始学习率
    optimizer="AdamW",  # 使用AdamW优化器，通常有更好的性能
    cos_lr=True,  # 使用余弦退火学习率调度策略
    patience=30,  # 增大EarlyStopping的patience值
    fliplr=0.5,  # 水平翻转数据增强
    flipud=0.2,  # 垂直翻转数据增强
    degrees=10,  # 旋转数据增强
    scale=0.2  # 缩放数据增强
)
