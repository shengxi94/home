import os

# 图像文件目录
image_dir = r"D:\phpstudy_pro\WWW\yoloapi.local\ultralytics-main\ultralytics\cfg\datasets\data-wh\image\train"
# 标签文件目录
label_dir = r"D:\phpstudy_pro\WWW\yoloapi.local\ultralytics-main\ultralytics\cfg\datasets\data-wh\labels\train"

# 获取图像文件列表
image_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
# 获取标签文件列表
label_files = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')]

# 检查是否有不匹配的文件
missing_images = [f for f in label_files if f not in image_files]
missing_labels = [f for f in image_files if f not in label_files]

if missing_images:
    print(f"以下标签文件没有对应的图像文件: {missing_images}")
if missing_labels:
    print(f"以下图像文件没有对应的标签文件: {missing_labels}")

if not missing_images and not missing_labels:
    print("所有图像文件和标签文件都匹配。")