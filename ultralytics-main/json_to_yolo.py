import json
import os
# -*- coding: gbk -*-
# 定义类别名称到类别 ID 的映射
class_name_to_id = {
    '脱落': 0,
    '掉漆': 1
}

def convert_labelme_to_yolo(json_file, output_dir):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_width = data['imageWidth']
    image_height = data['imageHeight']

    yolo_lines = []
    for shape in data['shapes']:
        label = shape['label']
        if label not in class_name_to_id:
            print(f"Warning: Unknown class {label} in {json_file}")
            continue
        class_id = class_name_to_id[label]

        points = shape['points']
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # 计算边界框的中心点坐标和宽高
        center_x = (x_min + x_max) / (2 * image_width)
        center_y = (y_min + y_max) / (2 * image_height)
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # 确保坐标值在 [0, 1] 范围内
        center_x = max(0, min(center_x, 1))
        center_y = max(0, min(center_y, 1))
        width = max(0, min(width, 1))
        height = max(0, min(height, 1))

        line = f"{class_id} {center_x} {center_y} {width} {height}\n"
        yolo_lines.append(line)

    base_name = os.path.basename(json_file).replace('.json', '.txt')
    out_path = os.path.join(output_dir, base_name)
    with open(out_path, 'w', encoding='utf-8') as file:
        file.writelines(yolo_lines)

def convert_all_json_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            json_file = os.path.join(input_dir, filename)
            convert_labelme_to_yolo(json_file, output_dir)

if __name__ == "__main__":
    # 输入 JSON 文件所在的目录
    input_dir = 'ultralytics/cfg/datasets/data-wh/labels/val'
    # 输出 TXT 文件的目录
    output_dir = 'ultralytics/cfg/datasets/data-wh/labels/val'
    convert_all_json_files(input_dir, output_dir)