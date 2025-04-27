import os

label_dir = r"D:\phpstudy_pro\WWW\yoloapi.local\ultralytics-main\ultralytics\cfg\datasets\data-wh\labels\train"
for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    print(f"{label_file}文件格式可能有误，行数为{len(parts)}")
                else:
                    class_id = parts[0]
                    try:
                        class_id = int(class_id)
                    except ValueError:
                        print(f"{label_file}中类别ID {class_id} 格式有误")
                    for i in range(1, len(parts)):
                        try:
                            coord = float(parts[i])
                            if coord < 0 or coord > 1:
                                print(f"{label_file}中坐标值 {parts[i]} 超出范围")
                        except ValueError:
                            print(f"{label_file}中坐标值 {parts[i]} 格式有误")