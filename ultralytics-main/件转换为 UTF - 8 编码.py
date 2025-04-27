import os
import chardet

def convert_to_utf8(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # 检测文件的原始编码
            with open(file_path, 'rb') as f:
                rawdata = f.read()
                encoding = chardet.detect(rawdata)['encoding']

            # 如果不是UTF - 8编码，则进行转换
            if encoding!= 'utf - 8':
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    with open(file_path, 'w', encoding='utf - 8') as f:
                        f.write(content)
                    print(f"已将 {filename} 转换为UTF - 8编码")
                except UnicodeDecodeError:
                    print(f"无法转换 {filename}，可能是文件格式问题或编码检测错误")

# 替换为你的labels目录路径
labels_directory = r"D:\phpstudy_pro\WWW\yoloapi.local\ultralytics-main\ultralytics\cfg\datasets\data-wh\labels\val"
convert_to_utf8(labels_directory)