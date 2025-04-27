import logging
from flask import Flask, request, jsonify, send_file
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
from flask_cors import CORS
import matplotlib.pyplot as plt

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] ='results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 允许跨域请求
CORS(app)

# 加载YOLOv8模型
model = YOLO('D:/phpstudy_pro/WWW/yoloapi.local/ultralytics-main/runs/segment/train18/weights/best.pt')  # 自定义模型路径

# 建筑物类别映射
building_classes = {
    0: 'Rebar-Exposure',
    1: 'Corrosion',
    2: 'Efflorescence',
    3: 'Cracked'
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4','mov', 'avi'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def draw_annotations(image, detections):
    """在图像上绘制检测框和标签"""
    annotated_image = image.copy()
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)

        # 绘制边界框
        color = (0, 255, 0)  # 绿色
        thickness = 2
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)

        # 绘制标签背景
        label = f"{detection['name']} {detection['confidence']}%"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)

        # 绘制标签文本
        cv2.putText(annotated_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return annotated_image


def preprocess_frame(frame):
    """图像预处理以提高检测率"""
    # 1. 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 直方图均衡化
    equalized = cv2.equalizeHist(gray)

    # 3. 边缘增强
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 4. 合并回BGR格式
    enhanced = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 5. 与原图叠加
    result = cv2.addWeighted(frame, 0.7, enhanced, 0.3, 0)

    return result


def generate_thermal_map(frame):
    """生成热成像图谱"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thermal_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal_map


def merge_frames(original_frame, thermal_frame):
    """融合原始帧和热成像帧"""
    alpha = 0.5
    merged_frame = cv2.addWeighted(original_frame, 1 - alpha, thermal_frame, alpha, 0)
    return merged_frame


def detect_buildings(image_path):
    """使用YOLOv8检测建筑物并返回结果和标注图像（两种效果）"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    # 预处理图像
    processed_img = preprocess_frame(img)

    # 检测建筑物 - 降低置信度阈值
    results = model.predict(processed_img, conf=0.5)

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            if class_id in building_classes:  # 只处理建筑物类别
                detections.append({
                    'name': building_classes[class_id],
                    'confidence': round(confidence * 100, 1),  # 转为百分比
                    'bbox': box.xyxy.tolist()[0]  # 边界框坐标
                })

    # 生成带标注的普通图像
    annotated_image_normal = None
    if detections:
        annotated_image_normal = draw_annotations(img, detections)

    # 生成热成像图谱
    thermal_frame = generate_thermal_map(img)
    # 融合原始帧和热成像帧
    merged_frame = merge_frames(img, thermal_frame)

    # 生成带标注的热成像图像
    annotated_image_thermal = None
    if detections:
        annotated_image_thermal = draw_annotations(merged_frame, detections)

    return detections, annotated_image_normal, annotated_image_thermal


def process_video(video_path):
    """处理视频文件，逐帧检测建筑物（两种效果）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出视频文件路径（普通）
    output_video_path_normal = os.path.join(app.config['RESULT_FOLDER'], f"annotated_{os.path.basename(video_path)}")
    # 输出视频文件路径（热成像）
    output_video_path_thermal = os.path.join(app.config['RESULT_FOLDER'], f"annotated_thermal_{os.path.basename(video_path)}")

    logger.info(f"标注视频（普通）保存路径: {output_video_path_normal}")
    logger.info(f"标注视频（热成像）保存路径: {output_video_path_thermal}")

    # 使用avc1编码
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_normal = cv2.VideoWriter(output_video_path_normal, fourcc, fps, (frame_width, frame_height))
    out_thermal = cv2.VideoWriter(output_video_path_thermal, fourcc, fps, (frame_width, frame_height))

    all_results = []
    current_frame = 0
    frame_confidences = []  # 用于存储每一帧的置信率

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理帧图像
        processed_frame = preprocess_frame(frame)

        # 检测建筑物
        results = model.predict(processed_frame, conf=0.3)

        frame_detections = []
        frame_max_confidence = 0  # 用于记录当前帧的最大置信率
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                if class_id in building_classes:
                    frame_detections.append({
                        'name': building_classes[class_id],
                        'confidence': round(confidence * 100, 1),
                        'bbox': box.xyxy.tolist()[0],
                        'frame': current_frame,
                        'time': round(current_frame / fps, 2)
                    })
                    frame_max_confidence = max(frame_max_confidence, confidence)

        frame_confidences.append(frame_max_confidence)  # 将当前帧的最大置信率添加到列表中

        # 绘制普通标注帧
        if frame_detections:
            annotated_frame_normal = draw_annotations(frame, frame_detections)
        else:
            annotated_frame_normal = frame

        # 生成热成像图谱
        thermal_frame = generate_thermal_map(frame)
        # 融合原始帧和热成像帧
        merged_frame = merge_frames(frame, thermal_frame)

        # 绘制热成像标注帧
        if frame_detections:
            annotated_frame_thermal = draw_annotations(merged_frame, frame_detections)
        else:
            annotated_frame_thermal = merged_frame

        all_results.extend(frame_detections)

        if out_normal.write(annotated_frame_normal):
            logger.info(f"成功写入普通帧 {current_frame}")
        else:
            logger.error(f"写入普通帧 {current_frame} 失败")

        if out_thermal.write(annotated_frame_thermal):
            logger.info(f"成功写入热成像帧 {current_frame}")
        else:
            logger.error(f"写入热成像帧 {current_frame} 失败")

        current_frame += 1

    cap.release()
    out_normal.release()
    out_thermal.release()

    # 使用 matplotlib 绘制置信率图表
    if frame_confidences:
        plt.plot(frame_confidences)
        plt.xlabel('Frame Number')
        plt.ylabel('Confidence Score')
        plt.title('Confidence Score per Frame')
        confidence_plot_path = os.path.join(app.config['RESULT_FOLDER'], 'confidence_plot.png')
        plt.savefig(confidence_plot_path)
        plt.close()
    else:
        confidence_plot_path = None

    return all_results, output_video_path_normal, output_video_path_thermal, confidence_plot_path


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error','message': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error','message': 'Empty filename'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # 检查文件类型
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            detection_results, annotated_image_normal, annotated_image_thermal = detect_buildings(save_path)

            if detection_results:
                result_filename_normal = f"annotated_{filename}"
                result_filename_thermal = f"annotated_thermal_{filename}"
                result_path_normal = os.path.join(app.config['RESULT_FOLDER'], result_filename_normal)
                result_path_thermal = os.path.join(app.config['RESULT_FOLDER'], result_filename_thermal)
                cv2.imwrite(result_path_normal, annotated_image_normal)
                cv2.imwrite(result_path_thermal, annotated_image_thermal)

                response = {
                    'status':'success',
                    'results': detection_results,
                    'annotated_image_url_normal': f"/result/{result_filename_normal}",
                    'annotated_image_url_thermal': f"/result/{result_filename_thermal}"
                }
                return jsonify(response)
            return jsonify({'status': 'error','message': '未检测到建筑物'}), 400
        else:
            return jsonify({'status': 'error','message': '请使用图片上传接口'}), 400

    return jsonify({'status': 'error','message': 'File type not allowed'}), 400


@app.route('/process_video', methods=['POST'])
def process_video_route():
    if 'file' not in request.files:
        return jsonify({'status': 'error','message': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error','message': 'Empty filename'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        if filename.lower().endswith(('.mp4', '.mov', '.avi')):
            try:
                detection_results, output_video_path_normal, output_video_path_thermal, confidence_plot_path = process_video(save_path)

                if not detection_results:
                    return jsonify({'status': 'error','message': '未在视频中检测到建筑物'}), 400

                # 确保视频文件存在
                if not os.path.exists(output_video_path_normal) or not os.path.exists(output_video_path_thermal):
                    return jsonify({'status': 'error','message': '视频处理失败'}), 500

                # 构建正确的URL路径
                video_url_normal = f"/result/{os.path.basename(output_video_path_normal)}"
                video_url_thermal = f"/result/{os.path.basename(output_video_path_thermal)}"
                confidence_plot_url = f"/result/{os.path.basename(confidence_plot_path)}" if confidence_plot_path else None

                response = {
                    'status':'success',
                    'results': detection_results,
                    'annotated_video_url_normal': video_url_normal,
                    'annotated_video_url_thermal': video_url_thermal,
                    'confidence_plot_url': confidence_plot_url
                }
                return jsonify(response)

            except Exception as e:
                logger.error(f"视频处理错误: {str(e)}")
                return jsonify({'status': 'error','message': '视频处理失败'}), 500
        else:
            return jsonify({'status': 'error','message': '请使用视频上传接口'}), 400

    return jsonify({'status': 'error','message': 'File type not allowed'}), 400


@app.route('/result/<filename>')
def get_result_image(filename):
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if os.path.exists(result_path):
        if filename.lower().endswith(('.mp4', '.mov', '.avi')):
            return send_file(result_path, mimetype='video/mp4')
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return send_file(result_path, mimetype='image/jpeg')
        elif filename == 'confidence_plot.png':
            return send_file(result_path, mimetype='image/png')
    return jsonify({'status': 'error','message': 'Image or video not found'}), 400


if __name__ == '__main__':
    app.run(host='10.14.145.132', port=5000, debug=True)