import os
# 仅显示 ERROR 级别及以上日志，屏蔽 INFO 和 WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import cv2
import sys
import numpy as np
from tensorflow.keras.models import load_model

# 设置模型与 Cascade 路径
dl_dir = "downloads"
model_path = os.path.join(dl_dir, "emotion_model.h5")
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# 检查模型文件
if not os.path.exists(model_path):
    print(f"[ERROR] 未找到情绪模型文件：{model_path}")
    print("请先将 emotion_model.h5 放到 downloads/ 目录下，或修改 model_path 指向正确位置。")
    sys.exit(1)

# 检查 Cascade 文件
if not os.path.exists(cascade_path):
    print(f"[ERROR] 未在默认路径找到 Cascade 文件：{cascade_path}")
    sys.exit(1)

# 加载模型与人脸检测器
face_cascade = cv2.CascadeClassifier(cascade_path)
model = load_model(model_path, compile=False)

# 情绪标签映射
emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] 无法打开摄像头")
    sys.exit(1)

print("[INFO] 按 ‘q’ 键退出。")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转灰度图并检测人脸
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        # 调整为模型输入 64×64
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float32") / 255.0
        # 增加 batch 维度与通道维度
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # 预测情绪
        preds = model.predict(roi, verbose=0)[0]
        label = emotion_dict[np.argmax(preds)]

        # 绘制人脸框及情绪标签
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
