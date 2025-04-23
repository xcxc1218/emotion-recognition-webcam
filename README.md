# Emotion Recognition Webcam

一个基于 OpenCV 和 TensorFlow 的实时表情识别项目。  
通过笔记本/摄像头捕获人脸，实时分类以下七种情绪：

- 生气 (Angry)  
- 厌恶 (Disgust)  
- 害怕 (Fear)  
- 高兴 (Happy)  
- 伤心 (Sad)  
- 惊讶 (Surprise)  
- 平静 (Neutral)  

## 准备

1. 克隆或下载本仓库到本地。  
2. 创建并进入虚拟环境（可选）：
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   
安装依赖：

pip install opencv-python tensorflow numpy

模型
在项目根目录创建 downloads/ 文件夹：

mkdir downloads
下载预训练模型并重命名为 emotion_model.h5 放入 downloads/：

curl -L \
  https://raw.githubusercontent.com/oarriaga/face_classification/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5 \
  -o downloads/emotion_model.h5

运行

python emotion_recognition_one_click.py
打开摄像头窗口，检测到的人脸会被框住并显示情绪标签。

按 Esc 键退出。

文件结构
.
├── downloads/
│   └── emotion_model.h5
└── emotion_recognition_one_click.py
