视频人脸裁剪项目
🚀 项目概述
本项目提供高效的多进程视频人脸检测与裁剪解决方案，支持批量处理视频文件，自动识别人脸区域并生成裁剪后的输出。

⚙️ 安装指南
环境配置
创建conda环境：
conda create -n crop python=3.10
conda activate crop
pip install --upgrade scenedetect[opencv]
安装FFmpeg：apt-get install ffmpeg
模型准备
项目依赖的预训练模型 face_landmarker_v2_with_blendshapes.task 已预先下载完成。

🏃 使用说明
多进程批量处理模式
创建输入/输出文件夹结构：
建立10个输入文件夹存放原始视频
建立对应的10个输出文件夹存放裁剪结果

通过以下命令启动并行处理：bash parallel_run.sh
提示：可通过修改代码中的进程数参数调整并行度

单进程模式
准备输入输出目录后运行：python extract_face_pipeline.py

📂 文件结构说明
项目根目录/
├── face_landmarker_v2_with_blendshapes.task  # 人脸关键点检测模型
├── parallel_run.sh                           # 并行处理脚本
├── extract_face_pipeline.py                  # 单进程处理主程序
└── ...                                       # 其他支持文件
建议运行时保持合理的输入/输出文件夹对应关系以确保处理结果有序存储。
  
