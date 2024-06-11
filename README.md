# 基于Emotion2Vec模型的批量音频情感自动标注工具
## 简介
基于[emotion2vec](https://www.modelscope.cn/models/iic/emotion2vec_base_finetuned/summary)对输入的音频进行情绪八分类（生气、厌恶、恐惧、开心、中立、其他、难过、吃惊）或[emotion2vec+large](https://www.modelscope.cn/models/iic/emotion2vec_plus_large/summary)对输入的音频进行情绪进行五分类（生气、开心、中性、伤心、未知）

此项目包含一个音频时长筛选/批量重命名脚本preprocess_audio.py批量推理脚本recognize.py和一个音频分类脚本classify.py和一个webui.py界面

## 依赖项

- Python 3.10.8
- 安装所需依赖`pip install -r requirements.txt`
## 快速使用

如果您想快速使用可以使用此[打包好的文件](https://www.123pan.com/s/BYgpjv-xVmJv.html)
数据集格式可以参考此[数据集](https://github.com/AI-Hobbyist/Genshin_Datasets)，或者在音频名上标注音频内容否则无法使用文本情感识别功能。
输入可以参考此格式:
```
input
└───speaker
   ├───xxx.wav
   └─── xxx.lab
```
参考输出为此格式:
```
output
└───speaker0
   └───emotion
       └───【emotion】{text}      
```
经过处理后的文件将会被自动归类并重命名，以反映出音频中包含的情感及其相关内容。一个典型的输出文件示例如下：

- `output/小明/生气/【生气】我很生气.wav`
- `output/小明/开心/【开心】我很开心.wav`


