# QImage
基于 CLIP 实现的简易图像特征提取与图像检索 Demo。

## 简介
本项目利用 OpenAI CLIP 模型对图片进行向量化编码，生成图像特征向量并持久化存储，实现基础的以图搜图功能，代码精简，适合学习图像检索、特征嵌入相关入门使用。

## 环境依赖
```bash
pip install torch torchvision pillow numpy
```

## 使用说明
1. 将待检索图片放入指定图片目录
2. 运行主程序，自动提取图像特征并保存至 `image_embeddings.pkl`
3. 输入查询图片，程序比对特征向量，返回相似图像结果

## 项目结构
```
QImage/
├── CLIP/            # CLIP 模型相关代码
├── main.py          # 主运行入口
├── lib.py           # 工具函数
├── *.ipynb          # 交互式演示笔记
└── image_embeddings.pkl  # 持久化图像特征文件
```

## 备注
- 项目为入门演示 Demo，仅实现基础检索能力
- 首次运行会自动加载 CLIP 预训练权重
- 可根据需求拓展批量处理、前端展示、向量数据库等功能
