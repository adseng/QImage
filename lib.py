import os
import torch
import numpy as np
from PIL import Image
import clip
import pickle

IMAGE_EMBEDDINGS_PKL = "image_embeddings.pkl"

# 加载CLIP模型和预处理函数
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device, jit=False, download_root='./clip_models')

image_dir = 'imgs'


# 优化
# 1，提前加载和编码所有图像,保存到缓存，多线程编码
# 2，计算动态缩放因子,根据图像大小动态调整
# 3，查询文本时直接使用已编码图像编码
# 4，保存最终结果
# 5，每次有新图片添加时,对新图片编码缓存
def preprocess_image():
    image_embeddings = []
    if os.path.exists("%s" % IMAGE_EMBEDDINGS_PKL):
        with open(IMAGE_EMBEDDINGS_PKL, "rb") as f:
            image_embeddings = pickle.load(f)
            # 取出image_embeddings中所有filename
    filenames_already = []
    if len(image_embeddings) > 0:
        filenames_already = [image_embedding['filename'] for image_embedding in image_embeddings]

    for filename in os.listdir(image_dir):
        # 已经编码过的图像
        if filename in filenames_already:
            continue
        # 没有编码过的图像
        # 原始图片
        original_image = Image.open(os.path.join(image_dir, filename))
        # 计算缩放因子，1024*1024的缩放因子=1
        width, height = original_image.size
        scale_factor = max(1, int(1024 * 1.0 / max(width, height)))
        # 缩放后的图片
        resized_image = original_image.resize((width // scale_factor, height // scale_factor))

        with torch.no_grad():
            # todo 多线程加快编码速度
            image_features = model.encode_image(preprocess(resized_image).unsqueeze(0).to(device))
            image_embedding = {'filename': filename, 'features': image_features}
            image_embeddings.append(image_embedding)

    with open(IMAGE_EMBEDDINGS_PKL, "wb") as f:
        pickle.dump(image_embeddings, f)
        image_embeddings = None

        # 测试代码
        # if os.path.exists("image_embeddings.pkl"):
        #     with open("image_embeddings.pkl", "rb") as f:
        #         image_embeddings = pickle.load(f)
        # else:
        #     image_embeddings = []


preprocess_image()


def encoding_image(filename: str):
    image_embeddings = []
    if os.path.exists("image_embeddings.pkl"):
        with open("image_embeddings.pkl", "rb") as f:
            image_embeddings = pickle.load(f)
            # 取出image_embeddings中所有filename
    print(len(image_embeddings))
    if len(image_embeddings) > 0:
        filenames_already = [image_embedding['filename'] for image_embedding in image_embeddings]
        if filename in filenames_already:
            return
            # 没有编码过的图像
    original_image = Image.open(os.path.join(image_dir, filename))
    # 计算缩放因子，1024*1024的缩放因子=1
    width, height = original_image.size
    scale_factor = max(1, int(1024 * 1.0 / max(width, height)))
    # 缩放后的图片
    resized_image = original_image.resize((width // scale_factor, height // scale_factor))

    with torch.no_grad():
        # todo 多线程加快编码速度
        image_features = model.encode_image(preprocess(resized_image).unsqueeze(0).to(device))
        image_embedding = {'filename': filename, 'features': image_features}
        image_embeddings.append(image_embedding)

    with open("image_embeddings.pkl", "wb") as f:
        print(len(image_embeddings))
        pickle.dump(image_embeddings, f)
        image_embeddings = None


def query_img_by_text(query=''):
    # 预处理查询文本
    text = clip.tokenize([query]).to(device)
    text_features = model.encode_text(text)

    image_embeddings = []

    if os.path.exists("image_embeddings.pkl"):
        with open("image_embeddings.pkl", "rb") as f:
            image_embeddings = pickle.load(f)

    if len(image_embeddings) == 0:
        # 抛出异常
        raise Exception("No image uploaded yet")

    # 遍历图像文件夹，计算每个图像与查询文本的相似度
    max_similarity = -np.inf
    max_path = ""

    for image_embedding in image_embeddings:
        similarity = torch.nn.functional.cosine_similarity(image_embedding['features'], text_features, dim=1)
        similarity = similarity.item()

        if similarity > max_similarity:
            max_similarity = similarity
            # max_path = image_path
            max_path = image_embedding['filename']

    # for filename in os.listdir(image_dir):
    #     image_path = os.path.join(image_dir, filename)
    #     image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    #
    #     with torch.no_grad():
    #         image_features = model.encode_image(image)
    #         # text_features = model.encode_text(text)
    #         # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze().item()
    #         similarity = torch.nn.functional.cosine_similarity(image_features, text_features, dim=1)
    #         similarity = similarity.item()
    #         print(image_path, similarity)
    #     if similarity > max_similarity:
    #         max_similarity = similarity
    #         # max_path = image_path
    #         max_path = filename

    return max_path, max_similarity


# 输出最匹配的图像路径
# print("最匹配的图像路径：", max_path)
while True:
    query = input("请输入查询文本:")
    max_path, max_similarity = query_img_by_text(query)
    print("最匹配的图像路径：", max_path)
