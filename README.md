# Img_Retriveler
Img_retriveler，基于SIFT，BoW实现图像检索


数据集选择
选用了101_ObjectCategories的图片作为本次模型训练数据。数据集下载地址：http://www.vision.caltech.edu/Image_Datasets/Caltech101/。 本文选取101类图片，每类选取50张进行实验。
代码环境
1)	python  3.7
2)	Numpy  1.16.2
3)	Matplotlib  3.0.3
4)	opencv-contrib-python   3.4.2.17
5)	scikit-learn  0.20.4

模块说明：
dataLoader.py 模块：主要用于提取101_ObjectCategories图片数据，用数组形式反馈包含图片数据的路径；
features_builder.py 模块：主要基于SIFT，K-means和Bow处理图像，获取所有图片的特征向量，其中K取250；
train_features_builder.py模块：主要基于数据，调用features_builder.py，训练数据特征，并保存为pkl文件，即图所示的“imgs_features.pkl”；
Img_retriveler.py模块：根据要搜索的主要图片，载入pkl的图片特征，比对后，反馈回12张最近内容的图片作为检索结果。

调用：
先运行：train_features_builder.py
再运行：Img_retriveler.py
