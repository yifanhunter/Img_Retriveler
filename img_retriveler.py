from datetime import datetime
import cv2
import numpy as np
from matplotlib import pyplot as plt
from images_retrieval import dataLoader
from images_retrieval import features_builder
from sklearn.externals import joblib
from sklearn import preprocessing

class ImgRetriveler(object):
    def __init__(self,img_features,num_close):
        self.img_features =img_features
        self.num_close = num_close

     #获取最近的图片，返回其路径
    def getNearestImg(self,feature):
        # 找出目标图像最像的几个: feature:目标图像特征, img_features: 图像数据库, NUM_CLOSE: 最近个数
        start_time = datetime.now()  # 测试时间
        #  归一化处理
        for i in range(len(feature)):
            feature[i] = feature[i] / (sum(feature[i]) * np.ones((len(feature[i])), 'float32'))  # 归一化
        for i in range(len(self.img_features)):
            self.img_features[i] = self.img_features[i] / (sum(self.img_features[i]) * np.ones((len(self.img_features[i])), 'float32'))  # 归一化

        # 是否使用tf-idf向量化
        n_occurences = np.sum((self.img_features > 0) * 1, axis=0)
        # print("TF-IDF", n_occurences)
        idf = np.array(np.log((1.0 * len(self.img_features) + 1) / (1.0 * n_occurences + 1)), 'float32')
        # 使用L2正则
        self.img_features = self.img_features * idf
        self.img_features = preprocessing.normalize(self.img_features, norm='l2')
        # 需要查的特征
        feature = feature * idf
        feature = preprocessing.normalize(feature, norm='l2')


        # # 方法一：使用余弦相似度
        # features = np.ones((self.img_features.shape[0], len(feature)), 'float32')
        # features = features * feature
        # dist = []
        # for i in range(img_features.shape[0]):
        #     dist.append(cosine_similarity([features[0], img_features[i]])[0][1])
        # dist_index = np.argsort(dist)[::-1]

        #方法二：“欧式距离”（Euclidean distance）
        # features = np.ones((self.img_features.shape[0], len(feature)), 'float32')
        # features = features * feature
        # dist = np.sum((features - self.img_features) ** 2, 1)
        # dist_index = np.argsort(dist)

        # “曼哈顿距离”（Manhattan distance）
        features = np.ones((self.img_features.shape[0], len(feature)), 'float32')
        features = features * feature
        dist=np.sum(abs(features-self.img_features),1)    #绝对值相似度
        dist_index = np.argsort(dist)


        print(sorted(dist))
        elapsed_time = datetime.now() - start_time  # 需要的时间
        print(" 计算待检索图像最近邻图像total_time ", elapsed_time, )
        print(dist_index[:self.num_close])
        return dist_index[:self.num_close]

    #展示图片
    def showImg(self,target_img_path, index, dataset_paths):
        #target_img_path:要搜索的图像的路径  dataset_paths：图像数据库所有图片的路径  显示最相似的图片集合
        paths = []
        for i in index:
            # print(i)
            paths.append(dataset_paths[i])
        plt.figure(figsize=(10, 20))  # figsize 用来设置图片大小
        plt.subplot(453), plt.imshow(plt.imread(target_img_path)), plt.title('target_image')
        for i in range(len(index)):
            plt.subplot(4, 5, i + 6), plt.imshow(plt.imread(paths[i]))
        plt.show()

    # 检索图像，展示最像的几个
    def retrieval_img(self,img_path,  centures, img_paths, num_words):
        #img_path 检索图像的路径， centures:聚类中心   img_features:图像数据库 matrix
        #  img_paths:图像数据库所有图像路径   num_close:显示最近邻的图像数目
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sift_det = cv2.xfeatures2d.SIFT_create()
        kp, des = sift_det.detectAndCompute(img, None)
        # print(len(des))
        # print(des)
        getFeatures = features_builder.GetFeatures()
        feature = getFeatures.des2feature(des, num_words, centures)
        sorted_index = self.getNearestImg(feature)
        self.showImg(img_path, sorted_index, img_paths)

if __name__ == "__main__":
    NUM_WORDS = 250  # 聚类中心数量
    NUM_TRAIN = 50 # 每个类别选取训练的样本数量
    NUM_CLOSE = 15  # 选出与目标最近的图片数量
    targe_path = 'D:\\datatest\\101_ObjectCategories\\accordion\\image_0001.jpg'
    paths = 'D:\\datatest\\101_ObjectCategories\\'
    centres, des_list, img_features = joblib.load("imgs_features.pkl")   #读取保存的特征
    print(len(des_list))

    DataLoader = dataLoader.DataLoader(paths, NUM_TRAIN)
    img_paths = DataLoader.load_data()
    print(img_paths)
    imgRetriveler = ImgRetriveler(img_features,NUM_CLOSE)
    imgRetriveler.retrieval_img(targe_path,  centres, img_paths, NUM_WORDS)
