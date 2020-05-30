from datetime import datetime
import cv2
import numpy as np
from sklearn.cluster  import KMeans,MiniBatchKMeans
class FeatureGetter(object):
    def __init__(self):
        self.sift_det = cv2.xfeatures2d.SIFT_create()
    def get_img(self, img_path):
        img = cv2.imread(img_path)
        return img
    def get_feature(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = self.sift_det.detectAndCompute(gray, None)

        return kp, des

class FeaturesBuilder(object):
    def __init__(self,num_words,img_paths,dataset_matrix=None):
        self.num_words = num_words   #聚类中心
        self.img_paths =img_paths
        self.dataset_matrix =dataset_matrix  #dataset_matrix：图像数据的矩阵表示   注：img_paths dataset_matrix这两个参数只需要指定一个
    def getClusterCentures(self):
        start_time = datetime.now()  # 测试时间
        feature_getter  =  FeatureGetter()
        des_list = []  # 特征描述
        des_matrix = np.zeros((1, 128))
        if self.img_paths != None:
            for path in self.img_paths:
                # kp表示输入的关键点，des表示输出的sift特征向量，通常是128维的。  检测发现是N*128，N是变动的
                kp, des = feature_getter.get_feature(path)
                if des.any() != None:
                    des_matrix = np.row_stack((des_matrix, des))
                des_list.append(des)
        elif self.dataset_matrix != None:
            for gray in range(self.dataset_matrix.shape[0]):
                sift_det = cv2.xfeatures2d.SIFT_create()
                kp, des = sift_det.detectAndCompute(gray, None)
                if des != None:
                    des_matrix = np.row_stack((des_matrix, des))
                des_list.append(des)
        else:
            raise ValueError('输入不合法')
        des_matrix = des_matrix[1:, :]  # the des matrix of sift
        # 计算聚类中心  构造视觉单词词典
        # kmeans = KMeans(n_clusters=self.num_words , random_state=33)
        kmeans =  MiniBatchKMeans(n_clusters=self.num_words , batch_size=200, random_state= 33)   #MiniBatchKMeans  加速优化
        kmeans.fit(des_matrix)
        centres = kmeans.cluster_centers_  # 视觉聚类中心
        elapsed_time = datetime.now() - start_time  # 需要的时间
        print(" 获取聚类中心total_time ", elapsed_time, )
        return centres, des_list
        #

class GetFeatures(object):
    # 将特征描述转换为特征向量
    def des2feature(self,des, NUM_WORDS, centures):
        # des:单幅图像的SIFT特征描述  centures:聚类中心坐标  centures:聚类中心坐标   NUM_WORDS*128
         # return: feature vector 1*NUM_WORDS
        img_feature_vec = np.zeros((1, NUM_WORDS), 'float32')
        for i in range(des.shape[0]):  # 遍历所有图片
            feature_k_rows = np.ones((NUM_WORDS, 128), 'float32')
            feature = des[i]
            feature_k_rows = feature_k_rows * feature
            feature_k_rows = np.sum((feature_k_rows - centures) ** 2, 1)
            index = np.argmax(feature_k_rows)
            img_feature_vec[0][index] += 1
        return img_feature_vec


    # 获取所有图片的特征向量
    def get_all_features(self,des_list, num_word,centres):
        # start_time = datetime.now()  # 测试时间
        allvec = np.zeros((len(des_list), num_word), 'float32')
        for i in range(len(des_list)):
            if des_list[i].any() != None:
                allvec[i] = self.des2feature(des_list[i], num_word,centres)
        # elapsed_time = datetime.now() - start_time  # 需要的时间
        # print(" 将特征描述转换为特征向量total_time ", elapsed_time, )
        return allvec
