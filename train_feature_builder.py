# Author:yifan
from datetime import datetime
from sklearn.cluster  import KMeans
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
from images_retrieval import dataLoader
from images_retrieval import features_builder

NUM_WORDS = 250  #聚类中心数量
NUM_TRAIN = 50  #每个类别选取训练的样本数量

DataLoader = dataLoader.DataLoader('D:\\datatest\\101_ObjectCategories\\',NUM_TRAIN)
img_paths= DataLoader.load_data()
print(len(img_paths))
featureGetter = features_builder.FeaturesBuilder(NUM_WORDS,img_paths,None)
centres,des_list=featureGetter.getClusterCentures()
# print(des_list[0].shape)
getFeatures = features_builder.GetFeatures()
img_features=getFeatures.get_all_features(des_list, NUM_WORDS,centres)
# print(img_features)
joblib.dump((centres, des_list,img_features), "imgs_features.pkl", compress=3)

