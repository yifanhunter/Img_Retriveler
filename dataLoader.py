import  os
class DataLoader(object):
    def __init__(self, imgs_data_path,train_nums):
        self.path = imgs_data_path
        self.nums = train_nums
    def load_data(self):
        train_path = self.path  # 训练样本文件夹路径
        category_names = os.listdir(train_path)  #外部路径显示种类
        # img_path_categorys = []   # 外部路径
        img_paths = []   #具体图片路径
        for  category_name in category_names:
            img_path_category = os.path.join(train_path,category_name)  #到种类文件夹路径
            img_names = os.listdir(img_path_category)
            i = 0
            for img_name in img_names:
                i += 1
                if i < self.nums:
                    img_path = os.path.join(img_path_category,img_name)
                    img_paths.append(img_path)
                else :break
        return img_paths
# DataLoader = DataLoader('D:\\datatest\\101_ObjectCategories\\',20)
# a= DataLoader.load_data()
# print(len(a))
# print(a)