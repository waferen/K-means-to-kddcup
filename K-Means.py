import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA



# 加载数据集

def load_dataset():
    with open("feature_name.txt", "r") as file:
        lines = file.readlines()

    labels = lines[0].strip().split(",")

    print("类别总数:",len(labels))
    #print(labels)

    features_name = []
    for i in range(1,len(lines)):
        parts = lines[i].strip().split(":")#去掉首尾空格并按冒号划分
        features_name.append(parts[0].strip()) 

    print("特征总数:", len(features_name))
    #print(features_name)

    #sample = "0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal."

    #sample_list = sample.split(',') #将字符串划成列表

    ##计算列表中元素的总数
    #num_elements = len(sample_list)

    #print("样本中的特征及标签总数:", num_elements)

    # 读取CSV文件
    data = pd.read_csv("kddcup_data.csv", header=0,  names=features_name + ["label"])

    minor_classes = ['satan.', 'ipsweep.', 'portsweep.', 'nmap.', 'back.', 'warezclient.', 'teardrop.', 
                    'pod.', 'guess_passwd.', 'buffer_overflow.', 'land.', 'warezmaster.', 'imap.', 
                    'rootkit.', 'loadmodule.', 'ftp_write.', 'multihop.', 'phf.', 'perl.', 'spy.']

    data['label'] = data['label'].apply(lambda x: 'other' if x in minor_classes else x)


    # 定义需要的总样本量
    total_samples = 1000

    # 计算每个类别应抽取的样本数量
    fraction = total_samples / len(data)
    samples_per_class = data['label'].value_counts() * fraction
    print(samples_per_class)

    # 初始化一个空的DataFrame来存储抽样结果
    sampled_data = pd.DataFrame()

    # 对每个类别进行抽样
    for label, count in samples_per_class.items():
        class_data = data[data['label'] == label]#包含当前类别样本的DataFrame
        sampled_class_data = class_data.sample(int(count), random_state=42)#从当前类别的数据中随机抽取指定数量的样本
        sampled_data = pd.concat([sampled_data, sampled_class_data])

    X = sampled_data.iloc[:,:-1].values
    y = sampled_data.iloc[:,-1].values

    # 检查每一列，如果其类型不是数值型，那么进行编码
    for i in range(X.shape[1]):
        if X[:, i].dtype.kind not in 'biufc':  # 检查列类型是否为数值型（整数、布尔、无符号整数、浮点、复数）
            X[:, i], _ = integer_encode_column(X[:, i])  # 对列进行整数编码

    # 假设我们有一个名为X的数据集并且我们想降至2维
    pca = PCA(n_components=2)

    # 将PCA应用到数据集
    X = pca.fit_transform(X)

    return X


# 将数据集中的非数值列转换为整数编码

def integer_encode_column(column):
    # 确保所有的输入数据为字符类型
    column_str = column.astype(str)

    # 创建一个字典来映射元组和整数
    unique_values = np.unique(column_str)
    mapping = {val : i for i, val in enumerate(unique_values)}

    # 使用字典来转换列
    column_integer_encoded = np.array([mapping[val] for val in column_str])

    return column_integer_encoded, mapping



# 计算两点之间的欧氏距离并返回

def elu_distance(a, b):
    dist = np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))
    return dist

# 从数据集dataset中随机选取k个数据作为中心(centroids)并返回

def initial_centroids(dataset, k):
    dataset   = list(dataset)
    centroids = random.sample(dataset, k)
    return centroids


# 对dataset中的每个点item, 计算item与centroids中k个中心的距离
# 根据最小距离将item加入相应的簇中并返回簇类结果cluster

def min_distance(dataset, centroids):
    cluster = dict()
    k       = len(centroids)
    for item in dataset:
        a        = item
        flag     = -1
        min_dist = float("inf")
        for i in range(k):
            b    = centroids[i]
            dist = elu_distance(a, b)
            if dist < min_dist:
                min_dist = dist
                flag     = i
        if flag not in cluster.keys():
            cluster[flag] = []
        cluster[flag].append(item)
    return cluster

# 根据簇类结果cluster重新计算每个簇的中心
# 返回新的中心centroids

def reassign_centroids(cluster):
    # 重新计算k个质心
    centroids = []
    for key in cluster.keys():
        centroid = np.mean(cluster[key], axis=0)
        centroids.append(centroid)
    return centroids


# 计算簇内样本与各自中心的距离，累计求和
# sum_dist刻画簇内样本相似度, sum_dist越小则簇内样本相似度越高

def closeness(centroids, cluster):
    # 计算均方误差，该均方误差刻画了簇内样本相似度
    # 将簇类中各个点与质心的距离累计求和
    sum_dist = 0.0
    for key in cluster.keys():
        a    = centroids[key]
        dist = 0.0
        for item in cluster[key]:
            b     = item
            dist += elu_distance(a, b)
        sum_dist += dist
    return sum_dist



# 展示聚类结果

def show_cluster(centroids, cluster):
    # 展示聚类结果
    print("聚类中心:")
    for centroid in centroids:
        print(centroid)
    print("聚类结果:")
    for key in cluster.keys():
        print("第", key, "类:")
        for item in cluster[key]:
            print(item)
    cluster_color  = ['or', 'ob', 'og', 'ok', 'oy']
    centroid_color = ['dr', 'db', 'dg', 'dk', 'dy']

    for key in cluster.keys():
        plt.plot(centroids[key][0], centroids[key][1], centroid_color[key], markersize=12)
        for item in cluster[key]:
            plt.plot(item[0], item[1], cluster_color[key])
    plt.title("K-Means Clustering")
    plt.show()



############################################################
# K-Means算法
# 1、加载数据集
# 2、随机选取k个样本作为中心
# 3、根据样本与k个中心的最小距离进行聚类
# 4、计算簇内样本相似度，并与上一轮相似度进行比较，两者误差小于阈值，则
#    停止运行，反之则更新各类中心，重复步骤3
############################################################

def k_means(k):
    dataset      = load_dataset()
    centroids    = initial_centroids(dataset, k)
    cluster      = min_distance(dataset, centroids)
    current_dist = closeness(centroids, cluster)
    old_dist     = 0

    while abs(current_dist - old_dist) >= 0.00001:
        centroids     = reassign_centroids(cluster)
        cluster       = min_distance(dataset, centroids)
        old_dist      = current_dist
        current_dist  = closeness(centroids, cluster)
    return centroids, cluster

# 程序执行入口

if __name__ == "__main__":
    k = 4
    centroids, cluster = k_means(k)
    show_cluster(centroids, cluster)


