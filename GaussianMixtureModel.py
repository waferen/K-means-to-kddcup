import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, adjusted_rand_score, adjusted_mutual_info_score, v_measure_score

### 数据预处理
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

sample = "0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal."

sample_list = sample.split(',') #将字符串划成列表

# 计算列表中元素的总数
num_elements = len(sample_list)

print("样本中的特征及标签总数:", num_elements)

# 读取CSV文件并提取前100个样本
df = pd.read_csv("kddcup_data.csv", header=0,  names=features_name + ["label"])
print(df.head(5))

### 自定义高斯混合模型
class GaussianMixtureModel:
    def __init__(self, n_components=1, max_iter=100, tol=1e-4, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar  # 正则化项
    
    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = [np.eye(n_features) for _ in range(self.n_components)]
    
    def _expectation_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * self._multivariate_gaussian(X, self.means_[k], self.covariances_[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities
    
    def _maximization_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        self.weights_ = responsibilities.sum(axis=0) / n_samples
        self.means_ = np.dot(responsibilities.T, X) / responsibilities.sum(axis=0)[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means_[k]
            cov = np.dot(responsibilities[:, k] * diff.T, diff) / responsibilities[:, k].sum()
            self.covariances_[k] = cov + self.reg_covar * np.eye(n_features)  # 添加正则化项
    
    def _multivariate_gaussian(self, X, mean, cov):
        n_features = X.shape[1]
        cov += self.reg_covar * np.eye(n_features)  # 确保协方差矩阵可逆
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        diff = (X - mean).T
        exponent = -0.5 * np.sum(np.dot(inv, diff) * diff, axis=0)
        prefactor = 1 / np.sqrt((2 * np.pi) ** n_features * det)
        return prefactor * np.exp(exponent)
    
    def fit(self, X):
        self._initialize_parameters(X)
        prev_likelihood = None
        for i in range(self.max_iter):
            responsibilities = self._expectation_step(X)
            self._maximization_step(X, responsibilities)
            likelihood = np.log(self._compute_likelihood(X)).sum()
            if prev_likelihood is not None and np.abs(likelihood - prev_likelihood) < self.tol:
                break
            prev_likelihood = likelihood
        self.labels_ = np.argmax(responsibilities, axis=1)
    
    def _compute_likelihood(self, X):
        n_samples = X.shape[0]
        likelihoods = np.zeros(n_samples)
        for k in range(self.n_components):
            likelihoods += self.weights_[k] * self._multivariate_gaussian(X, self.means_[k], self.covariances_[k])
        return likelihoods

# 编码非数值型特征
label_encoders = {}
for col in ["protocol_type", "service", "flag","label"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 提取特征和标签
X = df.drop(columns=["label"])
y = df["label"]

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拟合自定义高斯混合模型
gmm = GaussianMixtureModel(n_components=23)
gmm.fit(X_scaled)
labels = gmm.labels_

# 验证聚类结果
ari = adjusted_rand_score(y, labels)
ami = adjusted_mutual_info_score(y, labels)
v_measure = v_measure_score(y, labels)
conf_matrix = confusion_matrix(y, labels)

print(f"Adjusted Rand Index: {ari}")
print(f"Adjusted Mutual Information: {ami}")
print(f"V-measure: {v_measure}")
print(f"Confusion Matrix:\n{conf_matrix}")

# 绘制聚类结果
import matplotlib.pyplot as plt

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()