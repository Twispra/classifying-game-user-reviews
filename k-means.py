from sklearn.cluster import KMeans
import numpy as np

#导入数据
import pandas as pd
reviews = pd.read_csv(r"E:\Python pratice\game_reviews\cleaned_reviews.csv")
reviews = reviews["clean_review"].astype(str).tolist()
review_vectors = np.load("review_vectors.npy")


# 1. 设定K值并运行KMeans聚类
K = 5  # 这里假设5个聚类，可根据肘部法或需要调整
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(review_vectors)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 2. 对于每个簇，找出距离中心最近的N个点 (例如N=60)
N = 60
reviews_array = np.array(reviews)  # 将原始评论文本转为数组，以便通过索引获取
seed_base_indices = []  # 用于存储所有簇的代表实例索引
for cluster_id in range(K):
    # 计算该簇中心与所有点的距离（这里用欧氏距离平方，即惯性距离）
    # kmeans.fit后可以直接使用 transform 得到每个点到每个中心的距离
    distances = kmeans.transform(review_vectors)[:, cluster_id]
    # 获取属于当前簇的所有点索引
    cluster_points = np.where(labels == cluster_id)[0]
    # 在该簇的点中，按距离排序选最小的N个
    cluster_distances = distances[cluster_points]
    nearest_idx_within_cluster = cluster_points[np.argsort(cluster_distances)[:N]]
    seed_base_indices.extend(nearest_idx_within_cluster)

seed_base_indices = list(set(seed_base_indices))  # 去重（不同簇最近的点可能重复，但一般不会）
print(f"每个簇各选{N}个代表，共{len(seed_base_indices)}条评论作为Seed Base Set")

# 3. 从Seed Base Set中随机选取初始标注种子
initial_seed_count = 10  # 初始标注实例数
np.random.seed(42)
initial_seed_indices = np.random.choice(seed_base_indices, size=initial_seed_count, replace=False)
initial_seed_texts = reviews_array[initial_seed_indices]
print("初始标注种子实例索引：", initial_seed_indices)

# 4. 将初始标注种子写入CSV文件
seed_df = pd.DataFrame({"seed_index": initial_seed_indices, "seed_text": initial_seed_texts})
seed_df.to_csv("seed.csv", index=False)



