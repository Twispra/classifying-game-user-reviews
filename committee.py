import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.linear_model import LogisticRegression

# ---------------------------
# 1. 数据加载部分
# ---------------------------
# 加载清洗后的评论文本
df = pd.read_csv(r"E:\Python pratice\game_reviews\cleaned_reviews.csv")
reviews = df["clean_review"].astype(str).tolist()

# 加载预先生成的评论向量（假设已保存为 .npy 文件）
review_vectors = np.load(r"E:\Python pratice\game_reviews\review_vectors.npy")

# 加载已选取的种子索引（用于初始训练集），假设存放在 seed.csv 中，一列名为 "seed_index"
initial_seed_indices = pd.read_csv(r"E:\Python pratice\game_reviews\seed.csv")["seed_index"].tolist()

# ---------------------------
# 2. 主动学习中委员会部分（示例代码）
# ---------------------------

# 假设我们有一个列表 all_labels 存储所有评论的真实标签（实际情况中人工标注后才有）
# 如果没有真实标签，可以用下面这一行模拟生成随机标签（此处仅为示例）
possible_labels = ["Production", "Operations", "Design"]
all_labels = [np.random.choice(possible_labels) for _ in range(len(reviews))]

# 初始化已标注的训练集索引和未标注池索引
labeled_indices = initial_seed_indices.copy()  # 初始种子作为已标注样本
unlabeled_indices = [i for i in range(len(reviews)) if i not in labeled_indices]

# 构造训练集特征和标签
train_X = review_vectors[labeled_indices]
train_y = [all_labels[i] for i in labeled_indices]

# 主动学习的参数
committee_size = 10
max_queries = 90  # 查询次数上限
queries_made = 0

# 主动学习循环（这里只展示一次查询的流程，可在循环中多次执行）
while queries_made < max_queries and unlabeled_indices:
    # 1. 用bootstrap方法训练委员会模型
    committee_models = []
    for m in range(committee_size):
        # 自举采样：从当前训练集有放回随机采样
        sample_indices = np.random.choice(range(len(train_X)), size=len(train_X), replace=True)
        X_sample = train_X[sample_indices]
        y_sample = [train_y[idx] for idx in sample_indices]
        # 显式设置 multi_class='multinomial' 避免警告
        model = LogisticRegression(max_iter=500, solver='lbfgs')
        model.fit(X_sample, y_sample)
        committee_models.append(model)

    # 2. 在未标注集合中找出投票熵最大的样本
    best_normalized_entropy = -1
    best_index = None
    for idx in unlabeled_indices:
        vec = review_vectors[idx].reshape(1, -1)
        votes = [model.predict(vec)[0] for model in committee_models]
        vote_counts = Counter(votes)

        # 计算投票熵（未归一化）
        entropy = 0.0
        for count in vote_counts.values():
            p = count / committee_size
            if p > 0:
                entropy -= p * math.log2(p)

        # 如果vote_counts长度为1，直接归一化熵为0，否则归一化
        if len(vote_counts) > 1:
            norm_factor = math.log2(len(vote_counts))
            normalized_entropy = entropy / norm_factor
        else:
            normalized_entropy = 0.0

        if normalized_entropy > best_normalized_entropy:
            best_normalized_entropy = normalized_entropy
            best_index = idx

    # 如果找不到合适的样本则退出
    if best_index is None:
        break

    # 3. 模拟人工标注：取出该样本的真实标签
    query_idx = best_index
    queries_made += 1
    unlabeled_indices.remove(query_idx)
    queried_label = all_labels[query_idx]
    print(f"查询样本索引: {query_idx}, 投票熵: {best_normalized_entropy:.3f}, 标签: {queried_label}")

    # 4. 将新标注样本加入训练集
    labeled_indices.append(query_idx)
    train_X = np.vstack([train_X, review_vectors[query_idx]])
    train_y.append(queried_label)

    # 可选：在每轮查询后，可评估模型性能，这里省略


# 输出最终训练集大小
print(f"主动学习结束，总共标注样本数: {len(labeled_indices)}")


import pandas as pd

# 保存评论文本和对应标签（例如存为 CSV）
labeled_data = pd.DataFrame({
    "review": [reviews[i] for i in labeled_indices],
    "label": [all_labels[i] for i in labeled_indices]
})
labeled_data.to_csv(r"E:\Python pratice\game_reviews\labeled_data.csv", index=False)

# 如果你需要保存向量，也可以保存为 .npy 文件
labeled_vectors = review_vectors[labeled_indices]
np.save(r"E:\Python pratice\game_reviews\labeled_vectors.npy", labeled_vectors)
