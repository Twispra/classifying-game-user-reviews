from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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


# 从active learning中得到的100条标注数据的索引存储在 labeled_indices 中
# 构造最终训练集的特征和标签
final_X = review_vectors[labeled_indices]
final_y = [all_labels[i] for i in labeled_indices]

# 如果有足够数据，可进一步划分训练集和测试集来评估模型性能（这里采用20%的测试集）
X_train, X_test, y_train, y_test = train_test_split(final_X, final_y, test_size=0.2, random_state=42)

# 训练最终分类器
final_clf = LogisticRegression(max_iter=500, solver='lbfgs', multi_class='multinomial')
final_clf.fit(X_train, y_train)

# 评估模型
acc = final_clf.score(X_test, y_test)
print("最终分类器在测试集上的准确率：", acc)
