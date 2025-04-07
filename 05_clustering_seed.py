import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os
from sklearn.metrics import silhouette_score


def main():
    # 1. 确保数据目录存在
    os.makedirs("data", exist_ok=True)

    # 设置文件路径
    embeddings_pt = "data/review_embeddings.pt"
    preprocessed_csv = "data/preprocessed_reviews.csv"

    # 检查文件是否存在
    if not os.path.exists(embeddings_pt):
        raise FileNotFoundError(f"找不到嵌入向量文件: {embeddings_pt}")
    if not os.path.exists(preprocessed_csv):
        raise FileNotFoundError(f"找不到预处理评论文件: {preprocessed_csv}")

    # 解决 PyTorch 2.6 加载问题 - 方法1: 使用 weights_only=False
    try:
        embeddings = torch.load(embeddings_pt, weights_only=False)
    except Exception as e:
        print(f"尝试方法1失败: {e}")
        # 方法2: 添加安全全局变量
        from torch.serialization import add_safe_globals
        import numpy._core.multiarray as ma
        add_safe_globals([ma._reconstruct])
        embeddings = torch.load(embeddings_pt)

    # 加载对应文本
    df = pd.read_csv(preprocessed_csv)
    # 确保 review_text 列存在
    if "review_text" not in df.columns:
        possible_columns = df.columns.tolist()
        raise ValueError(f"在CSV中找不到 'review_text' 列。可用列: {possible_columns}")

    texts = df["review_text"].tolist()

    if embeddings.shape[0] != len(texts):
        raise ValueError(f"向量数({embeddings.shape[0]})和文本数({len(texts)})不一致，请检查数据！")

    print(f"成功加载嵌入向量，形状={embeddings.shape}，共{len(texts)}条文本。")

    # 2. 使用肘部法则自动选择最佳K值
    max_k = min(10, embeddings.shape[0] // 10)  # 最大聚类数，不超过样本总数的10%
    inertias = []
    silhouette_scores = []

    # 最少2个簇，最多max_k个簇
    for k in range(2, max_k + 1):
        print(f"正在尝试 k={k} 的聚类...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

        # 计算轮廓系数
        if k > 1:  # 轮廓系数需要至少2个簇
            sil_score = silhouette_score(embeddings, kmeans.labels_)
            silhouette_scores.append(sil_score)
            print(f"k={k} 的轮廓系数: {sil_score:.4f}")

    # 使用肘部法则选择k值
    # 计算拐点（简化实现）
    deltas = np.diff(inertias)
    acceleration = np.diff(deltas)
    k_idx = np.argmax(acceleration) + 2  # +2是因为我们从k=2开始，并且diff操作会减少一个元素
    optimal_k = min(k_idx + 2, max_k)  # 经验上拐点通常偏小，适当增加一些k

    # 也考虑轮廓系数
    best_sil_k = np.argmax(silhouette_scores) + 2

    print(f"肘部法则建议的k值: {k_idx + 2}")
    print(f"轮廓系数最大的k值: {best_sil_k}")

    # 综合考虑两种方法
    k = optimal_k
    print(f"选择的最终k值: {k}")

    # 3. 使用选定的k值进行K-means聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # 4. 计算每条评论到所属簇中心的距离
    distances = kmeans.transform(embeddings)  # [N, k]

    # 5. 对每个簇，选出离中心最近的topN条评论
    # 自适应确定topN - 根据总样本量和簇数调整
    total_samples = len(texts)
    seeds_per_cluster = max(10, min(60, total_samples // (k * 5)))  # 每个簇至少10条，最多60条

    print(f"每个簇将选择 {seeds_per_cluster} 条最接近中心的评论")

    seed_rows = []  # 用来存放 (cluster_id, review_text, 原始索引, 到中心距离)

    for cluster_id in range(k):
        # 找出属于该cluster_id的样本索引
        cluster_member_idx = np.where(labels == cluster_id)[0]

        # 计算这些样本到对应簇中心的距离
        member_dist = distances[cluster_member_idx, cluster_id]
        sorted_indices = np.argsort(member_dist)  # 按距离升序

        # 确保不超过该簇的样本数
        top_count = min(seeds_per_cluster, len(sorted_indices))
        selected_indices = cluster_member_idx[sorted_indices[:top_count]]

        # 收集信息: (簇ID, 评论文本, 原始索引, 到中心距离)
        for idx in selected_indices:
            seed_rows.append((
                cluster_id,
                texts[idx],
                idx,  # 原始索引，方便后续追踪
                distances[idx, cluster_id]  # 到中心的距离
            ))

    # 6. 将这些中心邻近评论输出到seeds_base.csv
    output_csv = "data/seeds_base.csv"
    df_seed = pd.DataFrame(seed_rows, columns=["cluster_id", "review_text", "original_index", "distance_to_center"])

    # 按簇ID和距离排序，方便查看
    df_seed.sort_values(["cluster_id", "distance_to_center"], inplace=True)

    # 保存结果
    df_seed.to_csv(output_csv, index=False, encoding='utf-8')

    # 7. 为主动学习准备初始种子集
    # 从每个簇选择更多样本，确保每个类别有足够的样本进行训练
    seeds_per_cluster_init = 30  # 每个簇选5条，总共k*5条作为初始标注
    initial_seeds = []

    for cluster_id in range(k):
        cluster_rows = df_seed[df_seed["cluster_id"] == cluster_id]
        initial_seeds.extend(cluster_rows.iloc[:seeds_per_cluster_init].to_dict('records'))

    initial_seeds_df = pd.DataFrame(initial_seeds)
    initial_seeds_csv = "data/initial_seeds.csv"
    initial_seeds_df.to_csv(initial_seeds_csv, index=False, encoding='utf-8')

    print(f"聚类完成。种子库已保存到 {output_csv}")
    print(f"每个簇中接近中心的 {seeds_per_cluster} 条评论已包含在内。")
    print(f"建议的初始标注集（每簇{seeds_per_cluster_init}条，共{len(initial_seeds)}条）已保存到 {initial_seeds_csv}")
    print("请在Excel中打开并添加'label'列进行人工标注，然后用于下一步的主动学习。")
    print("请为每个样本标注以下4个类别之一:")
    print("0: 竞技环境问题")
    print("1: 游戏厂商问题")
    print("2: 游戏性问题")
    print("3: 游戏优化问题")
    print("请确保每个类别至少有2个样本，以便进行模型训练。")


if __name__ == "__main__":
    main()