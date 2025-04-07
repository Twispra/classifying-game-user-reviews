import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models


def main():
    # 1. 加载第三步输出的BERT模型
    #    例如: "./model_ckpt/game_specific_bert/"
    model_path = "./model_ckpt/game_specific_bert"

    # 使用 SentenceTransformer 的方式:
    # 先构造一个 Transformer 模块，再拼接池化模块
    # (max_seq_length可以根据评论长度做适当调大)
    word_embedding_model = models.Transformer(model_path, max_seq_length=128)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True  # 论文中也提到Mean Pooling
    )
    game_bert = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 2. 读取需要向量化的评论文件
    #    示例里从第二步得到的 "preprocessed_reviews.csv" 中加载文本
    input_csv = "data/preprocessed_reviews.csv"
    df = pd.read_csv(input_csv)
    texts = df["review_text"].astype(str).tolist()

    print(f"Loaded {len(texts)} texts to encode...")

    # 3. 开始向量化
    #    batch_size 可结合显存大小调节
    embeddings = game_bert.encode(texts, batch_size=32, show_progress_bar=True)
    # embeddings形状: (num_texts, 768)

    print("Encoding done! Embeddings shape =", embeddings.shape)

    # 4. 保存向量
    #    保存成 pt 格式供下步(聚类)读取
    output_embeddings_pt = "data/review_embeddings.pt"
    torch.save(embeddings, output_embeddings_pt)
    print(f"Embeddings have been saved to {output_embeddings_pt}")


if __name__ == "__main__":
    main()
