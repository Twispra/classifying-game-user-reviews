import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

# 指定文件路径
file_path = r"E:\Python pratice\game_reviews\cleaned_reviews.csv"
# 读取CSV文件（假设列名为 "clean_review"）
df = pd.read_csv(file_path)
# 提取清洗后的文本列表
clean_reviews = df["clean_review"].astype(str).tolist()



# 指定领域预训练模型的路径（请确保路径正确）
domain_model_path = r"E:\Python pratice\game_reviews\game_domain_bert"
tokenizer = BertTokenizer.from_pretrained(domain_model_path)
bert_model = BertModel.from_pretrained(domain_model_path)
bert_model.eval()


# 如果有GPU，则将模型移动到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# 假设 clean_reviews 是一个包含所有清洗后评论文本的列表
# 例如：clean_reviews = df["clean_review"].astype(str).tolist()

batch_size = 32  # 可以根据显存情况调整batch_size大小
all_embeddings = []

for i in range(0, len(clean_reviews), batch_size):
    batch_texts = clean_reviews[i: i + batch_size]
    encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

    # 将编码数据移动到GPU（如果可用）
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = bert_model(**encodings)
    token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
    attention_mask = encodings["attention_mask"]

    # 均值池化，仅对非填充token取平均
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    all_embeddings.append(mean_embeddings.cpu())

# 拼接所有批次得到最终的向量矩阵
all_embeddings = torch.cat(all_embeddings, dim=0)
review_vectors = all_embeddings.numpy()
print("嵌入向量形状：", review_vectors.shape)


# 将向量存储起来
import numpy as np
np.save("review_vectors.npy", review_vectors)



