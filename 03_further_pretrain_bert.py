import os
import pandas as pd
import torch

from transformers import BertTokenizer, BertForMaskedLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

def read_texts_for_mlm(csv_file):
    """
    从预处理好的CSV文件中读取评论文本，用于Masked Language Modeling。
    假设 CSV 有一列 "review_text"。
    """
    df = pd.read_csv(csv_file)
    texts = df["review_text"].astype(str).tolist()
    return texts

class TextDatasetForMLM(torch.utils.data.Dataset):
    """
    自定义Dataset，将文本转换成BERT输入格式(input_ids, attention_mask等)，
    用于后续Trainer训练。
    """
    def __init__(self, texts, tokenizer, max_length=128):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length
        )
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }

if __name__ == "__main__":
    # 1. 配置
    base_model_name = "bert-base-uncased"   # 初始BERT
    input_csv = "data/preprocessed_reviews.csv"  # 第2步输出的预处理数据文件
    output_dir = "./model_ckpt/game_specific_bert"  # 本步输出的模型保存目录

    # 2. 读取文本
    mlm_texts = read_texts_for_mlm(input_csv)
    print(f"Loaded {len(mlm_texts)} texts for MLM pretraining.")

    # 3. 初始化 tokenizer 和 BertForMaskedLM
    tokenizer = BertTokenizer.from_pretrained(base_model_name)
    model = BertForMaskedLM.from_pretrained(base_model_name)

    # 4. 构建Dataset
    dataset = TextDatasetForMLM(mlm_texts, tokenizer, max_length=128)

    # 5. 数据收集器: 负责在batch里随机mask一定比例的token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # 这里常用15%
    )

    # 6. 训练参数
    #    可视情况增大 epoch/batch_size；若数据量很大，可以先测试小规模
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,              # 训练轮数
        per_device_train_batch_size=8,   # 每个设备的batch_size
        save_steps=1000,                # 每多少步保存一次
        save_total_limit=1,             # 保留多少个checkpoint
        logging_steps=500,
        logging_dir=os.path.join(output_dir, "logs"),
        do_train=True
    )

    # 7. 构建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    # 8. 训练
    trainer.train()

    # 9. 保存最终模型与tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Further pretraining done! Model saved to: {output_dir}")
