import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset


# 指定文件路径
file_path = r"E:\Python pratice\game_reviews\cleaned_reviews.csv"

# 读取CSV文件（假设列名为 "clean_review"）
df = pd.read_csv(file_path)

# 提取清洗后的文本列表
clean_reviews = df["clean_review"].astype(str).tolist()
# 使用清洗后的文本创建数据集
dataset = Dataset.from_dict({"text": clean_reviews})

# 加载预训练的BERT基础模型和分词器（这里使用bert-base-uncased，如需中文请替换为对应模型）
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)


# 定义分词函数
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)


# 对数据集进行分词编码
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 设置MLM数据整理器，自动随机遮蔽15%的token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# 设置训练参数（注意：根据RTX 3070显卡的内存，batch_size可以调整）
training_args = TrainingArguments(
    output_dir="game_bert",  # 模型保存目录
    overwrite_output_dir=True,
    num_train_epochs=5,  # 训练轮数，可根据实际效果调整
    per_device_train_batch_size=8,  # 每个GPU的batch大小
    learning_rate=5e-5,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=500,
    logging_dir="logs",
    report_to=[],
    fp16=True  # 使用混合精度加速训练
)

# 创建Trainer对象并开始训练
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

# 训练完成后保存模型
trainer.save_model("game_domain_bert")
tokenizer.save_pretrained("game_domain_bert")

