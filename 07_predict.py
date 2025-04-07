# predict.py
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np


CLASS_MAPPING = {
    0: "竞技环境问题",
    1: "游戏厂商问题",
    2: "游戏性问题",
    3: "游戏优化问题"
}


model_path = 'model_ckpt/model_final.pkl'
try:
    model = joblib.load(model_path)
    print(f"已加载模型: {model_path}")
except Exception as e:
    print(f"加载模型失败: {e}")
    exit(1)


try:
    embeddings_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("已加载嵌入模型")
except Exception as e:
    print(f"加载嵌入模型失败: {e}")
    print("请安装sentence-transformers库: pip install sentence-transformers")
    exit(1)


print("\n=== 游戏评论分类预测工具 ===")
print("输入'退出'结束程序\n")

while True:
    user_input = input("请输入游戏评论: ")
    if user_input.lower() in ['退出', 'exit', 'quit']:
        break

    # 生成嵌入向量
    embedding = embeddings_model.encode([user_input])

    # 预测
    prediction = model.predict(embedding)[0]
    probabilities = model.predict_proba(embedding)[0]

    # 输出结果
    print(f"\n预测类别: {CLASS_MAPPING[prediction]}")
    print("各类别概率:")
    for class_id, prob in enumerate(probabilities):
        print(f"  {CLASS_MAPPING[class_id]}: {prob:.4f}")
    print()