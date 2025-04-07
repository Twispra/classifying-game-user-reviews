# gradio_app.py
import gradio as gr
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# 定义类别映射
CLASS_MAPPING = {
    0: "竞技环境问题",
    1: "游戏厂商问题",
    2: "游戏性问题",
    3: "游戏优化问题"
}

# 加载模型
model = joblib.load('model_ckpt/model_final.pkl')
embeddings_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def predict_review(text):
    if not text.strip():
        return None, None, None

    # 生成嵌入向量
    embedding = embeddings_model.encode([text])

    # 预测
    prediction = model.predict(embedding)[0]
    probabilities = model.predict_proba(embedding)[0]

    # 准备可视化数据
    df = pd.DataFrame({
        '类别': list(CLASS_MAPPING.values()),
        '概率': probabilities
    })

    return CLASS_MAPPING[prediction], df, df.plot.barh(x='类别', y='概率', figsize=(8, 4)).figure


def process_file(file):
    if file is None:
        return None

    content = file.decode('utf-8')
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    results = []
    for line in lines:
        # 生成嵌入向量
        embedding = embeddings_model.encode([line])

        # 预测
        prediction = model.predict(embedding)[0]
        probabilities = model.predict_proba(embedding)[0]

        results.append({
            '评论': line,
            '预测类别': CLASS_MAPPING[prediction],
            '竞技环境问题': f"{probabilities[0]:.4f}",
            '游戏厂商问题': f"{probabilities[1]:.4f}",
            '游戏性问题': f"{probabilities[2]:.4f}",
            '游戏优化问题': f"{probabilities[3]:.4f}"
        })

    return pd.DataFrame(results)


# 创建Gradio界面
with gr.Blocks(title="游戏评论分类系统") as demo:
    gr.Markdown("# 游戏评论分类系统")
    gr.Markdown("输入游戏评论，系统将自动分类")

    with gr.Tab("单条评论分析"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="输入评论", lines=5)
                analyze_btn = gr.Button("分析")

            with gr.Column():
                prediction_output = gr.Textbox(label="预测类别")
                prob_table = gr.Dataframe(label="概率表")
                prob_chart = gr.Plot(label="概率图表")

        analyze_btn.click(predict_review, inputs=text_input, outputs=[prediction_output, prob_table, prob_chart])

    with gr.Tab("批量评论分析"):
        file_input = gr.File(label="上传评论文件（每行一条评论）")
        batch_results = gr.Dataframe(label="批量分析结果")

        file_input.change(process_file, inputs=file_input, outputs=batch_results)

# 启动Gradio应用
demo.launch()