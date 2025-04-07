# 游戏评论分类系统

这个项目是一个用于分类游戏用户评论的系统，使用主动学习方法来提高分类效率和准确性。

## 主要功能

- 多标签分类：将游戏评论分为"游戏环境"、"游戏优化"和"游戏性"三个维度
- 主动学习流程：通过查询委员会（Query By Committee）方法选择高价值的样本进行标注
- 标注工具：提供直观的GUI界面进行评论标注
- 模型训练：支持增量训练和最终模型的训练
- 预测：使用训练好的模型对新评论进行分类

## 文件说明

- `06_active_learning.py`: 主要的主动学习流程实现
- `annotation_tool.py`: 评论标注工具
- `app.py`: Web应用界面
- `07_predict.py`: 用于对新评论进行预测
- `03_further_pretrain_bert.py`: BERT模型的进一步预训练
- `04_vectorize_reviews.py`: 将评论转换为向量表示
- `05_clustering_seed.py`: 聚类方法选择初始标注样本

## 安装依赖

```bash
pip install torch numpy pandas scikit-learn matplotlib tkinter sentence-transformers transformers tqdm
```

## 使用说明

1. 数据准备：将评论数据放入`data`目录
2. 向量化：使用`04_vectorize_reviews.py`生成评论的向量表示
3. 主动学习：运行`06_active_learning.py`开始主动学习过程
4. 模型训练：主动学习过程会自动训练模型并保存到`model_ckpt`目录
5. 预测：使用`07_predict.py`对新评论进行预测

## 标签体系

使用3位二进制数表示评论的多个标签:
- 第1位 (左): 游戏环境 - 游戏社区、玩家体验、社交互动等
- 第2位 (中): 游戏优化 - 性能、卡顿、崩溃等技术问题
- 第3位 (右): 游戏性 - 玩法、难度、乐趣、游戏设计等 