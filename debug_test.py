import os
import sys
import json
import time
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
import torch

print("脚本开始运行")
print(f"Python版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")
print(f"命令行参数: {sys.argv}")

# 测试系统编码
print(f"文件系统编码: {sys.getfilesystemencoding()}")
print(f"默认编码: {sys.getdefaultencoding()}")

# 测试文件读写
test_file = "test_output.txt"
try:
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("测试中文写入\n")
        f.write("Test English text\n")
    print(f"成功写入文件: {test_file}")
    
    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"读取文件内容: {content}")
except Exception as e:
    print(f"文件操作失败: {e}")

# 测试数据处理库
try:
    # NumPy测试
    arr = np.array([1, 2, 3, 4, 5])
    print(f"NumPy数组: {arr}, 形状: {arr.shape}, 平均值: {arr.mean()}")
    
    # Pandas测试
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print(f"Pandas DataFrame:\n{df}")
    
    # Matplotlib测试
    plt.figure(figsize=(3, 2))
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.title("测试图表")
    plt.savefig("test_plot.png")
    print("已保存测试图表")
    
    # PyTorch测试
    tensor = torch.tensor([1.0, 2.0, 3.0])
    print(f"PyTorch张量: {tensor}, 设备: {tensor.device}")
    
    # scikit-learn测试
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    clf = LinearSVC(random_state=42)
    clf.fit(X, y)
    print(f"模型系数: {clf.coef_}")
    
except Exception as e:
    print(f"库测试失败: {e}")

print("测试完成") 