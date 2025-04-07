#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
游戏评论多标签分类 - 主动学习版本（简化版）

此文件实现了使用主动学习进行游戏评论多标签分类的核心功能。
包括训练多标签分类器、使用委员会进行查询选择、交互式用户标注等。
"""

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

# 导入标注工具
from annotation_tool import AnnotationInterface

# 设置随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 定义常量
DATA_DIR = Path("data")
EMBEDDING_FILE = DATA_DIR / "review_embeddings.pt"
PREPROCESSED_FILE = DATA_DIR / "preprocessed_reviews.csv"
ANNOTATION_FILE = DATA_DIR / "annotations.json"
MAX_SAMPLES_PER_ROUND = 10
MODEL_DIR = Path("models")
PLOTS_DIR = Path("plots")

# 确保必要的目录存在
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# 全局变量设置
NUM_CLASSES = 3

def load_preprocessed_data(file_path=PREPROCESSED_FILE):
    """加载预处理后的评论数据"""
    try:
        print(f"正在加载预处理数据: {file_path}")
        df = pd.read_csv(file_path)
        
        # 确保数据包含必要的列
        if "original_index" not in df.columns:
            print("警告: 数据中缺少'original_index'列，添加索引列")
            df["original_index"] = df.index
        
        print(f"成功加载 {len(df)} 条预处理评论数据")
        return df
    except Exception as e:
        print(f"加载预处理数据失败: {e}")
        sys.exit(1)

def load_embeddings(file_path=EMBEDDING_FILE):
    """加载嵌入向量"""
    print(f"正在加载嵌入向量: {file_path}")
    
    try:
        if not os.path.exists(file_path):
            print(f"错误: 找不到嵌入向量文件 {file_path}")
            sys.exit(1)
            
        # 获取文件大小信息
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"嵌入向量文件大小: {file_size_mb:.2f} MB")
        
        # 尝试加载嵌入向量
        try:
            # 方法1: 直接加载
            embeddings = torch.load(file_path)
            print("使用torch.load()成功加载嵌入向量")
        except Exception as e1:
            print(f"直接加载失败: {e1}")
            try:
                # 方法2: 使用weights_only参数
                embeddings = torch.load(file_path, weights_only=True)
                print("使用weights_only=True成功加载嵌入向量")
            except Exception as e2:
                print(f"weights_only加载失败: {e2}")
                try:
                    # 方法3: 使用map_location参数
                    embeddings = torch.load(file_path, map_location=torch.device('cpu'))
                    print("使用map_location=cpu成功加载嵌入向量")
                except Exception as e3:
                    print(f"map_location加载失败: {e3}")
                    try:
                        # 方法4: 使用pickle
                        with open(file_path, 'rb') as f:
                            embeddings = pickle.load(f)
                        print("使用pickle成功加载嵌入向量")
                    except Exception as e4:
                        print(f"pickle加载失败: {e4}")
                        print("所有加载方法都失败，无法加载嵌入向量")
                        sys.exit(1)
        
        # 转换为NumPy数组以便处理
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
        
        # 简单检查嵌入向量的形状和统计信息
        print(f"嵌入向量形状: {embeddings.shape}")
        print(f"嵌入向量类型: {type(embeddings)}")
        print(f"嵌入向量统计: min={embeddings.min():.4f}, max={embeddings.max():.4f}, mean={embeddings.mean():.4f}")
        
        return embeddings
    except Exception as e:
        print(f"加载嵌入向量失败: {e}")
        sys.exit(1)

def load_annotations(file_path=ANNOTATION_FILE):
    """加载已有标注"""
    print(f"正在加载标注数据: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"警告: 标注文件不存在 {file_path}，将创建新文件")
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            annotations = {int(k): int(v) for k, v in data.get('annotations', {}).items()}
        print(f"成功加载 {len(annotations)} 条标注")
        return annotations
    except Exception as e:
        print(f"加载标注数据失败: {e}，将使用空标注集")
        return {}

def save_annotations(annotations, file_path=ANNOTATION_FILE, is_complete=False):
    """保存标注到文件"""
    try:
        data = {
            "annotations": {str(k): v for k, v in annotations.items()},
            "timestamp": datetime.now().isoformat(),
            "completed": is_complete
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(annotations)} 条标注到 {file_path}")
        return True
    except Exception as e:
        print(f"保存标注失败: {e}")
        return False

def annotations_to_multilabel(annotations, num_classes=NUM_CLASSES):
    """将标注转换为多标签格式"""
    labels = []
    for index, label in annotations.items():
        binary = bin(label)[2:].zfill(num_classes)
        binary = binary[-num_classes:]  # 确保长度正确
        label_list = [i for i, bit in enumerate(binary) if bit == '1']
        labels.append((index, label_list))
    return labels

class QueryByCommittee:
    """使用委员会查询策略选择样本"""
    def __init__(self, embeddings, committee_size=3, batch_size=100):
        self.embeddings = embeddings
        self.committee_size = committee_size
        self.batch_size = batch_size
        self.committee = []
        self.best_model = None
        
    def initialize(self, X_labeled, y_labeled):
        """初始化委员会"""
        print(f"初始化委员会 (大小={self.committee_size})")
        
        # 重置委员会
        self.committee = []
        
        # 创建并训练委员会成员
        for i in range(self.committee_size):
            # 使用不同的随机种子创建模型
            seed = RANDOM_SEED + i
            np.random.seed(seed)
            
            # 创建模型
            model = OneVsRestClassifier(LinearSVC(random_state=seed))
            
            # 如果有标注数据，训练模型
            if X_labeled is not None and len(X_labeled) > 0:
                model.fit(X_labeled, y_labeled)
            
            # 添加到委员会
            self.committee.append(model)
        
        print(f"委员会初始化完成，共有 {len(self.committee)} 个成员")
        return self.committee
    
    def calculate_vote_entropy(self, predictions):
        """计算预测结果的投票熵"""
        try:
            # 确保predictions是3D数组 [committee_members, samples, classes]
            if len(predictions.shape) != 3:
                print(f"警告: 预测结果维度不正确: {predictions.shape}，期望3D数组")
                return np.zeros(predictions.shape[1])
            
            vote_entropy = np.zeros(predictions.shape[1])
            num_members = predictions.shape[0]
            
            # 对每个样本计算每个类的熵
            for i in range(predictions.shape[1]):  # 样本
                class_entropy = np.zeros(predictions.shape[2])  # 每个类的熵
                
                for j in range(predictions.shape[2]):  # 类别
                    # 计算类别j的投票分布
                    votes = predictions[:, i, j]
                    pos_votes = np.sum(votes == 1)
                    neg_votes = num_members - pos_votes
                    
                    # 计算熵 (使用log2)
                    if pos_votes == 0 or neg_votes == 0:
                        # 所有委员会成员一致，熵为0
                        entropy = 0.0
                    else:
                        # 计算熵: -p_pos*log2(p_pos) - p_neg*log2(p_neg)
                        p_pos = pos_votes / num_members
                        p_neg = neg_votes / num_members
                        entropy = -(p_pos * np.log2(p_pos) + p_neg * np.log2(p_neg))
                    
                    class_entropy[j] = entropy
                
                # 样本的总熵是所有类别熵的平均
                vote_entropy[i] = np.mean(class_entropy)
            
            return vote_entropy
        except Exception as e:
            print(f"计算投票熵时出错: {e}")
            return np.zeros(predictions.shape[1])
    
    def select_samples(self, labeled_indices, unlabeled_indices, n_samples=MAX_SAMPLES_PER_ROUND):
        """选择信息量大的样本供标注"""
        if not self.committee:
            print("错误: 委员会未初始化")
            return []
        
        if not unlabeled_indices or len(unlabeled_indices) == 0:
            print("警告: 没有未标注样本可供选择")
            return []
        
        try:
            print(f"从 {len(unlabeled_indices)} 个未标注样本中选择 {n_samples} 个样本")
            
            # 记录开始时间
            start_time = time.time()
            
            # 将unlabeled_indices转换为有效的数组
            valid_unlabeled_indices = np.array([idx for idx in unlabeled_indices 
                                                if idx >= 0 and idx < len(self.embeddings)])
            
            if len(valid_unlabeled_indices) == 0:
                print("警告: 没有有效的未标注样本索引")
                return []
            
            # 获取未标注样本的嵌入向量
            unlabeled_embeddings = self.embeddings[valid_unlabeled_indices]
            
            # 分批处理，避免内存溢出
            all_predictions = []
            batch_count = (len(valid_unlabeled_indices) + self.batch_size - 1) // self.batch_size
            
            for i in range(batch_count):
                batch_start = i * self.batch_size
                batch_end = min((i + 1) * self.batch_size, len(valid_unlabeled_indices))
                batch_indices = valid_unlabeled_indices[batch_start:batch_end]
                batch_embeddings = self.embeddings[batch_indices]
                
                # 收集每个委员会成员的预测
                batch_predictions = []
                for model in self.committee:
                    try:
                        # 预测多标签结果
                        pred = model.predict(batch_embeddings)
                        batch_predictions.append(pred)
                    except Exception as e:
                        print(f"委员会成员预测出错: {e}")
                
                # 将批次预测添加到总预测中
                if batch_predictions:
                    # 转换为numpy数组 [committee_members, samples, classes]
                    batch_predictions = np.array(batch_predictions)
                    all_predictions.append((batch_indices, batch_predictions))
            
            # 计算熵并选择样本
            entropy_scores = []
            
            for batch_indices, batch_predictions in all_predictions:
                # 计算每个样本的投票熵
                batch_entropy = self.calculate_vote_entropy(batch_predictions)
                
                # 将索引和熵打包
                for idx, entropy in zip(batch_indices, batch_entropy):
                    entropy_scores.append((idx, entropy))
            
            # 如果没有有效的熵分数，返回空列表
            if not entropy_scores:
                print("警告: 没有计算出有效的熵分数")
                return []
            
            # 按熵降序排序
            entropy_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 选择熵最高的样本
            selected_indices = [idx for idx, _ in entropy_scores[:n_samples]]
            
            # 计算用时
            elapsed_time = time.time() - start_time
            print(f"样本选择完成，用时 {elapsed_time:.2f} 秒")
            print(f"已选择 {len(selected_indices)} 个样本")
            
            return selected_indices
        except Exception as e:
            print(f"选择样本时发生错误: {e}")
            return []
    
    def update_models(self, X_labeled, y_labeled):
        """更新委员会成员模型"""
        if not self.committee:
            print("警告: 委员会未初始化，将先初始化")
            self.initialize(X_labeled, y_labeled)
            return
        
        print("更新委员会模型...")
        
        # 更新每个委员会成员
        for i, model in enumerate(self.committee):
            try:
                model.fit(X_labeled, y_labeled)
                print(f"成功更新委员会成员 {i+1}/{len(self.committee)}")
            except Exception as e:
                print(f"更新委员会成员 {i+1} 失败: {e}")
        
        # 将第一个模型作为"最佳模型"
        self.best_model = self.committee[0]
        
        print("委员会模型更新完成")
    
    def predict(self, X):
        """使用最佳模型预测样本标签"""
        if self.best_model is None:
            print("错误: 没有可用的模型进行预测")
            return None
        
        try:
            return self.best_model.predict(X)
        except Exception as e:
            print(f"预测出错: {e}")
            return None

def train_evaluate_model(X_train, y_train, X_test=None, y_test=None, random_state=RANDOM_SEED):
    """训练并评估模型"""
    model = OneVsRestClassifier(LinearSVC(random_state=random_state))
    
    try:
        # 训练模型
        model.fit(X_train, y_train)
        
        # 如果有测试集，计算测试性能
        if X_test is not None and y_test is not None:
            print("评估模型性能...")
            y_pred = model.predict(X_test)
            
            # 计算并打印分类报告
            print("\n分类报告:")
            print(classification_report(y_test, y_pred, zero_division=0))
            
            # 计算宏平均 F1 分数
            macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            print(f"宏平均 F1 分数: {macro_f1:.4f}")
            
            return model, macro_f1
    except Exception as e:
        print(f"训练/评估模型时出错: {e}")
    
    return model, 0.0

def save_model(model, file_path):
    """保存模型到文件"""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"成功保存模型到 {file_path}")
        return True
    except Exception as e:
        print(f"保存模型出错: {e}")
        return False

def load_model(file_path):
    """从文件加载模型"""
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"成功加载模型 {file_path}")
        return model
    except Exception as e:
        print(f"加载模型出错: {e}")
        return None

def plot_learning_curve(metrics, save_path):
    """绘制学习曲线"""
    try:
        iterations = list(range(len(metrics)))
        f1_scores = [m['f1_score'] for m in metrics]
        sample_counts = [m['sample_count'] for m in metrics]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # 绘制F1分数曲线
        ax1.plot(iterations, f1_scores, 'o-', color='blue')
        ax1.set_ylabel('宏平均F1分数')
        ax1.set_title('主动学习F1分数变化曲线')
        ax1.grid(True)
        
        # 绘制样本数量曲线
        ax2.plot(iterations, sample_counts, 's-', color='green')
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('已标注样本数')
        ax2.set_title('已标注样本数量变化曲线')
        ax2.grid(True)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"已保存学习曲线图表到 {save_path}")
        
        plt.close(fig)
    except Exception as e:
        print(f"绘制学习曲线出错: {e}")

def prepare_labeled_unlabeled_data(reviews_df, annotations, embeddings):
    """准备已标注和未标注数据"""
    print("准备标注/未标注数据集...")
    
    try:
        # 创建空的已标注/未标注数据结构
        labeled_indices = []
        unlabeled_indices = []
        
        # 创建索引到嵌入的映射
        index_mapping = {}
        
        # 检查每个评论是否已标注
        for i, row in reviews_df.iterrows():
            try:
                original_idx = int(row['original_index'])
                
                # 检查是否有标注
                if original_idx in annotations:
                    labeled_indices.append(i)
                else:
                    unlabeled_indices.append(i)
                
                # 添加到索引映射
                index_mapping[original_idx] = i
                
            except Exception as e:
                print(f"处理索引 {i} 时出错: {e}")
        
        print(f"已标注样本: {len(labeled_indices)}")
        print(f"未标注样本: {len(unlabeled_indices)}")
        
        # 准备多标签标注数据
        multi_labels = annotations_to_multilabel(annotations)
        
        # 创建标签二值化器
        mlb = MultiLabelBinarizer(classes=list(range(NUM_CLASSES)))
        mlb.fit([list(range(NUM_CLASSES))])
        
        # 准备已标注数据
        X_labeled = None
        y_labeled = None
        
        if labeled_indices:
            # 获取已标注样本的嵌入向量
            X_labeled = embeddings[labeled_indices]
            
            # 转换标签
            y_data = []
            for i in labeled_indices:
                original_idx = int(reviews_df.iloc[i]['original_index'])
                label_value = annotations.get(original_idx, 0)
                
                # 将整数标签转换为多标签格式
                binary = bin(label_value)[2:].zfill(NUM_CLASSES)
                binary = binary[-NUM_CLASSES:]  # 确保长度正确
                label_list = [j for j, bit in enumerate(binary) if bit == '1']
                y_data.append(label_list)
            
            # 二值化标签
            y_labeled = mlb.transform(y_data)
            
            print(f"已准备 X_labeled 形状: {X_labeled.shape}")
            print(f"已准备 y_labeled 形状: {y_labeled.shape}")
        
        return {
            'labeled_indices': labeled_indices,
            'unlabeled_indices': unlabeled_indices,
            'X_labeled': X_labeled,
            'y_labeled': y_labeled,
            'mlb': mlb,
            'index_mapping': index_mapping
        }
    except Exception as e:
        print(f"准备数据集时出错: {e}")
        return None

def main():
    """主函数 - 实现主动学习工作流程"""
    print("=" * 50)
    print("游戏评论多标签分类 - 主动学习版本（简化版）")
    print("=" * 50)
    
    # 1. 加载数据和嵌入向量
    reviews_df = load_preprocessed_data()
    embeddings = load_embeddings()
    annotations = load_annotations()
    
    # 2. 准备标注和未标注数据
    data = prepare_labeled_unlabeled_data(reviews_df, annotations, embeddings)
    
    if not data:
        print("准备数据失败，程序退出")
        sys.exit(1)
    
    labeled_indices = data['labeled_indices']
    unlabeled_indices = data['unlabeled_indices']
    X_labeled = data['X_labeled']
    y_labeled = data['y_labeled']
    mlb = data['mlb']
    index_mapping = data['index_mapping']
    
    # 3. 创建委员会查询策略
    query_strategy = QueryByCommittee(embeddings)
    
    # 4. 如果有已标注数据，初始化委员会
    if X_labeled is not None and len(X_labeled) > 0:
        print("使用已有标注数据初始化委员会")
        query_strategy.initialize(X_labeled, y_labeled)
    else:
        print("没有已标注数据，将在第一轮标注后初始化委员会")
    
    # 5. 主动学习循环
    max_iterations = 10
    metrics = []
    
    for iteration in range(max_iterations):
        print("\n" + "=" * 40)
        print(f"主动学习迭代 {iteration+1}/{max_iterations}")
        print("=" * 40)
        
        # 准备数据和模型
        data = prepare_labeled_unlabeled_data(reviews_df, annotations, embeddings)
        
        if not data:
            print("准备数据失败，跳过当前迭代")
            continue
        
        labeled_indices = data['labeled_indices']
        unlabeled_indices = data['unlabeled_indices']
        X_labeled = data['X_labeled']
        y_labeled = data['y_labeled']
        
        # 如果没有未标注数据，结束循环
        if not unlabeled_indices:
            print("所有样本已标注完毕，结束主动学习")
            break
        
        # 如果没有已标注数据，随机选择一些样本开始
        if not labeled_indices:
            print("没有已标注数据，随机选择第一批样本")
            selected_indices = random.sample(unlabeled_indices, min(MAX_SAMPLES_PER_ROUND, len(unlabeled_indices)))
        else:
            # 初始化/更新委员会模型
            print("更新委员会模型")
            query_strategy.update_models(X_labeled, y_labeled)
            
            # 使用委员会选择下一批样本
            print("使用委员会选择样本")
            selected_indices = query_strategy.select_samples(labeled_indices, unlabeled_indices)
        
        # 没有选择到样本，可能是因为没有未标注数据
        if not selected_indices:
            print("未能选择样本，跳过当前迭代")
            continue
        
        # 获取选择的样本对应的原始索引
        priority_indices = selected_indices
        
        # 开始用户标注
        print(f"请标注 {len(selected_indices)} 个样本")
        
        # 创建并运行标注界面
        annotation_interface = AnnotationInterface(
            reviews_df=reviews_df,
            existing_annotations=annotations,
            priority_indices=priority_indices
        )
        
        # 运行标注界面
        new_annotations = annotation_interface.run()
        
        # 更新标注
        annotations = new_annotations
        
        # 保存标注
        save_annotations(annotations)
        
        # 如果没有新标注，可能用户放弃了标注，跳过本次迭代
        if len(annotations) <= len(labeled_indices):
            print("未添加新标注，跳过本次迭代")
            continue
        
        # 准备评估数据
        print("准备评估数据")
        test_size = 0.3
        # 随机选择标注数据的一部分作为测试集
        random.seed(RANDOM_SEED + iteration)
        test_indices = random.sample(labeled_indices, int(len(labeled_indices) * test_size))
        train_indices = [idx for idx in labeled_indices if idx not in test_indices]
        
        # 获取训练集和测试集
        X_train = embeddings[train_indices]
        y_train_data = []
        for i in train_indices:
            original_idx = int(reviews_df.iloc[i]['original_index'])
            label_value = annotations.get(original_idx, 0)
            # 转换为多标签格式
            binary = bin(label_value)[2:].zfill(NUM_CLASSES)
            binary = binary[-NUM_CLASSES:]
            label_list = [j for j, bit in enumerate(binary) if bit == '1']
            y_train_data.append(label_list)
        y_train = mlb.transform(y_train_data)
        
        X_test = embeddings[test_indices]
        y_test_data = []
        for i in test_indices:
            original_idx = int(reviews_df.iloc[i]['original_index'])
            label_value = annotations.get(original_idx, 0)
            # 转换为多标签格式
            binary = bin(label_value)[2:].zfill(NUM_CLASSES)
            binary = binary[-NUM_CLASSES:]
            label_list = [j for j, bit in enumerate(binary) if bit == '1']
            y_test_data.append(label_list)
        y_test = mlb.transform(y_test_data)
        
        # 训练并评估模型
        print("训练并评估模型")
        model, f1_score = train_evaluate_model(X_train, y_train, X_test, y_test)
        
        # 保存模型
        model_path = MODEL_DIR / f"model_iter_{iteration}.pkl"
        save_model(model, model_path)
        
        # 记录指标
        metrics.append({
            'iteration': iteration,
            'sample_count': len(labeled_indices),
            'f1_score': f1_score
        })
        
        # 绘制学习曲线
        plot_path = PLOTS_DIR / "learning_curve.png"
        plot_learning_curve(metrics, plot_path)
    
    # 6. 训练最终模型
    print("\n" + "=" * 40)
    print("训练最终模型")
    print("=" * 40)
    
    # 准备最终数据
    data = prepare_labeled_unlabeled_data(reviews_df, annotations, embeddings)
    
    if data and data['X_labeled'] is not None:
        X_labeled = data['X_labeled']
        y_labeled = data['y_labeled']
        
        # 训练最终模型
        final_model, _ = train_evaluate_model(X_labeled, y_labeled)
        
        # 保存最终模型
        final_model_path = MODEL_DIR / "final_model.pkl"
        save_model(final_model, final_model_path)
        
        print(f"最终模型已保存到 {final_model_path}")
    else:
        print("没有足够的标注数据训练最终模型")
    
    print("\n主动学习过程完成！")
    
    # 7. 找出表现最好的模型
    if metrics:
        best_iter = max(range(len(metrics)), key=lambda i: metrics[i]['f1_score'])
        best_f1 = metrics[best_iter]['f1_score']
        best_model_path = MODEL_DIR / f"model_iter_{best_iter}.pkl"
        
        print(f"最佳模型来自迭代 {best_iter+1}，F1分数: {best_f1:.4f}")
        print(f"最佳模型路径: {best_model_path}")
    else:
        print("没有足够的指标数据确定最佳模型")
    
    print("\n程序结束")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        import traceback
        print(f"程序出现错误: {e}")
        traceback.print_exc() 