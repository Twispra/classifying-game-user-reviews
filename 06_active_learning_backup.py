import os
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.multioutput import MultiOutputClassifier
from transformers import BertTokenizer, BertModel
import time
from collections import Counter
from tqdm import tqdm

# ===== 标签体系定义 =====
# 类别映射: 二进制编码 -> 类别名称
CLASS_MAPPING = {
    "000": "无意义/情绪化评论",
    "001": "游戏性",
    "010": "游戏优化",
    "011": "游戏优化+游戏性",
    "100": "游戏环境",
    "101": "游戏环境+游戏性",
    "110": "游戏环境+游戏优化",
    "111": "游戏环境+游戏优化+游戏性"
}

# 每个维度对应的标签
LABEL_COMPONENTS = {
    "游戏环境": 0,  # 二进制第一位
    "游戏优化": 1,  # 二进制第二位
    "游戏性": 2     # 二进制第三位
}


# ===== 标签转换函数 =====
def binary_to_int(binary_str):
    """将二进制字符串转换为整数"""
    try:
        return int(binary_str, 2)
    except ValueError:
        print(f"无效的二进制字符串: {binary_str}")
        return 0

def int_to_binary(label_int, length=3):
    """将整数转换为指定长度的二进制字符串"""
    try:
        return format(int(label_int), f'0{length}b')
    except (ValueError, TypeError):
        print(f"无效的标签整数: {label_int}")
        return "0" * length

def binary_to_description(binary_str):
    """将二进制标签转换为中文描述"""
    # 检查是否在预定义映射中
    if binary_str in CLASS_MAPPING:
        return CLASS_MAPPING[binary_str]
    
    # 否则构建描述
    components = []
    if len(binary_str) >= 3:
        if binary_str[0] == '1': components.append("游戏环境")
        if binary_str[1] == '1': components.append("游戏优化")
        if binary_str[2] == '1': components.append("游戏性")
    
    return "+".join(components) if components else "无意义/情绪化评论"

def combine_labels(game_environment=False, game_optimization=False, game_playability=False):
    """将多个独立标签组合为一个整数标签"""
    binary = [
        '1' if game_environment else '0',
        '1' if game_optimization else '0',
        '1' if game_playability else '0'
    ]
    return binary_to_int(''.join(binary))

def extract_labels(label_int):
    """从整数标签中提取独立标签"""
    binary = int_to_binary(label_int)
    return {
        "游戏环境": binary[0] == '1',
        "游戏优化": binary[1] == '1',
        "游戏性": binary[2] == '1'
    }


# ===== 程序说明函数 =====
def print_class_info():
    """打印类别信息，方便用户理解标签体系"""
    print("\n===== 游戏评论多标签分类体系 =====")
    print("使用3位二进制数表示评论的多个标签:")
    print("第1位 (左): 游戏环境 - 游戏社区、玩家体验、社交互动等")
    print("第2位 (中): 游戏优化 - 性能、卡顿、崩溃等技术问题") 
    print("第3位 (右): 游戏性 - 玩法、难度、乐趣、游戏设计等")
    print("\n标签组合示例:")
    
    for binary, class_name in sorted(CLASS_MAPPING.items()):
        # 计算标签有多少个1
        ones_count = binary.count('1')
        
        # 添加指示符号
        if ones_count == 0:
            indicator = "(无标签)"
        elif ones_count == 1: 
            indicator = "(单一标签)"
        elif ones_count == 2:
            indicator = "(两个标签)"
        else:
            indicator = "(全部标签)"
            
        print(f"  {binary}: {class_name} {indicator}")
        
    print("\n您可以为每条评论选择多个标签组合")
    print("===============================")


class QueryByCommittee:
    """查询委员会主动学习方法"""
    def __init__(self, n_estimators=3, timeout=60, debug=False):
        """初始化委员会
        
        Args:
            n_estimators: 委员会成员数量
            timeout: 样本选择的超时时间（秒）
            debug: 是否输出调试信息
        """
        self.committee_models = []
        self.n_estimators = n_estimators
        self.timeout = timeout  # 添加超时参数
        self.debug = debug      # 添加调试标志
    
    def fit(self, X, y):
        """训练委员会模型"""
        if len(X) == 0 or len(y) == 0:
            print("警告: 没有训练数据，无法训练模型")
            return False
            
        # 确定有多少个类别
        multilabels = []
        for label_int in y:
            binary = int_to_binary(label_int)
            multilabel = [int(b) for b in binary]
            multilabels.append(multilabel)
        
        # 转换为多标签格式
        try:
            y_multilabel = np.array(multilabels)
        except Exception as e:
            print(f"转换为多标签格式出错: {e}")
            return False
        
        # 创建并训练模型
        self.committee_models = []
        
        # 基本随机森林模型
        base_model = RandomForestClassifier(
            n_estimators=50, 
            random_state=42
        )
        model = MultiOutputClassifier(base_model)
        model.fit(X, y_multilabel)
        self.committee_models.append(model)
        
        # 额外的模型，使用不同参数
        for i in range(1, self.n_estimators):
            random_seed = 42 + i * 10
            base_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=3 + i * 2,
                random_state=random_seed
            )
            model = MultiOutputClassifier(base_model)
            model.fit(X, y_multilabel)
            self.committee_models.append(model)
            
        print(f"已训练 {len(self.committee_models)} 个委员会模型")
        return True
    
    def add_pretrained_model(self, model):
        """添加预训练模型到委员会"""
        if model is not None:
            self.committee_models.append(model)
            return True
        return False
    
    def calculate_vote_entropy(self, X_unlabeled):
        """计算未标注样本的投票熵 (优化版)"""
        n_samples = len(X_unlabeled)
        n_positions = 3  # 三个二进制位置
        batch_size = min(100, n_samples)  # 批处理大小
        
        # 初始化存储结构
        position_entropies = np.zeros((n_samples, n_positions))
        
        print(f"正在计算 {n_samples} 个样本的熵值...")
        start_time = time.time()
        
        # 批量处理样本
        for batch_start in range(0, n_samples, batch_size):
            # 检查超时
            if time.time() - start_time > self.timeout:
                print(f"警告: 熵计算已运行 {self.timeout} 秒，超时中断")
                break
                
            batch_end = min(batch_start + batch_size, n_samples)
            if batch_start % (batch_size * 5) == 0 or batch_start + batch_size >= n_samples:
                print(f"处理样本 {batch_start+1}-{batch_end} / {n_samples} ({(batch_end/n_samples*100):.1f}%)")
            
            # 批量获取每个模型的预测
            batch_X = X_unlabeled[batch_start:batch_end]
            batch_votes = []
            
            for m, model in enumerate(self.committee_models):
                try:
                    preds = model.predict(batch_X)
                    batch_votes.append(preds)
                    if self.debug and m == 0 and batch_start == 0:
                        print(f"模型{m}示例预测: {preds[0]}")
                except Exception as e:
                    print(f"模型{m}预测出错: {e}")
                    # 创建空预测
                    empty_preds = np.zeros((len(batch_X), n_positions), dtype=int)
                    batch_votes.append(empty_preds)
            
            # 计算每个样本每个位置的熵
            for i in range(len(batch_X)):
                sample_idx = batch_start + i
                for pos in range(n_positions):
                    votes_0 = 0
                    votes_1 = 0
                    
                    # 统计所有模型对该位置的投票
                    for model_idx in range(len(self.committee_models)):
                        try:
                            vote = int(batch_votes[model_idx][i][pos])
                            if vote == 0:
                                votes_0 += 1
                            else:
                                votes_1 += 1
                        except Exception as e:
                            if self.debug:
                                print(f"处理模型{model_idx}样本{sample_idx}位置{pos}的投票出错: {e}")
                            votes_0 += 1  # 默认投0票
                    
                    # 计算熵
                    total = votes_0 + votes_1
                    entropy = 0
                    if total > 0:  # 防止除零
                        if votes_0 > 0:
                            p_0 = votes_0 / total
                            entropy -= p_0 * np.log2(p_0)
                        if votes_1 > 0:
                            p_1 = votes_1 / total
                            entropy -= p_1 * np.log2(p_1)
                    
                    position_entropies[sample_idx, pos] = entropy
        
        # 计算平均熵
        sample_entropies = np.mean(position_entropies, axis=1)
        
        # 输出耗时
        elapsed = time.time() - start_time
        print(f"熵计算完成，耗时 {elapsed:.2f} 秒")
        
        # 输出一些统计信息
        if len(sample_entropies) > 0:
            print(f"熵值统计: 最小={np.min(sample_entropies):.4f}, 最大={np.max(sample_entropies):.4f}, 平均={np.mean(sample_entropies):.4f}")
        
        return sample_entropies
    
    def select_samples(self, X_unlabeled, n_samples=5):
        """选择最有价值的样本进行标注 (增加超时保护)"""
        print(f"开始选择 {n_samples} 个高价值样本...")
        start_time = time.time()
        
        # 处理边缘情况
        if len(X_unlabeled) <= n_samples:
            print(f"可用样本数 ({len(X_unlabeled)}) 小于等于请求样本数 ({n_samples})，返回所有样本")
            return np.arange(len(X_unlabeled))
        
        # 随机选择一部分样本用于熵计算，避免计算过多样本
        if len(X_unlabeled) > 5000 and n_samples < 500:
            sample_size = min(5000, len(X_unlabeled) // 2)
            print(f"样本过多 ({len(X_unlabeled)})，随机选择 {sample_size} 个进行熵计算")
            subsample_indices = np.random.choice(len(X_unlabeled), sample_size, replace=False)
            X_subsample = X_unlabeled[subsample_indices]
        else:
            X_subsample = X_unlabeled
            subsample_indices = np.arange(len(X_unlabeled))
        
        # 计算样本熵
        try:
            entropies = self.calculate_vote_entropy(X_subsample)
            
            # 检查超时
            if time.time() - start_time > self.timeout:
                print(f"警告: 样本选择已运行 {self.timeout} 秒，超时中断，改为随机选择")
                return np.random.choice(len(X_unlabeled), min(n_samples, len(X_unlabeled)), replace=False)
            
            # 检查是否有有效熵值
            if len(entropies) == 0 or np.all(np.isnan(entropies)):
                print("警告: 没有获得有效的熵值，采用随机选择")
                return np.random.choice(len(X_unlabeled), min(n_samples, len(X_unlabeled)), replace=False)
            
            # 选择熵最高的样本
            high_entropy_indices = np.argsort(-entropies)[:min(n_samples, len(entropies))]
            
            # 如果使用了子样本，转换回原始索引
            if len(X_unlabeled) != len(X_subsample):
                selected_indices = subsample_indices[high_entropy_indices]
            else:
                selected_indices = high_entropy_indices
            
            print(f"已选择 {len(selected_indices)} 个高熵样本，耗时 {time.time() - start_time:.2f} 秒")
            return selected_indices
            
        except Exception as e:
            print(f"选择样本出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 出错时随机选择
            print("改为随机选择样本...")
            return np.random.choice(len(X_unlabeled), min(n_samples, len(X_unlabeled)), replace=False)


def load_embeddings(filepath):
    """加载嵌入向量，处理兼容性问题"""
    import numpy as np  # 将导入移到函数开头，确保在任何路径中都能使用np
    
    if not os.path.exists(filepath):
        print(f"错误: 嵌入向量文件不存在: {filepath}")
        raise FileNotFoundError(f"找不到文件: {filepath}")
        
    print(f"正在加载嵌入向量: {filepath}")
    print(f"文件大小: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
    
    try:
        # 方法1: 标准加载
        print("尝试标准加载方式...")
        embeddings = torch.load(filepath)
        print("标准加载成功")
    except Exception as e1:
        print(f"标准加载失败: {str(e1)}")
        
        try:
            # 方法2: 使用weights_only参数
            print("尝试使用weights_only参数...")
            embeddings = torch.load(filepath, weights_only=False)
            print("使用weights_only参数加载成功")
        except Exception as e2:
            print(f"使用weights_only参数加载失败: {str(e2)}")
            
            try:
                # 方法3: 使用map_location参数
                print("尝试使用map_location参数...")
                embeddings = torch.load(filepath, map_location='cpu')
                print("使用map_location参数加载成功")
            except Exception as e3:
                print(f"使用map_location参数加载失败: {str(e3)}")
                
                # 最后尝试
                print("尝试最后的加载方式...")
                try:
                    # 不需要再次导入numpy，已在函数开头导入
                    import pickle
                    
                    # 使用pickle直接加载
                    with open(filepath, 'rb') as f:
                        embeddings = pickle.load(f)
                    print("使用pickle加载成功")
                except Exception as e4:
                    print(f"所有加载方法都失败: {str(e4)}")
                    raise ValueError(f"无法加载嵌入向量文件: {filepath}")
    
    # 检查并转换数据类型
    if isinstance(embeddings, torch.Tensor):
        print(f"加载的是PyTorch张量, 形状: {embeddings.shape}, 类型: {embeddings.dtype}")
        embeddings = embeddings.cpu().numpy()
    elif isinstance(embeddings, np.ndarray):
        print(f"加载的是NumPy数组, 形状: {embeddings.shape}, 类型: {embeddings.dtype}")
    else:
        print(f"警告: 加载的数据类型是 {type(embeddings)}, 尝试转换为NumPy数组")
        
        # 尝试转换为NumPy数组
        try:
            embeddings = np.array(embeddings)
            print(f"成功转换为NumPy数组, 形状: {embeddings.shape}")
        except Exception as e:
            print(f"转换为NumPy数组失败: {str(e)}")
            raise ValueError("加载的嵌入向量格式不支持")
    
    # 输出一些统计信息以验证数据
    print(f"嵌入向量统计信息:")
    print(f"- 形状: {embeddings.shape}")
    print(f"- 均值: {np.mean(embeddings):.6f}")
    print(f"- 标准差: {np.std(embeddings):.6f}")
    print(f"- 最小值: {np.min(embeddings):.6f}")
    print(f"- 最大值: {np.max(embeddings):.6f}")

    return embeddings


class AnnotationInterface:
    """评论标注界面"""
    def __init__(self, reviews_df, model_predictions=None, existing_annotations=None, priority_indices=None):
        """初始化标注界面"""
        self.reviews_df = reviews_df
        self.reviews = reviews_df["review_text"].tolist()
        self.predictions = {} if model_predictions is None else model_predictions
        self.annotations = {} if existing_annotations is None else existing_annotations.copy()
        self.priority_indices = [] if priority_indices is None else priority_indices.copy()
        self.save_path = "data/annotations.json"
        self.current_index = None
        
        # 添加优先样本队列，用于追踪未标注的优先样本
        self.pending_priority_indices = self.priority_indices.copy()
        
        # 记录标注历史，方便回退
        self.history = []
        
        # 会话信息
        self.session_start = time.time()
        self.session_count = 0
        
        # 统计信息
        self.priority_annotated = 0
        self.normal_annotated = 0
        
        # 当前是否在优先样本模式
        self.priority_mode = True
        
        # GUI相关变量
        self.root = None
        self.var_environment = None
        self.var_optimization = None
        self.var_gameplay = None
        self.review_text = None
        self.pred_text = None
        self.status_text = None
        self.mode_indicator = None
    
    def run(self):
        """运行标注界面"""
        # 检查数据有效性
        if not self.reviews:
            print("错误: 没有评论数据可标注")
            return self.annotations
            
        # 创建GUI界面
        self.create_gui()
        
        # 首先尝试加载优先样本
        if self.pending_priority_indices:
            print(f"发现 {len(self.pending_priority_indices)} 个优先样本待标注")
            self.priority_mode = True
            self.find_next_priority_sample()
        else:
            print("没有优先样本，将显示普通未标注样本")
            self.priority_mode = False
            self.find_next_regular_sample()
        
        # 如果没有找到任何样本
        if self.current_index is None:
            print("提示: 没有未标注的样本")
            self.root.destroy()
            return self.annotations
            
        # 显示当前评论
        self.load_current_review()
        
        # 更新进度
        self.update_progress()
        
        # 更新模式指示器
        self.update_mode_indicator()
        
        # 运行主循环
        self.root.mainloop()
        
        return self.annotations
    
    def create_gui(self):
        """创建GUI界面"""
        self.root = tk.Tk()
        self.root.title("游戏评论标注工具")
        self.root.geometry("900x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部信息区域
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        # 标题
        title = ttk.Label(info_frame, text="游戏评论标注工具", font=("Arial", 14, "bold"))
        title.pack(pady=5)
        
        # 进度指示
        self.progress_text = ttk.Label(info_frame, text="")
        self.progress_text.pack(pady=5)
        
        # 模式指示器（优先/普通模式）
        self.mode_indicator = ttk.Label(info_frame, text="", font=("Arial", 12, "bold"))
        self.mode_indicator.pack(pady=5)
        
        # 优先样本指示
        self.priority_indicator = ttk.Label(info_frame, text="", font=("Arial", 12, "bold"))
        self.priority_indicator.pack(pady=5)
        
        # 评论区域
        review_frame = ttk.LabelFrame(main_frame, text="评论内容")
        review_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.review_text = tk.Text(review_frame, wrap=tk.WORD, height=15)
        self.review_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(review_frame, orient="vertical", command=self.review_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.review_text.config(yscrollcommand=scrollbar.set)
        
        # 预测信息
        pred_frame = ttk.LabelFrame(main_frame, text="模型预测")
        pred_frame.pack(fill=tk.X, pady=5)
        
        self.pred_text = ttk.Label(pred_frame, text="无预测信息", wraplength=800)
        self.pred_text.pack(anchor=tk.W, pady=5)
        
        # 标签选择
        label_frame = ttk.LabelFrame(main_frame, text="选择标签（可多选）")
        label_frame.pack(fill=tk.X, pady=5)
        
        # 标签说明
        info_text = ("标签说明：\n"
                    "1. 游戏环境：游戏社区、玩家体验、社交互动等\n"
                    "2. 游戏优化：性能、卡顿、崩溃等技术问题\n"
                    "3. 游戏性：玩法、难度、乐趣、游戏设计等")
        info_label = ttk.Label(label_frame, text=info_text, wraplength=800)
        info_label.pack(anchor=tk.W, pady=5)
        
        # 复选框
        checkbox_frame = ttk.Frame(label_frame)
        checkbox_frame.pack(fill=tk.X, pady=5)
        
        self.var_environment = tk.BooleanVar()
        self.var_optimization = tk.BooleanVar()
        self.var_gameplay = tk.BooleanVar()
        
        ttk.Checkbutton(checkbox_frame, text="游戏环境", variable=self.var_environment).pack(side=tk.LEFT, padx=20)
        ttk.Checkbutton(checkbox_frame, text="游戏优化", variable=self.var_optimization).pack(side=tk.LEFT, padx=20)
        ttk.Checkbutton(checkbox_frame, text="游戏性", variable=self.var_gameplay).pack(side=tk.LEFT, padx=20)
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 导航按钮
        ttk.Button(button_frame, text="上一条", command=self.previous_review).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="保存并继续", command=self.next_review).pack(side=tk.LEFT, padx=5)
        
        # 模式切换按钮
        self.mode_button = ttk.Button(button_frame, text="切换到普通样本", command=self.toggle_mode)
        self.mode_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="保存所有", command=self.save_all).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="完成标注", command=self.complete_annotation).pack(side=tk.RIGHT, padx=5)
        
        # 状态栏
        self.status_text = ttk.Label(main_frame, text="就绪")
        self.status_text.pack(anchor=tk.W, pady=5)
        
        # 设置自动保存
        self.root.after(60000, self.auto_save)  # 每分钟自动保存
    
    def update_mode_indicator(self):
        """更新模式指示器"""
        if self.priority_mode:
            self.mode_indicator.config(text="当前模式: 优先样本标注", foreground="#FF6600")
            self.mode_button.config(text="切换到普通样本")
        else:
            self.mode_indicator.config(text="当前模式: 普通样本标注", foreground="#006699")
            self.mode_button.config(text="切换到优先样本")
    
    def toggle_mode(self):
        """切换优先/普通样本模式"""
        self.save_current_annotation()
        
        self.priority_mode = not self.priority_mode
        
        if self.priority_mode:
            # 切换到优先样本模式
            if self.pending_priority_indices:
                self.find_next_priority_sample()
                self.status_text.config(text="已切换到优先样本模式")
            else:
                messagebox.showinfo("提示", "没有更多优先样本待标注")
                self.priority_mode = False  # 切回普通模式
                self.find_next_regular_sample()
        else:
            # 切换到普通样本模式
            self.find_next_regular_sample()
            self.status_text.config(text="已切换到普通样本模式")
        
        self.update_mode_indicator()
        self.load_current_review()
    
    def find_next_priority_sample(self):
        """查找下一个未标注的优先样本"""
        # 直接从优先队列中取样本
        while self.pending_priority_indices:
            idx = self.pending_priority_indices[0]
            # 检查索引有效性
            if idx < 0 or idx >= len(self.reviews):
                self.pending_priority_indices.pop(0)
                continue
                
            # 获取原始索引
            try:
                original_idx = int(self.reviews_df.iloc[idx]["original_index"])
                # 检查是否已标注
                if original_idx not in self.annotations:
                    self.current_index = idx
                    return
                else:
                    # 已标注，从待处理队列移除
                    self.pending_priority_indices.pop(0)
            except Exception as e:
                print(f"处理优先索引 {idx} 出错: {e}")
                self.pending_priority_indices.pop(0)
        
        # 没有找到未标注的优先样本
        self.current_index = None
    
    def find_next_regular_sample(self):
        """查找下一个未标注的普通样本（非优先）"""
        for i in range(len(self.reviews)):
            # 跳过优先样本
            if i in self.priority_indices:
                continue
                
            try:
                original_idx = int(self.reviews_df.iloc[i]["original_index"])
                if original_idx not in self.annotations:
                    self.current_index = i
                    return
            except:
                continue
        
        # 没有找到未标注样本
        self.current_index = None
    
    def load_current_review(self):
        """加载当前评论到界面"""
        if self.current_index is None or self.current_index >= len(self.reviews):
            self.status_text.config(text="警告: 无效的评论索引")
            return
            
        # 清空文本框
        self.review_text.delete(1.0, tk.END)
        
        try:
            # 获取评论
            review = str(self.reviews[self.current_index])
            original_idx = int(self.reviews_df.iloc[self.current_index]["original_index"])
            
            # 显示评论（处理过长内容）
            max_len = 5000
            if len(review) > max_len:
                review = review[:max_len] + "...[内容过长已截断]"
            
            self.review_text.insert(tk.END, f"评论编号: {original_idx}\n\n{review}")
            
            # 重置标签选择
            self.var_environment.set(False)
            self.var_optimization.set(False)
            self.var_gameplay.set(False)
            
            # 如果已有标注，显示已有标签
            if original_idx in self.annotations:
                label = self.annotations[original_idx]
                binary = format(label, '03b')
                self.var_environment.set(binary[0] == '1')
                self.var_optimization.set(binary[1] == '1')
                self.var_gameplay.set(binary[2] == '1')
            
            # 显示预测信息
            if original_idx in self.predictions:
                pred = self.predictions[original_idx]
                binary = format(pred, '03b')
                desc = binary_to_description(binary)
                self.pred_text.config(text=f"预测标签: {binary} ({desc})")
            else:
                self.pred_text.config(text="无预测信息")
            
            # 更新样本类型指示
            is_priority = self.current_index in self.priority_indices
            
            if is_priority:
                self.priority_indicator.config(text="⭐ 高价值样本 - 模型认为此样本信息量大 ⭐", foreground="#FF6600")
                self.review_text.config(bg="#FFEECC")  # 淡橙色背景
            else:
                self.priority_indicator.config(text="常规样本", foreground="#666666")
                self.review_text.config(bg="#FFFFFF")  # 白色背景
                
            # 滚动到顶部
            self.review_text.see("1.0")
            
        except Exception as e:
            print(f"加载评论出错: {e}")
            self.status_text.config(text=f"加载评论出错: {str(e)}")
    
    def update_progress(self):
        """更新进度显示"""
        total = len(self.reviews)
        annotated = len(self.annotations)
        
        # 计算未标注样本数量
        remaining_total = 0
        remaining_priority = 0
        
        for i in range(len(self.reviews)):
            try:
                original_idx = int(self.reviews_df.iloc[i]["original_index"])
                if original_idx not in self.annotations:
                    remaining_total += 1
                    if i in self.priority_indices:
                        remaining_priority += 1
            except:
                continue
        
        # 计算完成百分比
        completion = annotated / total * 100 if total > 0 else 0
        
        # 创建进度文本
        progress = f"已标注: {annotated}/{total} ({completion:.1f}%)"
        stats = f"待标注: {remaining_total}条 (其中优先样本: {remaining_priority}条)"
        session = f"本次会话: {self.session_count}条标注"
        
        # 更新标签文本
        self.progress_text.config(text=f"{progress} | {stats} | {session}")
    
    def save_current_annotation(self):
        """保存当前标注"""
        if self.current_index is None or self.current_index >= len(self.reviews):
            self.status_text.config(text="警告: 无效的评论索引，无法保存")
            return False
            
        try:
            # 获取标签选择
            env = int(self.var_environment.get())
            opt = int(self.var_optimization.get())
            gameplay = int(self.var_gameplay.get())
            
            # 构建二进制标签
            binary = f"{env}{opt}{gameplay}"
            label = int(binary, 2)
            
            # 获取原始索引
            original_idx = int(self.reviews_df.iloc[self.current_index]["original_index"])
            
            # 保存标注
            is_new = original_idx not in self.annotations
            self.annotations[original_idx] = label
            
            if is_new:
                self.session_count += 1
                # 更新统计信息
                if self.current_index in self.priority_indices:
                    self.priority_annotated += 1
                else:
                    self.normal_annotated += 1
                    
            # 如果是优先样本，从待处理队列中移除
            if self.current_index in self.pending_priority_indices:
                self.pending_priority_indices.remove(self.current_index)
                
            self.status_text.config(text=f"已保存评论 {original_idx} 的标注: {binary} ({binary_to_description(binary)})")
            self.update_progress()
            
            return True
        except Exception as e:
            error_msg = f"保存标注出错: {e}"
            print(error_msg)
            self.status_text.config(text=error_msg)
            return False
    
    def next_review(self):
        """保存当前标注并前进到下一条"""
        if self.save_current_annotation():
            # 每保存5条自动保存一次
            if self.session_count % 5 == 0:
                self.save_all(show_message=False)
                
            # 根据当前模式查找下一条
            if self.priority_mode:
                self.find_next_priority_sample()
                # 如果优先样本用完了，询问是否切换到普通模式
                if self.current_index is None:
                    if messagebox.askyesno("提示", "优先样本已全部标注完成。是否切换到普通样本模式？"):
                        self.priority_mode = False
                        self.find_next_regular_sample()
                        self.update_mode_indicator()
            else:
                self.find_next_regular_sample()
            
            # 如果找到了样本，加载它
            if self.current_index is not None:
                self.load_current_review()
            else:
                messagebox.showinfo("标注完成", "所有样本已标注完成！")
                self.status_text.config(text="已完成所有标注")
    
    def previous_review(self):
        """保存当前标注并返回上一条"""
        self.save_current_annotation()
        
        # TODO: 实现历史记录功能，支持回退到上一个标注过的样本
        messagebox.showinfo("提示", "历史浏览功能尚未实现")
    
    def save_all(self, show_message=True):
        """保存所有标注到文件"""
        self.save_current_annotation()
        
        try:
            # 准备数据
            data = {
                "annotations": {str(k): v for k, v in self.annotations.items()},
                "timestamp": datetime.now().isoformat(),
                "completed": False,
                "session_info": {
                    "duration": time.time() - self.session_start,
                    "count": self.session_count,
                    "priority_annotated": self.priority_annotated,
                    "normal_annotated": self.normal_annotated
                }
            }
            
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            # 保存数据
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            save_msg = f"已保存 {len(self.annotations)} 条标注到 {self.save_path}"
            self.status_text.config(text=save_msg)
            
            if show_message:
                messagebox.showinfo("保存成功", save_msg)
                
            return True
        except Exception as e:
            error_msg = f"保存标注文件出错: {e}"
            print(error_msg)
            if show_message:
                messagebox.showerror("保存失败", error_msg)
            return False
    
    def complete_annotation(self):
        """完成标注过程"""
        # 检查是否还有优先样本未标注
        remaining_priority = len([idx for idx in self.pending_priority_indices 
                                if idx >= 0 and idx < len(self.reviews)])
        
        if remaining_priority > 0:
            if not messagebox.askyesno("警告", 
                                     f"还有 {remaining_priority} 个优先样本未标注。\n"
                                     f"这些样本对模型训练特别重要。\n"
                                     f"确定要现在结束标注吗？"):
                return
        
        # 确认是否结束标注
        if not messagebox.askyesno("确认", 
                                  "标记为完成后将结束主动学习过程，直接训练最终模型。\n"
                                  "确定要完成标注吗？"):
            return
            
        # 保存当前标注
        self.save_current_annotation()
        
        try:
            # 准备数据，标记为已完成
            data = {
                "annotations": {str(k): v for k, v in self.annotations.items()},
                "timestamp": datetime.now().isoformat(),
                "completed": True,
                "session_info": {
                    "duration": time.time() - self.session_start,
                    "count": self.session_count,
                    "priority_annotated": self.priority_annotated,
                    "normal_annotated": self.normal_annotated
                }
            }
            
            # 保存数据
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            messagebox.showinfo("完成标注", "标注已标记为完成，将跳过主动学习阶段，直接训练最终模型")
            self.root.destroy()
            
        except Exception as e:
            error_msg = f"完成标注出错: {e}"
            print(error_msg)
            messagebox.showerror("错误", error_msg)
    
    def auto_save(self):
        """自动保存标注"""
        try:
            self.save_all(show_message=False)
            print("自动保存成功")
        except Exception as e:
            print(f"自动保存失败: {e}")
            
        # 每分钟重新调度
        self.root.after(60000, self.auto_save)
    
    def on_closing(self):
        """窗口关闭处理"""
        if messagebox.askyesno("确认", "是否保存当前标注？"):
            self.save_all(show_message=False)
        self.root.destroy()


class ModelManager:
    """模型管理器，负责模型的保存、加载和预测"""
    
    def __init__(self, model_dir="model_ckpt"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, model, iteration):
        """保存模型到指定目录"""
        try:
            import joblib
            
            # 规范化迭代标识
            if iteration == "final":
                model_path = os.path.join(self.model_dir, "model_final.pkl")
            else:
                # 确保iteration是整数并转换为统一格式
                try:
                    if isinstance(iteration, str) and iteration.startswith('iter_'):
                        iter_num = int(iteration.split('_')[1])
                    else:
                        iter_num = int(iteration)
                    model_path = os.path.join(self.model_dir, f"model_iter_{iter_num}.pkl")
                except:
                    # 如果转换失败，使用原始字符串
                    model_path = os.path.join(self.model_dir, f"model_{iteration}.pkl")
            
            # 保存模型
            joblib.dump(model, model_path)
            print(f"模型已保存到: {model_path}")
            return model_path
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            return None
    
    def load_model(self, iteration):
        """从指定目录加载模型"""
        try:
            import joblib
            
            # 规范化迭代标识
            if iteration == "final":
                model_path = os.path.join(self.model_dir, "model_final.pkl")
                # 如果final模型不存在，尝试加载最后一次迭代的模型
                if not os.path.exists(model_path):
                    print(f"未找到最终模型 {model_path}，尝试加载最后一次迭代的模型...")
                    # 查找最大的迭代次数
                    iter_files = [f for f in os.listdir(self.model_dir) if f.startswith("model_iter_") and f.endswith(".pkl")]
                    if iter_files:
                        iter_nums = [int(f.split("_")[2].split(".")[0]) for f in iter_files]
                        max_iter = max(iter_nums)
                        return self.load_model(max_iter)
            else:
                # 确保iteration是整数并转换为统一格式
                try:
                    if isinstance(iteration, str) and iteration.startswith('iter_'):
                        iter_num = int(iteration.split('_')[1])
                    else:
                        iter_num = int(iteration)
                    model_path = os.path.join(self.model_dir, f"model_iter_{iter_num}.pkl")
                except:
                    # 如果转换失败，使用原始字符串
                    model_path = os.path.join(self.model_dir, f"model_{iteration}.pkl")
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                print(f"模型已从 {model_path} 加载")
                return model
            else:
                print(f"模型文件不存在: {model_path}")
                return None
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return None
    
    def get_latest_model(self):
        """获取最新的模型"""
        # 首先尝试加载最终模型
        model = self.load_model("final")
        if model is not None:
            return model
            
        # 如果没有最终模型，尝试找到最高迭代次数的模型
        iter_files = [f for f in os.listdir(self.model_dir) if f.startswith("model_iter_") and f.endswith(".pkl")]
        if not iter_files:
            return None
            
        # 提取迭代次数并找到最大值
        iter_nums = []
        for f in iter_files:
            try:
                iter_num = int(f.split("_")[2].split(".")[0])
                iter_nums.append(iter_num)
            except:
                continue
                
        if not iter_nums:
            return None
            
        max_iter = max(iter_nums)
        return self.load_model(max_iter)
    
    def predict_new_reviews(self, model, new_reviews, embeddings_model):
        """
        预测新的评论
        
        Args:
            model: 训练好的模型
            new_reviews: 新的评论列表
            embeddings_model: 用于生成评论嵌入的模型
            
        Returns:
            predictions: 预测结果列表
            probabilities: 预测概率列表
        """
        # 确保模型存在
        if model is None:
            raise ValueError("模型为空，无法进行预测")
            
        # 生成新评论的嵌入向量
        new_embeddings = embeddings_model.encode(new_reviews)
        
        # 进行预测
        predictions = model.predict(new_embeddings)
        probabilities = model.predict_proba(new_embeddings)
        
        return predictions, probabilities


def custom_train_test_split(X, y, indices=None, test_size=0.2, random_state=None):
    """
    自定义的训练集/测试集分割函数，可以同时返回对应的索引
    
    参数:
    - X: 特征矩阵
    - y: 标签数组
    - indices: 可选，原始索引数组，如果提供则会返回对应的训练和测试索引
    - test_size: 测试集比例
    - random_state: 随机种子
    
    返回:
    - X_train, X_test, y_train, y_test, train_indices, test_indices
    """
    # 如果没有提供indices，创建一个假的索引数组
    if indices is None:
        indices = np.arange(len(X))
    
    # 使用sklearn的train_test_split划分数据
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    return X_train, X_test, y_train, y_test, train_indices, test_indices


def main():
    """主函数：实现主动学习流程"""
    # 创建必要目录
    os.makedirs("data", exist_ok=True)
    os.makedirs("model_ckpt", exist_ok=True)
    
    # 基本配置
    print_class_info()
    max_iterations = 10  # 最大迭代次数
    samples_per_iter = 5  # 每轮选择的样本数
    annotations_path = "data/annotations.json"
    embeddings_path = "data/review_embeddings.pt"
    model_manager = ModelManager()
    
    # ==== 1. 加载数据 ====
    print("\n[加载数据]")
    
    # 加载预处理后的评论数据
    try:
        all_reviews_df = pd.read_csv("data/preprocessed_reviews.csv")
        if 'original_index' not in all_reviews_df.columns:
            all_reviews_df['original_index'] = all_reviews_df.index
            print("已添加original_index列")
        print(f"已加载 {len(all_reviews_df)} 条评论数据")
    except Exception as e:
        print(f"加载评论数据出错: {e}")
        return
    
    # 加载嵌入向量
    try:
        embeddings = load_embeddings(embeddings_path)
        print(f"嵌入向量形状: {embeddings.shape}")
    except Exception as e:
        print(f"加载嵌入向量失败: {e}")
        return
    
    # 检查并加载标注数据
    existing_annotations = {}
    annotation_completed = False
    
    # 尝试从JSON文件加载标注
    if os.path.exists(annotations_path):
        print(f"发现标注文件: {annotations_path}")
        # 直接加载现有标注，取消询问是否重置
        try:
            with open(annotations_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_annotations = {int(k): int(v) for k, v in data['annotations'].items()}
                annotation_completed = data.get('completed', False)
                if annotation_completed:
                    print("标注已标记为完成。将继续使用现有标注数据进行模型训练。")
                print(f"已加载 {len(existing_annotations)} 条标注")
        except Exception as e:
            print(f"加载标注出错: {e}")
            existing_annotations = {}
            annotation_completed = False
    
    # 为标注为空的情况尝试加载初始种子数据
    if not existing_annotations:
        seed_paths = ["data/initial_seeds.csv", "data/labeled_initial_seeds.csv"]
        for path in seed_paths:
            if os.path.exists(path):
                try:
                    seeds_df = pd.read_csv(path)
                    if 'label' in seeds_df.columns and 'original_index' in seeds_df.columns:
                        for _, row in seeds_df.iterrows():
                            existing_annotations[int(row['original_index'])] = int(row['label'])
                        print(f"已从{path}加载 {len(existing_annotations)} 条初始种子标注")
                        break
                except Exception as e:
                    print(f"加载种子数据{path}出错: {e}")
    
    # 创建原始索引与嵌入向量索引的映射
    # 注意: 原始索引可能与嵌入向量索引不匹配，需要建立映射
    original_indices = all_reviews_df["original_index"].values
    
    print(f"评论数据索引范围: {min(original_indices)} 到 {max(original_indices)}")
    print(f"嵌入向量维度: {embeddings.shape}")
    
    # 检查embeddings长度是否与评论数量匹配
    if len(embeddings) != len(all_reviews_df):
        print(f"警告: 嵌入向量数量 ({len(embeddings)}) 与评论数量 ({len(all_reviews_df)}) 不匹配")
        
        # 检查是否可以通过original_index建立映射
        if max(original_indices) < len(embeddings):
            print("使用original_index作为嵌入向量的索引")
        else:
            print("警告: original_index超出嵌入向量范围，可能存在索引错误")
            
    # 建立映射数组: original_index -> embedding_index
    # 假设original_index是embeddings的有效索引
    index_map = {}
    for i, idx in enumerate(original_indices):
        if idx < len(embeddings):
            index_map[int(idx)] = int(idx)
    
    print(f"建立了 {len(index_map)} 个索引映射")
    
    # 获取已标注和未标注样本的索引
    labeled_orig_indices = np.array([idx for idx in existing_annotations.keys() if idx in index_map])
    unlabeled_orig_indices = np.array([idx for idx in original_indices if idx not in existing_annotations and idx in index_map])
    
    print(f"已标注样本: {len(labeled_orig_indices)}")
    print(f"未标注样本: {len(unlabeled_orig_indices)}")
    
    # 提取已标注样本的特征和标签
    if len(labeled_orig_indices) > 0:
        # 使用索引映射获取嵌入向量
        labeled_embed_indices = np.array([index_map[idx] for idx in labeled_orig_indices])
        X_labeled = embeddings[labeled_embed_indices]
        y_labeled = np.array([existing_annotations[idx] for idx in labeled_orig_indices])
        
        # 显示标签分布
        print("\n当前标签分布:")
        label_counts = Counter([int_to_binary(label) for label in y_labeled])
        for binary, count in sorted(label_counts.items()):
            print(f"{binary} ({binary_to_description(binary)}): {count}条")
    else:
        X_labeled = np.array([])
        y_labeled = np.array([])
        print("警告: 没有已标注样本，请先创建一些初始标注")
    
    # ==== 2. 主动学习迭代过程 ====
    if annotation_completed:
        print("\n[跳过主动学习] 标注已标记为完成，直接训练最终模型")
    elif len(labeled_orig_indices) == 0:
        print("\n[跳过主动学习] 没有初始标注样本，请先创建一些标注")
    else:
        # 初始化迭代计数和指标
        current_iteration = find_latest_iteration("model_ckpt")
        print(f"\n[开始主动学习] 从迭代 {current_iteration+1} 开始")
        
        # 记录性能指标
        metrics_history = {
            'iteration': [],
            'accuracy': [],
            'macro_f1': [],
            'weighted_f1': [],
            'samples_count': []
        }
        
        # 主动学习循环
        for i in range(current_iteration + 1, max_iterations + 1):
            print(f"\n=== 第 {i}/{max_iterations} 轮主动学习 ===")
            print(f"当前已标注: {len(labeled_orig_indices)}/{len(original_indices)} 个样本")
            
            # 步骤1: 训练委员会模型
            print("\n[步骤1] 训练委员会模型")
            qbc = QueryByCommittee()
            
            # 加载上一轮模型（如果有）
            if i > 1:
                prev_model = model_manager.load_model(i-1)
                if prev_model:
                    qbc.add_pretrained_model(prev_model)
            
            # 训练当前模型
            qbc.fit(X_labeled, y_labeled)
            
            # 保存当前模型
            model_path = model_manager.save_model(qbc.committee_models[0], i)
            
            # 评估当前模型（如果有足够样本）
            if len(labeled_orig_indices) >= 10:
                try:
                    # 用20%数据评估
                    X_train, X_val, y_train, y_val, _, _ = custom_train_test_split(
                        X_labeled, y_labeled, labeled_orig_indices, test_size=0.2, random_state=42
                    )
                    metrics = evaluate_model(qbc.committee_models[0], X_val, y_val)
                    
                    # 记录性能指标
                    metrics_history['iteration'].append(i)
                    metrics_history['accuracy'].append(metrics['accuracy'])
                    metrics_history['macro_f1'].append(metrics['macro_f1'])
                    metrics_history['weighted_f1'].append(metrics['weighted_f1'])
                    metrics_history['samples_count'].append(len(labeled_orig_indices))
                    
                    # 输出性能曲线
                    plot_learning_curves(metrics_history)
                except Exception as e:
                    print(f"评估模型出错: {e}")
            
            # 步骤2: 选择高价值样本
            print("\n[步骤2] 选择高价值样本")
            
            # 检查是否标注完毕
            if len(unlabeled_orig_indices) == 0:
                print("所有样本已标注完毕")
                break
            
            # 提取未标注样本的嵌入向量
            unlabeled_embed_indices = np.array([index_map[idx] for idx in unlabeled_orig_indices])
            X_unlabeled = embeddings[unlabeled_embed_indices]
            
            # 选择高熵样本
            try:
                priority_indices = qbc.select_samples(X_unlabeled, n_samples=min(samples_per_iter, len(X_unlabeled)))
                # 转换回原始索引
                priority_orig_indices = unlabeled_orig_indices[priority_indices]
                
                # 将原始索引转换为DataFrame行索引
                priority_rows = []
                for idx in priority_orig_indices:
                    rows = all_reviews_df.index[all_reviews_df['original_index'] == idx].tolist()
                    if rows:
                        priority_rows.append(rows[0])
                
                if not priority_rows:
                    print("注意: 无法找到有效的优先样本，将使用随机选择的样本")
                    # 随机选择行索引
                    random_indices = np.random.choice(len(all_reviews_df), 
                                                   min(samples_per_iter, len(all_reviews_df)), 
                                                   replace=False)
                    priority_rows = random_indices.tolist()
                else:
                    print(f"已选择 {len(priority_rows)} 个优先样本供标注 (高熵值样本)")
                
                # 生成预测以辅助标注
                committee_predictions = {}
                for idx in priority_orig_indices:
                    try:
                        # 找到样本在X_unlabeled中的位置
                        sample_idx = np.where(unlabeled_orig_indices == idx)[0][0]
                        X_sample = X_unlabeled[sample_idx:sample_idx+1]
                        
                        # 获取预测
                        pred_multilabel = qbc.committee_models[0].predict(X_sample)[0]
                        binary = ''.join(map(str, pred_multilabel.astype(int)))
                        pred_int = binary_to_int(binary)
                        
                        committee_predictions[int(idx)] = pred_int
                    except Exception as e:
                        print(f"预测样本 {idx} 时出错: {e}")
                
                # 步骤3: 打开标注界面
                print("\n[步骤3] 打开标注界面，请标注选定的样本")
                
                # 创建标注界面实例
                annotation_interface = AnnotationInterface(
                    reviews_df=all_reviews_df,
                    model_predictions=committee_predictions,
                    existing_annotations=existing_annotations,
                    priority_indices=priority_rows
                )
                
                # 显示标注提示信息
                print("请在标注界面中标注评论，优先标注高价值样本")
                print("优先样本会用橙色背景高亮显示")
                print("完成后关闭窗口或点击'完成标注'按钮")
                
                # 运行标注界面
                new_annotations = annotation_interface.run()
                
                # 检查标注结果
                print(f"标注会话结束，共标注了 {annotation_interface.session_count} 条评论")
                if annotation_interface.priority_annotated > 0:
                    print(f"其中标注了 {annotation_interface.priority_annotated} 条优先样本")
                
                # 检查标注文件是否存在，不存在则创建
                if not os.path.exists(annotations_path) and new_annotations:
                    # 手动创建标注文件
                    data = {
                        "annotations": {str(k): v for k, v in new_annotations.items()},
                        "timestamp": datetime.now().isoformat(),
                        "completed": False,
                        "session_info": {
                            "count": annotation_interface.session_count,
                            "priority_annotated": annotation_interface.priority_annotated,
                            "normal_annotated": annotation_interface.normal_annotated
                        }
                    }
                    with open(annotations_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"已创建标注文件: {annotations_path}")
                
                # 步骤4: 更新标注数据
                print("\n[步骤4] 更新标注数据")
                try:
                    # 重新加载标注
                    with open(annotations_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        existing_annotations = {int(k): int(v) for k, v in data['annotations'].items()}
                        annotation_completed = data.get('completed', False)
                    
                    # 更新已标注和未标注索引
                    labeled_orig_indices = np.array([idx for idx in existing_annotations.keys() if idx in index_map])
                    unlabeled_orig_indices = np.array([idx for idx in original_indices if idx not in existing_annotations and idx in index_map])
                    
                    print(f"更新后 - 已标注: {len(labeled_orig_indices)}, 未标注: {len(unlabeled_orig_indices)}")
                    
                    # 更新已标注特征和标签
                    labeled_embed_indices = np.array([index_map[idx] for idx in labeled_orig_indices])
                    X_labeled = embeddings[labeled_embed_indices]
                    y_labeled = np.array([existing_annotations[idx] for idx in labeled_orig_indices])
                    
                    # 显示更新的标签分布
                    print("\n更新后的标签分布:")
                    label_counts = Counter([int_to_binary(label) for label in y_labeled])
                    for binary, count in sorted(label_counts.items()):
                        print(f"{binary} ({binary_to_description(binary)}): {count}条")
                    
                    # 检查是否结束迭代
                    if annotation_completed:
                        print("标注已标记为完成，结束主动学习循环")
                        break
                        
                except Exception as e:
                    print(f"更新标注数据出错: {e}")
                    import traceback
                    traceback.print_exc()
                    print("将使用原始标注继续")
                
            except Exception as e:
                print(f"选择样本出错: {e}")
                import traceback
                traceback.print_exc()
                continue  # 继续下一轮迭代
    
    # ==== 3. 训练最终模型 ====
    print("\n[训练最终模型]")
    
    # 确保有标注数据
    if len(labeled_orig_indices) == 0:
        print("错误: 没有标注数据，无法训练模型")
        return
    
    # 使用所有标注训练最终模型
    try:
        final_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 转换为多标签格式
        y_multilabel = np.zeros((len(y_labeled), 3), dtype=int)
        for i, label in enumerate(y_labeled):
            binary = int_to_binary(label)
            for j in range(3):
                y_multilabel[i, j] = int(binary[j])
        
        # 创建多输出分类器
        final_model = MultiOutputClassifier(final_model)
        final_model.fit(X_labeled, y_multilabel)
        
        # 保存最终模型
        final_path = model_manager.save_model(final_model, "final")
        print(f"最终模型已保存到: {final_path}")
        
        # 评估最终模型
        if len(labeled_orig_indices) >= 10:
            X_train, X_val, y_train, y_val, _, _ = custom_train_test_split(
                X_labeled, y_labeled, labeled_orig_indices, test_size=0.2, random_state=42
            )
            evaluate_model(final_model, X_val, y_val)
        
        # 预测所有评论
        print_prediction_stats(final_model, all_reviews_df, embeddings, 
                             labeled_orig_indices, unlabeled_orig_indices, 
                             y_labeled, index_map)
                             
        print("\n程序执行完毕！")
        print("训练好的模型保存在 model_ckpt/ 目录下")
        print("您可以使用 model_final.pkl 进行新评论的预测")
    except Exception as e:
        print(f"训练最终模型出错: {e}")
        import traceback
        traceback.print_exc()


def find_latest_iteration(model_dir):
    """查找最新的迭代次数"""
    iter_files = [f for f in os.listdir(model_dir) if f.startswith('model_iter_') and f.endswith('.pkl')]
    if not iter_files:
        return 0
        
    iter_nums = []
    for f in iter_files:
        try:
            iter_num = int(f.split('iter_')[1].split('.')[0])
            iter_nums.append(iter_num)
        except:
            continue
            
    return max(iter_nums) if iter_nums else 0


def print_prediction_stats(model, all_reviews_df, embeddings, labeled_indices, unlabeled_indices, y_labeled, index_map=None):
    """打印预测统计信息"""
    print("\n[预测结果统计]")
    
    # 计算标注情况
    print(f"已标注样本: {len(labeled_indices)} 条")
    print(f"未标注样本: {len(unlabeled_indices)} 条")
    
    # 标签分布
    label_counts = Counter([int_to_binary(label) for label in y_labeled])
    print("\n已知标签分布:")
    for binary, count in sorted(label_counts.items()):
        print(f"{binary} ({binary_to_description(binary)}): {count} 条 ({count/len(y_labeled)*100:.1f}%)")
    
    # 预测未标注样本
    if len(unlabeled_indices) > 0:
        try:
            # 准备嵌入向量
            if index_map is not None:
                # 使用索引映射
                unlabeled_embed_indices = [index_map[idx] for idx in unlabeled_indices if idx in index_map]
                if not unlabeled_embed_indices:
                    print("警告: 无法找到有效的未标注样本嵌入")
                    return
                    
                X_unlabeled = embeddings[unlabeled_embed_indices]
            else:
                # 直接使用索引
                X_unlabeled = embeddings[unlabeled_indices]
            
            # 预测
            pred_multilabel = model.predict(X_unlabeled)
            
            # 转换为整数标签
            pred_labels = np.zeros(len(X_unlabeled), dtype=int)
            for i, pred_row in enumerate(pred_multilabel):
                try:
                    binary = ''.join(map(str, pred_row.astype(int)))
                    pred_labels[i] = binary_to_int(binary)
                except Exception as e:
                    print(f"处理预测行 {i} 出错: {e}")
                    pred_labels[i] = 0
            
            # 打印预测分布
            pred_counts = Counter([int_to_binary(label) for label in pred_labels])
            print("\n预测标签分布:")
            for binary, count in sorted(pred_counts.items()):
                print(f"标签 '{binary}' ({binary_to_description(binary)}): {count} 条评论 ({count / len(pred_labels) * 100:.2f}%)")
        except Exception as e:
            print(f"预测未标注样本出错: {e}")
            import traceback
            traceback.print_exc()


# 辅助函数：评估模型
def evaluate_model(model, X_val, y_val):
    """评估模型性能"""
    if len(X_val) == 0 or len(y_val) == 0:
        print("警告: 没有验证数据，无法评估模型")
        return {'accuracy': 0, 'macro_f1': 0, 'weighted_f1': 0}
        
    try:
        # 预测
        y_pred_multilabel = model.predict(X_val)
        
        # 将多标签预测转换为整数
        y_pred = np.zeros(len(X_val), dtype=int)
        for i, pred_row in enumerate(y_pred_multilabel):
            binary_str = ''.join(map(str, pred_row.astype(int)))
            y_pred[i] = binary_to_int(binary_str)
        
        # 计算指标
        accuracy = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        # 输出结果
        print(f"\n模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"宏平均F1: {macro_f1:.4f}")
        print(f"加权F1: {weighted_f1:.4f}")
        
        # 分类报告
        report = classification_report(y_val, y_pred, zero_division=0)
        print("分类报告:")
        print(report)
        
        return {'accuracy': accuracy, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}
        
    except Exception as e:
        print(f"评估模型出错: {e}")
        import traceback
        traceback.print_exc()
        return {'accuracy': 0, 'macro_f1': 0, 'weighted_f1': 0}


# 辅助函数：绘制学习曲线
def plot_learning_curves(metrics_history):
    """输出学习曲线(仅控制台表格)"""
    try:
        # 表格输出
        print("\n主动学习性能指标:")
        print(f"{'迭代':<8}{'样本数':<8}{'准确率':<8}{'宏平均F1':<8}{'加权F1':<8}")
        print("-" * 40)
        
        for i in range(len(metrics_history['iteration'])):
            iter_num = metrics_history['iteration'][i]
            samples = metrics_history['samples_count'][i]
            acc = metrics_history['accuracy'][i]
            mf1 = metrics_history['macro_f1'][i]
            wf1 = metrics_history['weighted_f1'][i]
            
            print(f"{iter_num:<8}{samples:<8}{acc:.4f}  {mf1:.4f}  {wf1:.4f}")
            
    except Exception as e:
        print(f"输出学习曲线出错: {e}")


# 辅助函数：保存预测结果
def save_predictions(model, all_reviews_df, embeddings, labeled_indices, unlabeled_indices, y_labeled, annotations):
    """打印模型对所有评论的预测统计结果"""
    print("\n=== 预测结果统计 ===")
    
    # 计算已标注样本数量
    print(f"已标注样本: {len(labeled_indices)} 条")
    
    # 计算未标注样本数量
    print(f"未标注样本: {len(unlabeled_indices)} 条")
    
    # 统计标签分布
    label_counts = Counter([int_to_binary(label) for label in y_labeled])
    print("\n已知标签分布:")
    for binary, count in sorted(label_counts.items()):
        print(f"标签 '{binary}' ({binary_to_description(binary)}): {count} 条评论 ({count / len(y_labeled) * 100:.2f}%)")
    
    # 如果有未标注样本，预测它们的标签分布
    if len(unlabeled_indices) > 0:
        try:
            # 对未标注样本进行预测
            X_unlabeled = embeddings[unlabeled_indices]
            pred_multilabel = model.predict(X_unlabeled)
            
            # 将多标签预测转换为整数
            pred_labels = np.zeros(len(X_unlabeled), dtype=int)
            for i, pred_row in enumerate(pred_multilabel):
                if isinstance(pred_row, np.ndarray):
                    binary_str = ''.join(map(str, pred_row))
                    pred_labels[i] = binary_to_int(binary_str)
                else:
                    # 处理非数组类型的预测结果
                    pred_labels[i] = int(pred_row) if pred_row is not None else 0
            
            # 打印未标注样本的预测标签分布
            pred_label_counts = Counter([int_to_binary(label) for label in pred_labels])
            print("\n预测标签分布:")
            for binary, count in sorted(pred_label_counts.items()):
                print(f"标签 '{binary}' ({binary_to_description(binary)}): {count} 条评论 ({count / len(pred_labels) * 100:.2f}%)")
        except Exception as e:
            print(f"预测未标注样本时出错: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()