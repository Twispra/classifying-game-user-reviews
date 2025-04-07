"""
极简游戏评论标注工具

这个脚本实现了一个极简版的游戏评论标注工具，
只包含核心功能，便于测试和调试。
"""

import os
import json
import pandas as pd
import numpy as np
import random
from pathlib import Path

# 定义类别
CLASSES = ["游戏环境", "游戏优化", "游戏性"]
NUM_CLASSES = len(CLASSES)

# 数据路径
DATA_DIR = Path("data")
ANNOTATIONS_FILE = DATA_DIR / "annotations.json"
PREPROCESSED_FILE = DATA_DIR / "preprocessed_reviews.csv"

# 创建必要的目录
DATA_DIR.mkdir(exist_ok=True)

def load_preprocessed_data(file_path=PREPROCESSED_FILE):
    """加载预处理后的评论"""
    try:
        print(f"加载评论数据: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告: 找不到评论数据文件 {file_path}")
            # 创建示例数据
            sample_data = pd.DataFrame({
                'original_index': list(range(10)),
                'review_text': [f"这是示例评论 {i}" for i in range(10)]
            })
            sample_data.to_csv(file_path, index=False)
            print(f"已创建示例数据文件：{file_path}")
            return sample_data
            
        # 加载数据
        df = pd.read_csv(file_path)
        
        # 确保有original_index列
        if "original_index" not in df.columns:
            print("警告: 数据中没有original_index列，添加该列")
            df["original_index"] = df.index
            
        print(f"已加载 {len(df)} 条评论")
        return df
    except Exception as e:
        print(f"加载数据出错: {e}")
        # 返回空数据框
        return pd.DataFrame({'original_index': [], 'review_text': []})

def load_annotations(file_path=ANNOTATIONS_FILE):
    """加载已有标注"""
    try:
        if not os.path.exists(file_path):
            print(f"未找到标注文件: {file_path}，将创建新文件")
            return {}
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            annotations = {int(k): int(v) for k, v in data.get('annotations', {}).items()}
        
        print(f"已加载 {len(annotations)} 条标注")
        return annotations
    except Exception as e:
        print(f"加载标注出错: {e}")
        return {}

def save_annotations(annotations, file_path=ANNOTATIONS_FILE):
    """保存标注到文件"""
    try:
        data = {
            "annotations": {str(k): v for k, v in annotations.items()},
            "timestamp": pd.Timestamp.now().isoformat(),
            "completed": False
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"已保存 {len(annotations)} 条标注到 {file_path}")
        return True
    except Exception as e:
        print(f"保存标注出错: {e}")
        return False

def binary_to_description(binary_str):
    """将二进制标签转换为文字描述"""
    components = []
    for i, bit in enumerate(binary_str):
        if bit == '1' and i < len(CLASSES):
            components.append(CLASSES[i])
    
    if not components:
        return "无意义/情绪化评论"
    
    return "+".join(components)

def int_to_binary(label_int, length=NUM_CLASSES):
    """将整数标签转换为二进制字符串"""
    try:
        return format(int(label_int), f'0{length}b')
    except:
        return "0" * length

def binary_to_int(binary_str):
    """将二进制字符串转换为整数"""
    try:
        return int(binary_str, 2)
    except:
        return 0

def console_annotate(reviews_df, existing_annotations=None, n_samples=5):
    """简单的控制台标注界面"""
    if existing_annotations is None:
        existing_annotations = {}
    
    annotations = existing_annotations.copy()
    
    # 获取所有未标注的索引
    all_original_indices = set(reviews_df['original_index'].values)
    labeled_indices = set(annotations.keys())
    unlabeled_indices = all_original_indices - labeled_indices
    
    # 如果全部标注完成，返回
    if not unlabeled_indices:
        print("所有样本已标注完成！")
        return annotations
    
    # 随机选择一些未标注的样本
    sample_indices = random.sample(list(unlabeled_indices), min(n_samples, len(unlabeled_indices)))
    
    print(f"\n将标注 {len(sample_indices)} 个样本:")
    
    # 标注每个样本
    for idx in sample_indices:
        row = reviews_df[reviews_df['original_index'] == idx].iloc[0]
        review_text = row['review_text']
        
        print("\n" + "="*50)
        print(f"评论ID: {idx}")
        print("-"*50)
        
        # 显示评论内容 (最多500个字符)
        if len(review_text) > 500:
            print(f"{review_text[:500]}... (内容过长已截断)")
        else:
            print(review_text)
        
        print("-"*50)
        print("请选择标签 (可多选):")
        
        for i, class_name in enumerate(CLASSES):
            print(f"{i+1}. {class_name}")
        
        print("0. 无意义/情绪化评论")
        print("-"*50)
        
        # 获取用户输入
        while True:
            try:
                choice = input("请输入标签编号 (多选请用逗号分隔，如'1,3'，输入'q'退出): ")
                
                # 检查是否退出
                if choice.lower() == 'q':
                    print("标注已取消")
                    return annotations
                
                # 解析输入
                if choice == '0':
                    selected = []
                else:
                    selected = [int(x.strip())-1 for x in choice.split(',') if x.strip()]
                
                # 验证输入
                if all(0 <= s < NUM_CLASSES for s in selected):
                    break
                else:
                    print(f"输入无效，请输入0-{NUM_CLASSES}之间的数字")
            except ValueError:
                print("格式错误，请重新输入")
        
        # 构建二进制标签
        binary = ['0'] * NUM_CLASSES
        for s in selected:
            binary[s] = '1'
        
        binary_str = ''.join(binary)
        label_int = binary_to_int(binary_str)
        
        # 保存标注
        annotations[idx] = label_int
        
        # 显示结果
        print(f"已标注为: {binary_str} ({binary_to_description(binary_str)})")
    
    # 保存标注
    save_annotations(annotations)
    
    print("\n标注已完成并保存")
    return annotations

def main():
    """主函数"""
    print("="*50)
    print("极简游戏评论标注工具")
    print("="*50)
    
    # 加载数据
    reviews_df = load_preprocessed_data()
    if len(reviews_df) == 0:
        print("错误: 无法加载评论数据")
        return
    
    # 加载现有标注
    annotations = load_annotations()
    
    print("\n现有标注统计:")
    if annotations:
        # 标签分布
        labels = [int_to_binary(label) for label in annotations.values()]
        counts = {}
        for binary in labels:
            desc = binary_to_description(binary)
            counts[desc] = counts.get(desc, 0) + 1
        
        for desc, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {desc}: {count}条")
    else:
        print("  - 尚无标注")
    
    # 开始标注
    print("\n开始标注过程...")
    annotations = console_annotate(reviews_df, annotations)
    
    # 保存最终标注
    save_annotations(annotations)
    
    print("\n程序结束")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    except Exception as e:
        import traceback
        print(f"\n程序出错: {e}")
        traceback.print_exc() 