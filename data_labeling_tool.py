import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import json
from datetime import datetime

class LabelingTool:
    def __init__(self, master, data_path, output_path=None):
        """
        初始化标注工具
        
        参数:
        - master: tkinter主窗口
        - data_path: 数据文件路径
        - output_path: 输出文件路径（默认与数据文件同目录）
        """
        self.master = master
        self.master.title("游戏评论多标签标注工具")
        self.master.geometry("800x600")
        
        # 加载数据
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        
        # 确保有review_text和original_index列
        if 'review_text' not in self.df.columns or 'original_index' not in self.df.columns:
            raise ValueError("数据文件必须包含review_text和original_index列")
        
        # 如果已有label列，加载现有标注
        self.existing_labels = {}
        if 'label' in self.df.columns:
            for idx, label in zip(self.df['original_index'], self.df['label']):
                if not pd.isna(label):  # 跳过空值
                    self.existing_labels[idx] = int(label)
        
        # 输出路径
        self.output_path = output_path if output_path else os.path.join(
            os.path.dirname(data_path), 
            "labeled_" + os.path.basename(data_path)
        )
        
        # 创建JSON保存路径（用于中间保存）
        self.json_save_path = os.path.join(
            os.path.dirname(data_path),
            "annotations.json"
        )
        
        # 标注状态
        self.current_index = 0
        self.total_reviews = len(self.df)
        self.annotations = self.existing_labels.copy()
        
        # 创建UI
        self.create_widgets()
        
        # 加载第一条评论
        self.load_review()
        
    def create_widgets(self):
        """创建UI组件"""
        # 主框架
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 评论文本区域
        review_frame = ttk.LabelFrame(main_frame, text="评论内容", padding="5")
        review_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.review_text = tk.Text(review_frame, wrap=tk.WORD, height=10)
        self.review_text.pack(fill=tk.BOTH, expand=True)
        self.review_text.config(state=tk.DISABLED)
        
        # 标签选择区域
        label_frame = ttk.LabelFrame(main_frame, text="选择标签（可多选）", padding="5")
        label_frame.pack(fill=tk.X, pady=5)
        
        # 标签说明
        label_info = ttk.Label(label_frame, text="标签说明：\n"
                               "1. 游戏环境：游戏社区、玩家体验、社交互动等\n"
                               "2. 游戏优化：性能、卡顿、崩溃等技术问题\n"
                               "3. 游戏性：玩法、难度、乐趣、游戏设计等")
        label_info.pack(anchor=tk.W, pady=5)
        
        # 复选框框架
        checkbox_frame = ttk.Frame(label_frame)
        checkbox_frame.pack(fill=tk.X, pady=5)
        
        # 复选框变量
        self.var_environment = tk.BooleanVar(value=False)
        self.var_optimization = tk.BooleanVar(value=False)
        self.var_gameplay = tk.BooleanVar(value=False)
        
        # 创建复选框
        cb_environment = ttk.Checkbutton(checkbox_frame, text="游戏环境", variable=self.var_environment)
        cb_environment.grid(row=0, column=0, padx=10, sticky=tk.W)
        
        cb_optimization = ttk.Checkbutton(checkbox_frame, text="游戏优化", variable=self.var_optimization)
        cb_optimization.grid(row=0, column=1, padx=10, sticky=tk.W)
        
        cb_gameplay = ttk.Checkbutton(checkbox_frame, text="游戏性", variable=self.var_gameplay)
        cb_gameplay.grid(row=0, column=2, padx=10, sticky=tk.W)
        
        # 操作按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.prev_button = ttk.Button(button_frame, text="上一条", command=self.previous_review)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(button_frame, text="保存并下一条", command=self.next_review)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        self.skip_button = ttk.Button(button_frame, text="跳过", command=self.skip_review)
        self.skip_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="保存所有标注", command=self.save_annotations)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        self.complete_button = ttk.Button(button_frame, text="完成标注", command=self.complete_labeling)
        self.complete_button.pack(side=tk.RIGHT, padx=5)
        
        # 进度显示
        self.progress_var = tk.StringVar(value="进度: 0/0")
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        progress_label.pack(anchor=tk.W, pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(anchor=tk.W, pady=5)
    
    def load_review(self):
        """加载当前索引的评论"""
        if 0 <= self.current_index < self.total_reviews:
            review = self.df.iloc[self.current_index]
            original_index = review['original_index']
            
            # 更新评论文本
            self.review_text.config(state=tk.NORMAL)
            self.review_text.delete(1.0, tk.END)
            self.review_text.insert(tk.END, review['review_text'])
            self.review_text.config(state=tk.DISABLED)
            
            # 更新进度显示
            self.progress_var.set(f"进度: {self.current_index + 1}/{self.total_reviews} (索引: {original_index})")
            
            # 加载现有标注（如果有）
            if original_index in self.annotations:
                label = self.annotations[original_index]
                binary = self.int_to_binary(label)
                self.var_environment.set(binary[0] == '1')
                self.var_optimization.set(binary[1] == '1')
                self.var_gameplay.set(binary[2] == '1')
            else:
                # 重置复选框
                self.var_environment.set(False)
                self.var_optimization.set(False)
                self.var_gameplay.set(False)
            
            self.status_var.set(f"已加载评论 {original_index}")
        else:
            messagebox.showinfo("标注完成", "所有评论已标注完成！")
    
    def get_current_label(self):
        """获取当前选择的标签组合对应的整数值"""
        environment = 1 if self.var_environment.get() else 0
        optimization = 1 if self.var_optimization.get() else 0
        gameplay = 1 if self.var_gameplay.get() else 0
        
        binary = f"{environment}{optimization}{gameplay}"
        return self.binary_to_int(binary)
    
    def binary_to_int(self, binary_str):
        """将二进制字符串转换为整数"""
        return int(binary_str, 2)
    
    def int_to_binary(self, num):
        """将整数转换为3位二进制字符串"""
        binary = bin(num)[2:]  # 去掉'0b'前缀
        return binary.zfill(3)  # 补足3位
    
    def next_review(self):
        """保存当前标注并前进到下一条评论"""
        if 0 <= self.current_index < self.total_reviews:
            # 获取当前评论的original_index
            original_index = self.df.iloc[self.current_index]['original_index']
            
            # 保存标注
            label = self.get_current_label()
            self.annotations[original_index] = label
            
            # 更新状态
            binary = self.int_to_binary(label)
            self.status_var.set(f"已保存评论 {original_index} 的标注: {binary}")
            
            # 前进到下一条
            self.current_index += 1
            if self.current_index < self.total_reviews:
                self.load_review()
            else:
                messagebox.showinfo("标注完成", "所有评论已标注完成！")
                self.save_annotations()
    
    def previous_review(self):
        """返回上一条评论"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_review()
    
    def skip_review(self):
        """跳过当前评论"""
        self.current_index += 1
        if self.current_index < self.total_reviews:
            self.load_review()
        else:
            messagebox.showinfo("标注完成", "所有评论已标注完成！")
    
    def save_annotations(self):
        """保存标注到CSV和JSON文件"""
        # 保存到CSV
        result_df = self.df.copy()
        result_df['label'] = None  # 创建或重置label列
        
        # 填充标注
        for idx, label in self.annotations.items():
            mask = result_df['original_index'] == idx
            result_df.loc[mask, 'label'] = label
        
        # 保存CSV
        result_df.to_csv(self.output_path, index=False)
        
        # 保存JSON（包含时间戳和完成状态）
        json_data = {
            "annotations": {str(k): v for k, v in self.annotations.items()},
            "timestamp": datetime.now().isoformat(),
            "completed": False  # 默认未完成
        }
        
        with open(self.json_save_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        self.status_var.set(f"已保存 {len(self.annotations)} 条标注")
        messagebox.showinfo("保存成功", f"已保存 {len(self.annotations)} 条标注到:\n{self.output_path}\n{self.json_save_path}")
    
    def complete_labeling(self):
        """标记标注为已完成并保存"""
        if messagebox.askyesno("确认完成", "确定要标记标注为已完成吗？这将允许程序跳过主动学习阶段直接训练最终模型。"):
            # 保存到CSV
            result_df = self.df.copy()
            result_df['label'] = None  # 创建或重置label列
            
            # 填充标注
            for idx, label in self.annotations.items():
                mask = result_df['original_index'] == idx
                result_df.loc[mask, 'label'] = label
            
            # 保存CSV
            result_df.to_csv(self.output_path, index=False)
            
            # 保存JSON（包含时间戳和完成状态）
            json_data = {
                "annotations": {str(k): v for k, v in self.annotations.items()},
                "timestamp": datetime.now().isoformat(),
                "completed": True  # 标记为已完成
            }
            
            with open(self.json_save_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            self.status_var.set(f"已标记标注为完成并保存 {len(self.annotations)} 条标注")
            messagebox.showinfo("保存成功", "已标记标注为完成，程序将跳过主动学习阶段直接训练最终模型。")


def main():
    # 创建主窗口
    root = tk.Tk()
    
    # 默认数据路径
    data_path = "data/initial_seeds.csv"
    
    # 如果数据文件不存在，提示选择
    if not os.path.exists(data_path):
        from tkinter import filedialog
        messagebox.showinfo("选择数据文件", "请选择要标注的CSV数据文件")
        data_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("CSV文件", "*.csv")]
        )
        if not data_path:
            messagebox.showerror("错误", "未选择数据文件，程序将退出")
            return
    
    # 创建标注工具
    labeling_tool = LabelingTool(root, data_path)
    
    # 运行主循环
    root.mainloop()


if __name__ == "__main__":
    main() 