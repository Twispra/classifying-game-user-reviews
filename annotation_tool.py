import os
import pandas as pd
import json
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import time

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
    # 标签映射
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

# ===== 标注界面类 =====
class AnnotationInterface:
    """评论标注界面"""
    def __init__(self, reviews_df, existing_annotations=None, priority_indices=None):
        """初始化标注界面"""
        self.reviews_df = reviews_df
        self.reviews = reviews_df["review_text"].tolist()
        self.annotations = {} if existing_annotations is None else existing_annotations.copy()
        self.priority_indices = [] if priority_indices is None else priority_indices.copy()
        self.save_path = "data/annotations.json"
        self.current_index = None
        
        # 添加优先样本队列，用于追踪未标注的优先样本
        self.pending_priority_indices = self.priority_indices.copy()
        
        # 会话信息
        self.session_start = time.time()
        self.session_count = 0
        
        # 当前是否在优先样本模式
        self.priority_mode = True
        
        # GUI相关变量
        self.root = None
        self.var_environment = None
        self.var_optimization = None
        self.var_gameplay = None
        self.review_text = None
        self.status_text = None
        self.mode_indicator = None
        
        print(f"初始化标注界面完成，共有{len(reviews_df)}条评论，{len(priority_indices)}条优先样本")
    
    def run(self):
        """运行标注界面"""
        # 检查数据有效性
        if not self.reviews:
            print("错误: 没有评论数据可标注")
            return self.annotations
            
        try:
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
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"运行标注界面出错: {e}")
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
        remaining_total = total - annotated
        
        # 计算优先样本数量
        remaining_priority = len(self.pending_priority_indices)
        
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
        messagebox.showinfo("提示", "暂不支持浏览历史标注")
    
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
                    "count": self.session_count
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
        if not messagebox.askyesno("确认", "确定要完成标注吗？"):
            return
            
        # 保存当前标注
        self.save_current_annotation()
        
        try:
            # 准备数据，标记为已完成
            data = {
                "annotations": {str(k): v for k, v in self.annotations.items()},
                "timestamp": datetime.now().isoformat(),
                "completed": True
            }
            
            # 保存数据
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            messagebox.showinfo("完成标注", "标注已标记为完成")
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

# ===== 示例使用 =====
def main():
    try:
        # 创建必要目录
        os.makedirs("data", exist_ok=True)
        
        # 检查数据文件是否存在
        csv_path = "data/preprocessed_reviews.csv"
        if not os.path.exists(csv_path):
            print(f"错误: 找不到评论数据文件 {csv_path}")
            sample_data = pd.DataFrame({
                'original_index': list(range(10)),
                'review_text': [f"这是示例评论 {i}" for i in range(10)]
            })
            sample_data.to_csv(csv_path, index=False)
            print(f"已创建示例数据文件：{csv_path}")
        
        # 加载评论数据
        reviews_df = pd.read_csv(csv_path)
        print(f"已加载 {len(reviews_df)} 条评论数据")
        
        # 加载现有标注（如果有）
        existing_annotations = {}
        annotations_path = "data/annotations.json"
        if os.path.exists(annotations_path):
            try:
                with open(annotations_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing_annotations = {int(k): int(v) for k, v in data['annotations'].items()}
                print(f"已加载 {len(existing_annotations)} 条现有标注")
            except Exception as e:
                print(f"加载标注失败: {e}")
        
        # 随机选择优先样本（实际项目中应由模型选择）
        import random
        priority_indices = random.sample(range(len(reviews_df)), min(5, len(reviews_df)))
        print(f"已选择 {len(priority_indices)} 个优先样本: {priority_indices}")
        
        # 创建并运行标注界面
        annotation_interface = AnnotationInterface(
            reviews_df=reviews_df,
            existing_annotations=existing_annotations,
            priority_indices=priority_indices
        )
        
        print("正在启动标注界面...")
        new_annotations = annotation_interface.run()
        print(f"标注完成，共标注 {len(new_annotations)} 条评论")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main() 