
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties  # 字体管理器
import random


def plot_confusion_matrix(true_label, pred_label, save_path=None):
    # 设置汉字格式
    font = FontProperties(size=12)
    
    sns.set()
    f, ax = plt.subplots()
    C2 = confusion_matrix(true_label, pred_label, labels=[0, 1])
    sns.heatmap(C2, annot=True, ax=ax)  # 画热力图
    
    plt.title("混淆矩阵", fontproperties=font)  # 设置标题
    plt.xlabel('预测值', fontproperties=font)  # 设置x轴标题
    plt.ylabel('真实值', fontproperties=font)  # 设置y轴标签
    
    # 保存绘图
    if save_path is not None:
        plt.savefig(save_path)
    
    # 绘图
    plt.show()
    

def multiple_line_plot( labels, value_lists, save_path=None):
    """绘制多条折线图

    Args:
        labels: str list. 每条折线的标签
        value_lists: list of list. 每条折线的值
        save_path: 图片保存路径. Defaults to None.
    """
    ls_options = ['-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted']
    for label, value_list in zip(labels, value_lists):
        plt.plot(value_list, marker="o", label=label, ls="--")
        # plt.plot(value_list, marker="o", label=label)
  
    
    plt.xlabel("month")
    plt.ylabel("income")
    plt.title("income of each Company")
    
    plt.grid(color='lightpink', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()
    
    if save_path is not None:
        plt.savefig(save_path)


if __name__ == "__main__":
    def test_confusion_matrix():
        true_label = [random.randint(0, 1) for i in range(0, 100)]
        pred_label = [random.randint(0, 1) for i in range(0, 100)]
        plot_confusion_matrix(true_label, pred_label)
                
    def test_multiple_line_plot():
        label_name = ["A", "B", "C", "D"]
        metric_name = "income"
        lists = [[random.randint(0, 100) for i in range(0, 12)] for i in range(0, 4)]
        multiple_line_plot(label_name, lists)
        
    
    test_multiple_line_plot()