# -*- coding: utf-8 -*-
# @File    :   stats.py
# @Time    :   2023/04/03 14:31:17
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def draw_hist_percentage(numbers, nbins=50, title=None, save_file=None):
    """直方图 分成nbins个区间, y轴是每个区间数量的百分比

    Args:
        numbers: list 一组数字
        nbins: Int. Defaults to 50.
        title: Str. 图片的标题
        save_file: Str. filepath for saving the plot. Defaults to None.
    """
    fig, ax = plt.subplots(1, 1)
    # Plot the distribution using a histogram
    ax.hist(numbers, bins=nbins)
    ax.yaxis.set_major_formatter(PercentFormatter(len(numbers)))
    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Percentage')
    title = title if title is not None else 'Distribution of numbers'
    plt.title(title)

    # Save the plot as a PNG file
    if save_file is not None:
        plt.savefig('logicv2_distr.png')


def test():
    numbers = np.random.randn(1000)
    draw_hist_percentage(numbers, nbins=50, title='Distribution of numbers', save_file=None)
    plt.show()


if __name__ == '__main__':
    test()