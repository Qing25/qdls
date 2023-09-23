

# 20230922 更新

```shell
.
├── README.md
├── data                    # 存放实验数据
├── notebooks               # 统计数据、绘图
├── preliminary             # 初期的验证实验
└── src                     # 代码
    ├── configs             # 实验配置文件
    │   └── example.yaml
    ├── data_utils          # 不同的数据集
    │   ├── __init__.py
    │   └── common.py
    ├── frame.py            # 训练、预测等函数框架
    ├── main.py             # 入口文件
    ├── models              # 定义的各种模型
    │   └── __init__.py
    ├── pl_callbacks.py     # 训练过程中用到的callbacks
    ├── results             # 存放预测结果的目录
    ├── scripts             # 批量实验的脚本
    └── utils.py            # 通用函数
```