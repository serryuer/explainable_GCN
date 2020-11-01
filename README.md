# explainable_GCN


## 模型训练

1. 输入id，经过embedding得到nodesize维的向量

|  参数   |  acc |
|  ----  | ----  |
| lr=0.01, hidden=200 | 0.748 |
| lr=0.01, hidden=300 | 0.753 |
| lr=0.01, hidden=512 | 0.754 |
| lr=0.01, hidden=1024 | 0.754 |
| lr=0.001, hidden=200 | 0.752 |
| lr=0.001, hidden=300 | 0.753 |
| lr=0.001, hidden=512 | 0.754 |
| lr=0.001, hidden=1024 | 0.752 |

2. 调整embedding输出向量维度，embedding权重可学习

|  参数   |  acc |
|  ----  | ----  |
| lr=0.01, hidden=512, embed=node_size| 0.754 |
| lr=0.01, hidden=512, embed=10000 | 0.759 |
| lr=0.01, hidden=512, embed=5000 | 0.748 |
| lr=0.01, hidden=512, embed=1024 | 0.751 |
| lr=0.01, hidden=512, embed=512 | 0.762 |
| lr=0.01, hidden=512, embed=300 | 0.757 |
| lr=0.01, hidden=512, embed=200 | 0.765 |

3. 再次调整hidden_dim

|  参数   |  acc |
|  ----  | ----  |
| lr=0.01, hidden=1024, embed=200| 0.753 |
| lr=0.01, hidden=512, embed=200 | 0.765 |
| lr=0.01, hidden=256, embed=200 | 0.764 |

4. 调整layer层数
TODO:

|  参数   |  acc |
|  ----  | ----  |
| lr=0.01, hidden=256, embed=200, layer=1 | 0.764 |
| lr=0.01, hidden=256, embed=200, layer=2 | 0.753 |
| lr=0.01, hidden=256, embed=200, layer=3 | 0.753 |
| lr=0.01, hidden=256, embed=200, layer=4 | 0.753 |


## 模型可解释性

算法参考[Captum Algorithm](https://captum.ai/docs/algorithms)

