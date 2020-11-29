# explainable_GCN


## 环境配置
```
conda create -n gcn-py36 python=3.6.10
conda activate gcn-py36
pip install -r requirements.txt
```
如果环境配置失败，可以下载我上传到google driver的环境https://drive.google.com/file/d/1W-Ti1ycIhS7zxcvApMgIwCSHyjGkFGbg/view?usp=sharing，找我加一下权限

## 模型训练

```
# 数据构造：

python build_graoh.py mr

python train_gcn.py \
    -lr 0.01 \
    -dataset_name mr \
    -dropout 0.5 \
    -epochs 100 \
    -gcn_layers 1 \
    -embed_dim 200 \
    -embed_fintune True \
    -hidden_size 256 \
    -node_size 29426 \
    -random_seed 42 \
    -device 2 \
    -model_name mr_id \
    -save_best_model False \
    -output_size 2
```

### TextGCN

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


### Add label 

#### MR

|  模型  |  参数 |  acc |
|  ----  | ----  | ---- |
| TextGCN | lr=0.01, hidden=512, embed=512, layer=2 | 0.764 |
| TextGCN-Label | lr=0.01, hidden=512, embed=512, layer=2 | 0.760 |
| TextHGCN | lr=0.001, hidden=256, embed=256, layer=3 | 0.751 |
| TextHGCN-Label| lr=0.001, hidden=256, embed=256, layer=2 | 0.734 |

#### R8

|  模型  |  参数 |  acc |
|  ----  | ----  | ---- |
| TextGCN | lr=0.001, hidden=512, embed=512, layer=2 | 0.9707 |
| TextGCN-Label | lr=0.001, hidden=512, embed=512, layer=2 | 0.9703 |
| TextHGCN | lr=0.001, hidden=256, embed=256, layer=3 | 0.9643 |
| TextHGCN-Label| lr=0.001, hidden=256, embed=256, layer=2 | 0.9607 |

## 模型可解释性

```
explain_gcn.ipynb
```

算法参考[Captum Algorithm](https://captum.ai/docs/algorithms)



