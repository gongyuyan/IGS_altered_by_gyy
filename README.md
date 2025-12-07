# 本科毕业论文

是对以下论文的复现: 

> [Interpretable Sparsification of Brain Graphs: Better Practices and Effective Designs for Graph Neural Networks](https://arxiv.org/abs/2306.14375)

## 原项目地址

> [GaotangLi/IGS: Code for "Interpretable Sparsification of Brain Graphs: Better Practices and Effective Designs for Graph Neural Networks"](https://github.com/GaotangLi/IGS)


## 环境
在kaggle中运行的
```
!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
!pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
!pip install torch-geometric
```

## 修改

1.添加了绘制损失曲线图的命令来指导调参 

