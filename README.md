# Forecasting with covariates

## Paper1 ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables

#### Motivation

在实际的时间序列预测任务中，协变量（如节假日、天气、促销活动等）对预测结果影响巨大，能提供关键的上下文信息。

当前很多强大的预训练时间序列模型（例如 Chronos、TimesFM、MOMENT 等）在大规模无协变量的语料上训练，原生并不支持协变量输入，或者支持有限。

因此需要设计轻量、模块化的适配器结构，将过去和未来的协变量信息注入到这些冻结的预训练模型中，既保持预训练知识，又提升在协变量丰富任务上的下游性能。
![Figure 1.1 ](Figure/1.1.png)




#### Method

提出了ChronosX方法，通过在预训练模型中插入两种模块来处理外生变量：一是输入注入模块（IIB），将过去时刻的协变量与对应的Token嵌入分别经过线性映射后在ReLU激活和全连接网络（FFN）中融合，更新Token嵌入；二是输出注入模块（OIB），将未来时刻的协变量与模型最后隐藏状态分别映射后在ReLU和FFN中融合，生成对原始logits的调整项，两者相加后输出，从而在不改动预训练权重的情况下同时利用过去和未来的外生信息提升预测性能
![Figure 1.1 ](Figure/1.2.png)
