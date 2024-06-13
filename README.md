# 1. 论文介绍
《MAMO: Memory-Augmented Meta-Optimization for Cold-start Recommendation》
## 1.1 背景介绍

推荐系统在网络和移动应用中扮演着越来越重要的角色。但是,大多数推荐系统都面临着冷启动问题,即对于新用户或新商品,由于缺乏足够的用户-商品交互记录,推荐系统难以给出准确的推荐。

## 1.2 论文方法


论文主要研究了推荐系统中的冷启动问题，并提出了一种新的记忆增强的元优化方法（Memory-Augmented Meta-Optimization，简称MAMO）来解决这一问题

本文提出了一种内存增强的元优化推荐模型(MAMO),以解决冷启动问题。MAMO包括两个主要部分:

- 推荐模型:预测用户对商品的偏好评分。
- 内存增强元优化学习器:为推荐模型提供个性化的参数初始化,以及用于快速预测用户偏好的任务相关记忆。


## 1.3 数据集介绍
原论文使用了两个公开的推荐系统数据集进行评估：

MovieLens 1M：包含约6000名用户对3000多部电影的大约100万条评分，评分范围是1到5。
Book-crossing：在过滤掉没有相关用户或项目信息的记录后，该数据集包含大约50万名用户对50多万本书的约60万条评分，评分范围是1到10。


## 1.4 论文复现

根据课程内容和要求，该作业pytorch版本基于[论文代码](https://github.com/dongmanqing/Code-for-MAMO)实现，之后将pytorch框架的代码迁移到华为MindSpore框架。

# 2. pytorch实现版本

## 2.1 准备工作

### 创建环境：

```
conda create -n mamo python=3.7
```

### 安装依赖包：

```
certifi==2024.6.2
charset-normalizer==3.3.2
idna==3.7
numpy==1.21.6
pandas==1.3.5
Pillow==9.5.0
python-dateutil==2.9.0.post0
pytz==2024.1
requests==2.31.0
six==1.16.0
torch==1.12.1+cu113
torchaudio==0.12.1+cu113
torchvision==0.13.1+cu113
tqdm==4.65.0
typing_extensions==4.7.1
urllib3==2.0.7
```

### 数据集下载：
1. 原始数据集可以从以下位置下载：
```
https://files.grouplens.org/datasets/movielens/ml-1m.zip
```
2. 将数据集放入相应的 `data_raw` 文件夹中
3. 创建一个名为 `data_processed` 的文件夹，您可以通过以下命令处理原始数据集：
```
python prepareDataset.py
```
4. 已处理数据集的结构

```
- data_processed
 
  - movielens
    - raw
      sample_1_x1.p
      sample_1_x2.p
      sample_1_y.p
      sample_1_y0.p
      ...
    item_dict.p
    item_state_ids.p
    ratings_sorted.p
    user_dict.p
    user_state_ids.p
```

### 参数设置
```
config_settings = {
    'rand': True,  # 是否随机化
    'random_state': 100,  # 随机种子
    'split_ratio': 0.8,  # 训练集和测试集的分割比例
    'support_size': 15,  # 支持集大小
    'query_size': 5,  # 查询集大小
    'embedding_dim': 100,  # 嵌入维度
    'n_layer': 2,  # 神经网络层数
    'alpha': 0.5,  # 超参数alpha
    'beta': 0.05,  # 超参数beta
    'gamma': 0.1,  # 超参数gamma
    'rho': 0.01,  # 本地学习率rho
    'lamda': 0.05,  # 全局学习率lamda
    'tao': 0.01,  # 超参数tao
    'n_k': 3,# 内循环次数
    'batch_size': 5,  # 批量大小
    'n_epoch': 5,  # 迭代次数
    'n_inner_loop': 1,  # 内循环次数
    'active_func': 'leaky_relu',  # 激活函数
    'cuda_option': 'cuda:0'  # GPU选项
}

```

## 2.2 运行代码

1. 您可以通过如下命令运行代码：
```
python mamoRec.py
```
2. 输出结果如下：

```
cuda:0
Model parameters:
Param name: None, shape: torch.Size([2, 100])
Param name: None, shape: torch.Size([7, 100])
Param name: None, shape: torch.Size([21, 100])
Param name: None, shape: torch.Size([6, 100])
Param name: None, shape: torch.Size([100, 25])
Param name: None, shape: torch.Size([100, 2186])
Param name: None, shape: torch.Size([81, 100])
Param name: None, shape: torch.Size([150, 300])
Param name: None, shape: torch.Size([150])
Param name: None, shape: torch.Size([100, 150])
Param name: None, shape: torch.Size([100])
Param name: None, shape: torch.Size([200, 400])
Param name: None, shape: torch.Size([200])
Param name: None, shape: torch.Size([100, 200])
Param name: None, shape: torch.Size([100])
Param name: None, shape: torch.Size([200, 200])
Param name: None, shape: torch.Size([200])
Param name: None, shape: torch.Size([100, 200])
Param name: None, shape: torch.Size([100])
Param name: None, shape: torch.Size([5, 100])
Param name: None, shape: torch.Size([5])
=====================================================================
训练开始
训练结束
测试开始
0 2104
1 4865
2 1067
...
...
1204 4799
1205 328
1206 2091
测试结束
Epoch 1 cost 50.39s
=====================================================================
训练开始
训练结束
测试开始
0 5520
1 2471
2 2425
...
...
1204 4481
1205 2795
1206 2207
测试结束
Epoch 2 cost 37.57s
=====================================================================
训练开始
训练结束
测试开始
0 4461
1 4580
2 1009
3 5429
...
...
1204 4481
1205 2795
1206 2207
测试结束
Epoch 3 cost 38.89s
=====================================================================
训练开始
训练结束
测试开始
0 4461
1 4580
2 1009
3 5429
...
...
1204 5225
1205 5883
1206 3195
测试结束
Epoch 4 cost 38.88s
=====================================================================
训练开始
训练结束
测试开始
0 541
1 4580
2 1213
...
...
1204 4489
1205 276
1206 207
测试结束
Epoch 5 cost 39.71s

Process finished with exit code 0
```
3. 每个 Epoch 使用时间如下：
```
Epoch 1 cost 50.39s
Epoch 2 cost 37.57s
Epoch 3 cost 38.89s
Epoch 4 cost 38.88s
Epoch 5 cost 39.71s
```


# 3. mindspore实现版本

## 3.1 mindspore框架介绍
MindSpore 是一个开源的深度学习框架，由华为公司开发和维护。它的设计目标是提供一种高效、易用、灵活且安全的深度学习开发平台，支持在各种硬件设备上进行分布式训练和推理。以下是 MindSpore 框架的一些关键特点和功能：

1. **自动并行化：** MindSpore 提供了自动并行化的能力，可以根据硬件资源和模型结构自动实现并行计算，提高训练效率。

2. **模块化设计：** MindSpore 的设计采用了模块化的思想，将整个深度学习流程拆分成多个模块，提供了更灵活的使用方式，并支持自定义扩展。

3. **轻量级：** MindSpore 的核心代码库非常精简，易于理解和定制，同时框架本身支持混合精度训练，可以在保证模型性能的同时减少计算资源消耗。

4. **跨平台支持：** MindSpore 不仅支持在 CPU、GPU 上进行训练和推理，还能够很好地与华为自研的昇腾 AI 处理器（Ascend）配合使用，实现高性能的深度学习计算。

5. **安全性：** MindSpore 在设计上考虑了模型隐私和安全性，提供了可信执行环境（Trusted Execution Environment，TEE）等功能，保护用户数据和模型不受攻击。

6. **易用性：** MindSpore 提供了丰富的 API 接口和开发工具，使得用户能够快速上手并进行深度学习模型的开发和调试。

总的来说，MindSpore 是一个功能丰富、灵活易用的深度学习框架，适用于各种场景下的模型训练和推理任务。


## 3.2 准备工作

使用华为 ECS (弹性云服务器)，操作系统 Ubuntu 22.04。

### 安装 anaconda：

```
wget -c https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh
```

### 创建环境：

```
conda create -n mamo python=3.7
conda activate mamo
```

### 安装依赖包：
```
asttokens==2.4.1
astunparse==1.6.3
mindspore==2.0.0
numpy==1.21.6
packaging==24.0
pandas==1.3.5
Pillow==9.5.0
protobuf==4.24.4
psutil==5.9.8
python-dateutil==2.9.0.post0
pytz==2024.1
scipy==1.7.3
six==1.16.0
tqdm==4.65.0

```
或者，您可以使用以下命令安装依赖包：

```
pip install -r requirements.txt
```

### 数据集下载：
1. 原始数据集可以从以下位置下载
```
https://files.grouplens.org/datasets/movielens/ml-1m.zip
```
2. 将数据集放入相应的 `data_raw` 文件夹中
3. 创建一个名为 `data_processed` 的文件夹，您可以通过以下命令处理原始数据集
```
python prepareDataset.py
```
4. 已处理数据集的结构

```
- data_processed
 
  - movielens
    - raw
      sample_1_x1.p
      sample_1_x2.p
      sample_1_y.p
      sample_1_y0.p
      ...
    item_dict.p
    item_state_ids.p
    ratings_sorted.p
    user_dict.p
    user_state_ids.p
```

### 参数设置
```
config_settings = {
    'rand': True,  # 是否随机化
    'random_state': 100,  # 随机种子
    'split_ratio': 0.8,  # 训练集和测试集的分割比例
    'support_size': 15,  # 支持集大小
    'query_size': 5,  # 查询集大小
    'embedding_dim': 100,  # 嵌入维度
    'n_layer': 2,  # 神经网络层数
    'alpha': 0.5,  # 超参数alpha
    'beta': 0.05,  # 超参数beta
    'gamma': 0.1,  # 超参数gamma
    'rho': 0.01,  # 本地学习率rho
    'lamda': 0.05,  # 全局学习率lamda
    'tao': 0.01,  # 超参数tao
    'n_k': 3,# 内循环次数
    'batch_size': 5,  # 批量大小
    'n_epoch': 5,  # 迭代次数
    'n_inner_loop': 1,  # 内循环次数
    'active_func': 'leaky_relu',  # 激活函数
    'cuda_option': 'cuda:0'  # GPU选项
}

```

## 3.3 模型迁移

查阅[华为官方文档](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/pytorch_api_mapping.html)，将模型代码中的pytorch接口替换为mindspore的接口。

此外华为还提供了[MindConverter](https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.7/migrate_3rd_scripts_mindconverter.html)，`Mindspore生态适配工具——MSadapter`等工具，方便从pytorch迁移模型。

本实验中更改的部分接口：

| pytorch API        | mindspore API                | 接口功能                          | 区别                                                     |
| ------------------------- | ---------------------------------- | ----------------------------- | ------------------------------------------------------------ |
| torch.from_numpy          | mindspore.tensor.from_numpy        | 从numpy得到tensor             | 无                                                           |
| torch.tensor.to           | mindspore.tensor.to_device         | 将tensor传入指定的设备        | 无                                                           |
| torch.utils.data.Dataset  | mindspore.dataset.GeneratorDataset | 数据集类                      | PyTorch：自定义数据集的抽象类，自定义数据子类可以通过调用`__len__()`和`__getitem__()`这两个方法继承这个抽象类。<br />MindSpore：通过每次调用Python层自定义的Dataset以生成数据集。 |
| torch.zeros_like          | mindspore.ops.ZerosLike            | 获得指定shape的全零元素tensor | 无                                                           |
| torch.nn.Sigmoid          | mindspore.nn.Sigmoid               | 激活函数                      | 无                                                           |
| torch.nn.Tanh             | mindspore.nn.Tanh                  | 激活函数                      | 无                                                           |
| torch.nn.ReLU             | mindspore.nn.ReLU                  | 激活函数                      | 无                                                           |
| torch.nn.Softmax          | mindspore.nn.Softmax               | 归一化                        | 无                                                           |
| torch.nn.LeakyReLU        | mindspore.nn.LeakyReLU             | 激活函数                      | 无                                                           |
| torch.nn.Sequential       | mindspore.nn.SequentialCell        | 整合多个网络模块              | 无                                                           |
| torch.argmax              | mindspore.ops.argmax               | 返回最大值下标                | PyTorch：沿着给定的维度返回Tensor最大值所在的下标，返回值类型为torch.int64。<br />MindSpore：MindSpore此API实现功能与PyTorch基本一致，返回值类型为int32. |
| torch.abs                 | mindspore.ops.abs                  | 计算tensor绝对值              | PyTorch：计算输入的绝对值。<br />MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名不同。 |
| torch.mean                | mindspore.ops.ReduceMean           | 计算均值                      | 无                                                           |
| torch.optim.Adam          | mindspore.nn.Adam                  | 优化器                        | 无                                                           |
| torch.nn.CrossEntropyLoss | mindspore.nn.CrossEntropyLoss      | 损失函数                      | PyTorch：计算预测值和目标值之间的交叉熵损失。<br />MindSpore：MindSpore此API实现功能与PyTorch基本一致，而且目标值支持两种不同的数据形式：标量和概率。 |
| torch.nn.Module           | mindspore.nn.Cell                  | 神经网络的基本构成单位        |                                                              |
| torch.nn.Linear           | mindspore.nn.Dense                 | 全连接层                      | PyTorch：全连接层，实现矩阵相乘的运算。<br />MindSpore：MindSpore此API实现功能与PyTorch基本一致，而且可以在全连接层后添加激活函数。 |
| torch.cat                 | mindspore.ops.concat               | tensor按照指定维度拼接        | 无                                                           |
| torch.randn               | mindspore.ops.StandardNormal       | 获得正态分布数据的tensor      | 无                                                           |
| torch.mm                  | mindspore.ops.MatMul               | 矩阵乘法                      | 无                                                           |
| torch.sqrt                | mindspore.ops.Sqrt                 | 开根号                        | 无                                                           |
| torch.sum                 | mindspore.ops.ReduceSum            | 求和                          | 无                                                           |
| torch.Tensor.mul          | mindspore.ops.Mul                  | 相乘                          | 无                                                           |
| torch.div                 | mindspore.ops.div                  | 除法                          | 无                                                           |
| torch.nn.Embedding        | mindspore.nn.Embedding             |                               | PyTorch：支持使用`_weight`属性初始化embedding，并且可以通过`weight`变量获取当前embedding的权重。<br />MindSpore：支持使用`embedding_table`属性初始化embedding，并且可以通过`embedding_table`属性获取当前embedding的权重。除此之外，当`use_one_hot`为True时，你可以得到具有one-hot特征的embedding。 |
| torch.tensor.repeat       | mindspore.ops.tile                 | 对tensor进行重复叠加          | 无                                                           |
| torch.tensor.view         | mindspore.ops.Reshape              | 重新排列tensor的维度          | 无                                                           |
| Adam.zero_grad            | Adam.clear_grad                    | 清除梯度                      | 无                                                           |
|                           |                                    |                               |                                                              |



## 3.4 迁移过程

### 数据处理部分

```python
import mindspore.dataset as ds

class UserDataLoader:
    def __init__(self, x1, x2, y, y0, transform=None):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.y0 = y0
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if isinstance(idx, mindspore.tensor):
            idx = idx.asnumpy()

        user_info = self.x1[idx]
        item_info = self.x2[idx]
        ratings = self.y[idx]
        cold_labels = self.y0[idx]

        return user_info, item_info, ratings, cold_labels
    
# 数据加载方法
dataset_generator = UserDataLoader()
dataset = ds.GeneratorDataset(dataset_generator, ["user_info", "item_info", "ratings", "cold_labels"], shuffle=False)

for data in dataset.create_dict_iterator():
    print(data["user_info"], data["item_info"], data["ratings"], data["cold_labels"])
```



### 网络训练部分

```python
import mindspore

# 继承父类 Cell
class BASEModel(mindspore.nn.Cell):
    def __init__(self, input1_module, input2_module, embedding1_module, embedding2_module, rec_module):
        super(BASEModel, self).__init__(auto_prefix=False)

        self.input_user_loading = input1_module
        self.input_item_loading = input2_module
        self.user_embedding = embedding1_module
        self.item_embedding = embedding2_module
        self.rec_model = rec_module
	
    # 对应于forward
    def construct(self, x1, x2):
        pu, pi = self.input_user_loading(x1), self.input_item_loading(x2)
        eu, ei = self.user_embedding(pu), self.item_embedding(pi)
        rec_value = self.rec_model(eu, ei)
        return rec_value

    def get_weights(self):
        u_emb_params = get_params(self.user_embedding.parameters())
        i_emb_params = get_params(self.item_embedding.parameters())
        rec_params = get_params(self.rec_model.parameters())
        return u_emb_params, i_emb_params, rec_params

    def get_zero_weights(self):
        zeros_like_u_emb_params = get_zeros_like_params(self.user_embedding.parameters())
        zeros_like_i_emb_params = get_zeros_like_params(self.item_embedding.parameters())
        zeros_like_rec_params = get_zeros_like_params(self.rec_model.parameters())
        return zeros_like_u_emb_params, zeros_like_i_emb_params, zeros_like_rec_params

    def init_weights(self, u_emb_para, i_emb_para, rec_para):
        init_params(self.user_embedding.parameters(), u_emb_para)
        init_params(self.item_embedding.parameters(), i_emb_para)
        init_params(self.rec_model.parameters(), rec_para)

    def get_grad(self):
        u_grad = get_grad(self.user_embedding.parameters())
        i_grad = get_grad(self.item_embedding.parameters())
        r_grad = get_grad(self.rec_model.parameters())
        return u_grad, i_grad, r_grad

    def init_u_mem_weights(self, u_emb_para, mu, tao, i_emb_para, rec_para):
        init_u_mem_params(self.user_embedding.parameters(), u_emb_para, mu, tao)
        init_params(self.item_embedding.parameters(), i_emb_para)
        init_params(self.rec_model.parameters(), rec_para)

    def init_ui_mem_weights(self, att_values, task_mem):
        # init the weights only for the mem layer
        u_mui = task_mem.read_head(att_values)
        init_ui_mem_params(self.rec_model.mem_layer.parameters(), u_mui)

    def get_ui_mem_weights(self):
        return get_params(self.rec_model.mem_layer.parameters())
```

## 3.5 运行代码

1. 您可以通过如下命令运行代码：
```
python mamoRec.py
```
2. 将 pytorch 模型转为 mindspore 模型后的实验结果如下：

```
CPU
Model parameters:
Param name: input_user_loading.embedding_gender.embedding_table, shape: (2, 100)
Param name: input_user_loading.embedding_age.embedding_table, shape: (7, 100)
Param name: input_user_loading.embedding_occupation.embedding_table, shape: (21, 100)
Param name: input_item_loading.emb_rate.embedding_table, shape: (6, 100)
Param name: input_item_loading.emb_genre.weight, shape: (100, 25)
Param name: input_item_loading.emb_genre.bias, shape: (100,)
Param name: input_item_loading.emb_director.weight, shape: (100, 2186)
Param name: input_item_loading.emb_director.bias, shape: (100,)
Param name: input_item_loading.emb_year.embedding_table, shape: (81, 100)
Param name: user_embedding.fc.0.weight, shape: (150, 300)
Param name: user_embedding.fc.0.bias, shape: (150,)
Param name: user_embedding.final_layer.0.weight, shape: (100, 150)
Param name: user_embedding.final_layer.0.bias, shape: (100,)
Param name: item_embedding.fc.0.weight, shape: (200, 400)
Param name: item_embedding.fc.0.bias, shape: (200,)
Param name: item_embedding.final_layer.0.weight, shape: (100, 200)
Param name: item_embedding.final_layer.0.bias, shape: (100,)
Param name: rec_model.mem_layer.weight, shape: (200, 200)
Param name: rec_model.mem_layer.bias, shape: (200,)
Param name: rec_model.fc.0.weight, shape: (100, 200)
Param name: rec_model.fc.0.bias, shape: (100,)
Param name: rec_model.final_layer.0.weight, shape: (5, 100)
Param name: rec_model.final_layer.0.bias, shape: (5,)
=====================================================================
训练开始
训练结束
测试开始
0 2872
1 315
2 4907
...
...
1204 1627
1205 4245
1206 1328
测试结束
Epoch 1 cost 308.15s
=====================================================================
训练开始
训练结束
测试开始
0 222
1 2911
2 4112
...
...
1204 242
1205 245
1206 1879
测试结束
Epoch 2 cost 335.12s
=====================================================================
训练开始
训练结束
测试开始
0 2873
1 3151
2 407
...
...
1204 127
1205 356
1206 99
测试结束
Epoch 3 cost 310.09s
=====================================================================
训练开始
训练结束
测试开始
0 153
1 3215
2 4936
...
...
1204 1647
1205 445
1206 12
测试结束
Epoch 4 cost 431.27s
=====================================================================
训练开始
训练结束
测试开始
0 22
1 3151
2 490
...
...
1204 167
1205 4891
1206 1316
测试结束
Epoch 5 cost 309.11s

Process finished with exit code 0

```
3. 每个 Epoch 使用时间如下：
```
Epoch 1 cost 316.22s
Epoch 2 cost 335.12s
Epoch 3 cost 310.09s
Epoch 4 cost 431.27s
Epoch 5 cost 309.11s
```









