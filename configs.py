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
