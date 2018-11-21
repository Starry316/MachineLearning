import numpy as np

def mean(data):
    """计算样本均值
    Args:
        data : 样本集
    Returns:
        样本均值
    """
    u = data.mean(1)
    return u

def sq_sigma(data, u):
    """最大似然估计的方差
    Args:
        data : 样本集
        u : 样本均值
    Returns:
        方差
    """
    # 使用numpy的boardcast机制，将样本集中所有的样本减去均值
    res =(data.T - u).T
    # 将矩阵中每个元素平方
    res = res**2
    # 计算所有元素的和
    res = np.sum(res)
    # 除以样本数 n
    res = res/data.shape[1]
    return res

def cov(data, u):
    """最大似然估计的协方差矩阵
    Args:
        data : 样本集
        u : 样本均值
    Returns:
        协方差矩阵
    """
    # 将样本集中所有的样本减去均值
    res =(data.T - u).T
    # 上面得到的矩阵与自身的转置计算点积
    res =np.dot(res,res.T)
    # 除以样本数 n
    res = res/data.shape[1]
    return res

"""
录入数据集
"""
w1 = np.array([[0.42, -0.087, 0.58],
            [-0.2, -3.3, -3.4],
            [1.3, -0.32, 1.7],
            [0.39, 0.71, 0.23],
            [-1.6, -5.3, -0.15],
            [-0.029, 0.89, -4.7],
            [-0.23, 1.9, 2.2],
            [0.27, -0.3, -0.87],
            [-1.9, 0.76, -2.1],
            [0.87, -1.0, -2.6]])
w1 = w1.T
w2 = np.array([[-0.4, 0.58, 0.089],
            [-0.31, 0.27, -0.04],
            [0.38, 0.055, -0.035],
            [-0.15, 0.53, 0.011],
            [-0.35, 0.47, 0.034],
            [0.17, 0.69, 0.1],
            [-0.011, 0.55, -0.18],
            [-0.27, 0.61, 0.12],
            [-0.065, 0.49, 0.0012],
            [-0.12, 0.054, -0.063]])
w2 = w2.T

# 第一问
for i in range(3):
    s = np.array([w1[i]])
    u = mean(s)
    sigma2 = sq_sigma(s, u)
    print("特征x%d最大似然估计的均值为%f,方差为%f\n"%(i+1,u,sigma2))

# 第二问
parms = [[0,1],[0,2],[1,2]]
for i in range(3):
    s = np.array([w1[parms[i][0]], w1[parms[i][1]]])
    u = mean(s)
    covM = cov(s, u)
    print("特征x%d和x%d的最大似然估计的均值为:"%(parms[i][0]+1,
                                            parms[i][1]+1))
    print(u)
    print("协方差矩阵为:")
    print(covM)
    print()

# 第三问
u = mean(w1)
covM = cov(w1, u)
print("最大似然估计的均值为:")
print(u)
print("协方差矩阵为:")
print(covM)
print()

# 第四问
print("协方差矩阵的参数为：")
for i in range(3):
    s = np.array([w2[i]])
    u = mean(s)
    sigma2 = sq_sigma(s, u)
    print(sigma2)
print("w2的均值为：")
print(mean(w2))