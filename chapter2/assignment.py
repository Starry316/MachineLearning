import numpy as np
"""
录入数据集
"""
w1 = np.array([[-5.01,-8.12,-3.68],
             [-5.43,-3.48,-3.54],
             [1.08,-5.52,1.66],
             [0.86,-3.78,-4.11],
             [-2.67,0.63,7.39],
             [4.94,3.29,2.08],
              [-2.51,2.09,-2.59],
              [-2.25,-2.13,-6.94],
              [5.56,2.86,-2.26],
              [1.03,-3.33,4.33]])
w2 = np.array([[-0.91,-0.18,-0.05],
              [1.30,-2.06,-3.53],
              [-7.75,-4.54,-0.95],
              [-5.47,0.50,3.92],
              [6.14,5.72,-1.85],
              [3.60,1.26,4.36],
              [5.37,-4.63,-3.65],
              [7.18,1.46,-6.66],
              [-7.39,1.17,6.30],
              [-7.50,-6.32,-0.31]])
w3 = np.array([[5.35,2.26,8.13],
              [5.12,3.22,-2.66],
               [-1.34,-5.31,-9.87],
               [4.48,3.42,5.19],
               [7.11,2.39,9.21],
               [7.17,4.33,-0.98],
               [5.75,3.97,6.65],
               [0.77,0.27,2.41],
               [0.90,-0.43,-8.71],
               [3.52,-0.36,6.43]])
# 上面的格式是为了方便录入，还需要进行转置
w1 = w1.T
w2 = w2.T
w3 = w3.T
# data中存储所有的样本数据
data = [w1,w2,w3]

"""
录入测试点
"""
# x中存储测试点
x = []
x0 = np.array([1,2,1])
x1 = np.array([5,3,2])
x2 = np.array([0,0,0])
x3 = np.array([1,0,0])
x.append(x0)
x.append(x1)
x.append(x2)
x.append(x3)


# color = ['y','r','g']
# ax = plt.subplot(111, projection='3d')
# for i in range(len(data)):
#     wT = data[i].T
#     for j in range(len(wT)):
#         ax.scatter(wT[j][0],wT[j][1], wT[j][2], c=color[i])
# ax.set_zlabel('Z')  # 坐标轴
# ax.set_ylabel('Y')
# ax.set_xlabel('X')
# plt.show()

def mahalanobis_distance(x ,dataset):
    """计算马氏距离
    Args:
        x : 特征向量
        dataset: 样本数据集

    Returns:
        马氏距离的值
    """
    #  平方后的马氏距离
    res = sq_mahalanobis_distance(x ,dataset)
    #  开根号
    res = np.sqrt(res)
    return res

def sq_mahalanobis_distance(x ,dataset):
    """计算马氏距离的平方
    Args:
        x : 特征向量
        dataset: 样本数据集
    Returns:
        特征向量和样本数据集马氏距离的平方
    """
    #   u为数据集的样本均值
    u = dataset.mean(1)
    #   S为数据集的协方差矩阵
    S = np.cov(dataset)
    #   将x和u变成列向量
    x = x.T
    u = u.T
    #   首先计算 x - u
    res = x-u
    #   计算（x - u）的转置和S的逆的点积
    res = np.dot(res.T, np.linalg.inv(S))
    #  上面的结果和（x - u）计算点积
    res = np.dot(res, (x-u))
    return res

def g(x ,dataset, pw):
    """概率密度函数服从多元正态分布的判别函数
    Args:
        x : 特征向量
        dataset: 样本数据集
        pw：先验概率
    Returns:
        判别函数计算的结果
    """
    d = x.shape[0]
    # 平方马氏距离
    res = sq_mahalanobis_distance(x ,dataset)
    # 乘以-1/2
    res = res * (-0.5)
    # -（d/2 ）* ln （2pi） 实际上这一项可以去掉
    res = res - (d * np.log(2 * np.pi))* 0.5
    # - （1/2） * ln （协方差矩阵的行列式）
    res = res - 0.5 * np.log(np.linalg.det( np.cov(dataset) ))
    # ln (pw)
    res = res +  np.log(pw)
    return res

def classifier(x ,P):
    """计算特征向量在每个类别的判别函数得分，返回得分最高的类别
    Args:
        x : 特征向量
        P : 先验概率列表
    Returns:
        分类的类别
    """
    score = []
    for i in range(len(data)):
        score.append(g(x ,data[i], P[i]))
    return np.argmax(np.array(score))

"""
问题一的结果
"""
for i in range(4):
    for j in range(3):
         print("x%d 与 w%d 之间的马氏距离: %f"%(
               i,
               j+1,
               mahalanobis_distance(x[i], data[j])))

"""
问题二的结果
"""
P = [1/3, 1/3, 1/3]
for i in range(4):
    print("x%d 的分类类别是 w%d"%(i,classifier(x[i], P)+1))

"""
问题三的结果
"""
P = [0.8, 0.1, 0.1]
for i in range(4):
    print("x%d 的分类类别是 w%d"%(i,classifier(x[i], P)+1))
