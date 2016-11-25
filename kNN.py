#!coding=utf-8
from numpy import *

class kNN():
    def __init__(self):
        pass

    # 测试时产生样本数据
    def createData(self):
        group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']
        return  group,labels

    # 分类函数
    def classify(self, initX, dataSet, labels, k):
        '''
        k近邻分类函数
        :param initX: array, 待分类数据
        :param dataSet: array, 训练数据集
        :param labels: array, 训练数据集的标签
        :param k: int, k值
        :return: 分类结果标签
        '''
        distance = sum((dataSet - initX) ** 2, axis=1) ** 0.5   #计算距离
        sortedDistance = distance.argsort()     #根据距离升序排序，并返回index

        # 计算距离最近的K个样本中，每个标签类的样本个数
        classCounte = {}
        for i in range(k):
            if classCounte.has_key(labels[sortedDistance[i]]):
                classCounte[labels[sortedDistance[i]]] += 1
            else:
                classCounte[labels[sortedDistance[i]]] = 1
        # 根据样本个数降序对标签进行排序
        result = sorted(classCounte.items(), key=lambda item:item[1],reverse=True)
        return result[0][0]

if __name__=='__main__':
    C = kNN()
    dataSet,labels = C.createData()
    print C.classify([1.1, 1.2], dataSet, labels, 3)