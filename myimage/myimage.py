#!coding=utf-8
from PIL import Image
import numpy as np
import pandas as pd
from time import sleep
from NeuralNetWorks import *

class myimage():
    def __int__(self):
        pass

    def image2array(self, filename):
        return np.array(Image.open(filename))

    def array2image(self, _array):
        return Image.fromarray(_array)

    # 读取图片，灰度化，并转化为数组
    def image2gray2array(self, filename):
        return np.array(Image.open(filename).convert('L'))

    def cut(self, _array, localtion, size):
        return _array[localtion[0]:localtion[0] + size[1], localtion[1]: localtion[1] + size[0]]

    def scanning(self, _array, size, step):
        m, n = _array.shape
        for i in range((m-size[1])/step):
            for j in range((n-size[0])/step):
                a = self.cut(_array, (i * step, j * step), size)
                ip = self.array2image(a)
                name = './datas/' + str(i) + '_' + str(j) + '.jpg'
                ip.save(name)
                # ip.show()
                # sleep(1)
                # ip.close()

def image2file():
    im = myimage()
    dataSet = np.array(np.zeros((1, 70 * 100)))
    for i in range(1000):
        name = './vcode/' + str(i) + '.gif'
        a = im.image2gray2array(name)
        dataSet = np.row_stack((dataSet, a.reshape((1, 70 * 100))))
    X = dataSet[1:, :]
    recode = np.loadtxt('./vcode/recode.csv', delimiter=',', dtype=str, skiprows=1)
    y = sety().reshape((1000, 1))
    X = np.hstack((X, y))
    print X.shape
    np.savetxt('X.csv', X)

def sety():
    df = pd.read_csv('./vcode/recode.csv')
    a = np.array(df)
    yo = list(a[:, 2])
    a = list(set(list(a[:, 2])))
    a.sort()
    # print a
    # print yo
    y = []
    for i in yo:
        y.append(a.index(i) + 1)
    return np.array(y)



if __name__ == '__main__':
    image2file()