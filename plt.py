#!coding=utf-8
import matplotlib
from numpy import *
import matplotlib.pyplot as plot

class plt():
    def __init__(self):
        pass

    def plotScatter(self, dataSet, labels):
        fig = plot.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dataSet[:,0], dataSet[:,1], s=30.0*labels, c=10.0*labels, marker='o')
        plot.show()


if __name__=='__main__':
    a = array([[1,1],[1,1.2],[0.1,0.1],[0.2,0.4]])
    b = array([1,1,2,2])
    p = plt()
    p.plotScatter(a,b)