import numpy as np
from sklearn.metrics import auc

xx=[]
for i in range(3):
    xx.append(i)
# xx = [.1, .2, .3]
yy = [4, 7, 8]
zz = [4, 7.1, 8.2]
t=auc(xx,yy)
# print('computed AUC using sklearn.metrics.auc: {}'.format(t))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(xx,zz)))
