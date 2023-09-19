# -*- coding: utf-8 -*-
# @Time    : 2023/9/19 8:57
# @Name    : svd.py
# @email   : yangemail2@163.com
# @Author  : haoyunjixiang
import numpy as np
import math
import scipy.linalg as linalg
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#旋转矩阵 欧拉角
def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix
# 分别是x,y和z轴,也可以自定义旋转轴
axis_x, axis_y, axis_z = [1,0,0], [0,1,0], [0, 0, 1]
rand_axis = [0,0,1]
#旋转角度
yaw = math.pi/180
#返回旋转矩阵
rot_matrix = rotate_mat(rand_axis, yaw)
print(rot_matrix)


# 计算点绕着轴运动后的点

A= np.random.randint(1,25,[3,10])
print(A.T)

tran_a = np.dot(rot_matrix, A)
print(tran_a.T)

mean_A = A.mean(axis=1)
print(mean_A)
x = A.T - mean_A
y = tran_a.T - tran_a.mean(axis=1)
print(x)

S = x.T.dot(y)
print(S)

u,sigma,vt = np.linalg.svd(S)
print("****")
# print(vt.T.dot(u))

print(u.dot(vt))
print(np.matmul(vt.T,u.T))


# A = np.asarray([[2,4],[1,3],[0,0],[0,0]])
#
# u,sigma,vt = np.linalg.svd(A)
# print(u,sigma,vt)