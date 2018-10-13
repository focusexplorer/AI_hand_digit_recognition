from PIL import Image
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import re
import copy



def relu(x):
	y=copy.deepcopy(x);
	y[y<0]*=0.1;
#	print 'x=',x,'y=',y
	return y;
#	if x>=0:
#		return x;
#	else:
#		return 0.1*x;

def drelu(x):
	y=copy.deepcopy(x);
	y[y>=0]=1;
	y[y<0]=0.1;
#	print 'x=',x,'y=',y
	return y;
#	if x>=0:
#		return 1;
#	else:
#		return 0.1;

x=np.random.random((5,5))-0.5;
print x;
print relu(x);
print drelu(x);
sys.exit();


def change(e):
	e[1,1]=1000;

def cv(d):
	d[1,1]=100;
	change(d);
	return d;

b1=np.zeros((10,10),dtype=np.double)
print b1
c=b1;
f=cv(c);
print b1
f[1,1]=555;
print 'f=',f

x1=10*np.ones((10,10),dtype=np.double)
print 1/x1


def sigmoid(x):
    return 1 / (1 + np.e ** -x);

def sigmoid_all(ax):
	print 'shape[0]=',ax.shape[0]
	print 'shape[1]=',ax.shape[1]
	for i in range(ax.shape[0]):
		for j in range(ax.shape[1]):
			ax[i,j]=sigmoid(ax[i,j]);
	return ax;
def active(x):
	return sigmoid(x);
	return relu(x);
def dactive(x):
	return dsigmoid(x);
	return drelu(x);
def active_all(x):
	return sigmoid_all(x);
	return relu_all(x);
print '=================='
f=np.ones((10,1),dtype=np.double)
#f=f[0,:];
print f
print active(f)
print active_all(f)