from PIL import Image
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import re


def cv(b1):
	b1[1,1]=100;

b1=np.zeros((10,10),dtype=np.double)
print b1
c=b1;
cv(c);
print b1

x1=10*np.ones((10,10),dtype=np.double)
print 1/x1
