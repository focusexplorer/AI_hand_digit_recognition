from PIL import Image
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import re

#define base type and functions
def relu(x):
	if x>=0:
		return x;
	else:
		return 0.1*x;
def cal_forward(indata,w,b):
	n=w.shape[0];
#	print w.shape
#	print 'n=',n
	outdata=np.zeros((n,n),dtype=np.double);
	for i in range(0,n):
		for j in range(0,n):
#			print 'i=',i,'j=',j
			outdata[i,j]=relu(w[i,j]*np.sum(indata[i:i+3,j:j+3])+b[i,j]);
	return outdata;

def chi_hua(indata):
	n=indata.shape[0]/2;
	outdata=np.zeros((n,n),dtype=np.double);
	for i in range(0,n):
		for j in range(0,n):
			outdata[i,j]=np.mean(indata[2*i:2*i+2,2*j:2*j+2]);
	return outdata;

def cal_backward(dr1,r0):
	n=dr1.shape[0];
	dw=np.zeros((n,n),dtype=np.double);
	for i in range(0,n):
		for j in range(0,n):
			dw[i,j]=dr1[i,j]/relu(np.sum(r0[i:i+3,j:j+3]));
	return dw;

def cal_backward_db(dr1):
	n=dr1.shape[0];
	db=np.zeros((n,n),dtype=np.double);
	for i in range(0,n):
		for j in range(0,n):
			db[i,j]=relu(dr1[i,j]);
	return db;

def cal_backward2(dr1,w1):
	n=dr1.shape[0];
	dr0=np.zeros((n+2,n+2),dtype=np.double);
	for i in range(0,n+2):
		for j in range(0,n+2):
			for x in range(0,3):
				for y in range(0,3):
					x1=i-x;
					y1=j-y;
					if x1<0 or x1>=n or y1<0 or y1>=n:
						continue;
					else:
						dr0[i,j]=dr0[i,j]-dr1[x1,y1]/relu(w1[x1,y1]);
	return dr0;
#init

w1=0.1*np.ones((26,26),dtype=np.double)
b1=np.zeros((26,26),dtype=np.double)
r1=np.zeros((26,26),dtype=np.double)
w2=0.1*np.ones((24,24),dtype=np.double)
b2=np.zeros((24,24),dtype=np.double)
r2=np.zeros((24,24),dtype=np.double)

w4=0.1*np.ones((10,10),dtype=np.double)
b4=np.zeros((10,10),dtype=np.double)
r4=np.zeros((10,10),dtype=np.double)
w5=0.1*np.ones((8,8),dtype=np.double)
b5=np.zeros((8,8),dtype=np.double)
r5=np.zeros((8,8),dtype=np.double)

w6=0.1*np.ones((8,8),dtype=np.double)
print w1
print r1
learn_rate=0.01;
#sys.exit();
#read all pictures
apa=[]
seq=0;
for filename in glob.glob(r'mnist\train\*.png'):
	seq=seq+1
	if seq>2:
		break;
	print 'seq=',seq
	print filename
	img=np.array(Image.open(filename))
	#print img.shape
	p=img[:,:,0];
	apa.append(p)
	r0=p;
	r1=cal_forward(r0,w1,b1);
#	print 'r1=',r1;
	r2=cal_forward(r1,w2,b2);
#	print 'r2=',r2;
	r3=chi_hua(r2);
#	print 'r3=',r3;
	r4=cal_forward(r3,w4,b4);
#	print 'r4=',r4
	r5=cal_forward(r4,w5,b5);
	print 'r5=',r5

	r6=np.sum(r5*w6);
	print 'r6=',r6
#	sys.exit();
	
	real_value=int(re.split(r"[_.]",filename)[1])*100;
	print 'real_value=',real_value

	f6=(r6-real_value);
	print 'diff=',f6

	dw6=f6/r5;
	w6=w6-learn_rate*dw6;   
	print 'w6=',w6

	dr5=(r6-real_value)/w6;
	dw5=cal_backward(dr5,r4);
	w5=w5-learn_rate*dw5;
	print 'w5=',w5
	db5=cal_backward_db(dr5);
	b5=b5-learn_rate*db5;

	dr4=cal_backward2(dr5,w5);
	dw4=cal_backward(dr4,r3);
	w4=w4-learn_rate*dw4;
	db4=cal_backward_db(dr4);
	b4=b4-learn_rate*db4;

	dr3=cal_backward2(dr4,w4);
	dr2=np.zeros((dr3.shape[0]*2,dr3.shape[0]*2),dtype=np.double)
	for i in range(0,dr3.shape[0]):
		for j in range(0,dr3.shape[0]):
			for x in range(0,2):
				for y in range(0,2):
					dr2[2*i+x,2*j+y]=dr3[i,j]*4;
	dw2=cal_backward(dr2,r1);
	w2=w2-learn_rate*dw2;
	db2=cal_backward_db(dr2);
	b2=b2-learn_rate*db2;

	dr1=cal_backward2(dr2,w2);
	dw1=cal_backward(dr1,r0);
	w1=w1-learn_rate*dw1;
	print 'w1=',w1;
	db1=cal_backward_db(dr1);
	b1=b1-learn_rate*db1;
	print 'b1=',b1


	
	
#train pictures one by one

#output test-result