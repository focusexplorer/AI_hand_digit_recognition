from PIL import Image
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import re
import inspect
import copy

STUDY_RATE=0.1;#sigmoid

def sigmoid(x):
    return 1 / (1 + np.e ** -x);
def dsigmoid(x):
	return sigmoid(x)*(1-sigmoid(x));

def active(x):
	return sigmoid(x);
def dactive(x):
	return dsigmoid(x);

def back_level(de_do,o,on,a,b):

#				print 'restrict';
#	a=a-STUDY_RATE*de_da;//this will lead w not changed out side;

	a-=STUDY_RATE*de_da;

#	print 'de_da=',de_da
#	print de_da[I,J]
#	print 'a=',a
#	print 'STUDY_RATE=',STUDY_RATE
#	a[5,5]=50;#--------------------------------------------------------------------
	
	return de_db;	

def back_level2(de_do,o,on,a,b):
	do_dn=dactive(on);
	de_da=np.dot(de_do*do_dn,b.T);
	a-=STUDY_RATE*de_da;
	de_db=np.dot(a.T,de_do*do_dn);
#use matrix directly will accelerate computation
#	de_da=np.zeros(a.shape,dtype=np.double);
#	for i in range(0,a.shape[0]):
#		for j in range(0,a.shape[1]):
#			de_da[i,j]=de_do[i]*dactive(on[i])*b[j];
#	a-=STUDY_RATE*de_da;
#	de_db=np.zeros(b.shape,dtype=np.double); 
#	for j in range(0,a.shape[1]):
#		for i in range(0,a.shape[0]):
#			de_db[j]+=de_do[i]*dactive(on[i])*a[i,j];
	return de_db;	
def judge(p,w1,w2):
	on1=np.dot(w1,p);
	o1=active(on1);
	on2=np.dot(w2,o1);
	o2=active(on2);
	return o2;
	
def train(p,w1,w2,rv):
	on1=np.dot(w1,p);
	o1=active(on1);
	on2=np.dot(w2,o1);
	o2=active(on2);
	de_do2=o2-rv;	
	de_do1=back_level2(de_do2,o2,on2,w2,o1);
	de_dp=back_level2(de_do1,o1,on1,w1,p);
	de=de_do2*de_do2;
#	print 'o2=',o2.T
#	print 'e==================================================================',de.sum();
	
N=784;#number of input
H=100;#number of hidden nodes
w1=np.random.normal(0.0,pow(H,-0.5),(H,N));
w2=np.random.normal(0.0,pow(10,-0.5),(10,H));
print 'w1=',w1,'w2=',w2
seq=0;
epochs=5
for e in range(epochs):
	for filename in glob.glob(r'mnist\train\*.png'):
		seq=seq+1
	#	print 'seq=',seq
	#	print filename
		if seq%1000==0:
			print 'seq=',seq
			

		img=np.array(Image.open(filename))
		p=img[:,:,0];
		p=p.reshape(p.size,1)/255.0*0.99+0.01;
		real_value=int(re.split(r"[_.]",filename)[1]);
		rv=np.zeros((10,1),dtype=np.double)+0.01;
		rv[real_value,0]=0.99; 
		
		single=0000000
		if single>10:
			for i in range(single):
				train(p,w1,w2,rv);
			sys.exit();
		else:
			train(p,w1,w2,rv);
#sys.exit();

seq=0;
succ=0;
for filename in glob.glob(r'mnist\val\*.png'):
	seq=seq+1
	if seq%1000==0:
		print 'seq=',seq
		
	img=np.array(Image.open(filename))
	p=img[:,:,0];
	p=p.reshape(p.size,1)/255.0;
	real_value=int(re.split(r"[_.]",filename)[1]);
	o2=judge(p,w1,w2);
	max_value=o2[0];
	max_index=0;
	for i in range(o2.shape[0]):
		if max_value<o2[i]:
			max_value=o2[i];
			max_index=i;
	print seq,filename,'=>',max_index
	
	if max_index==real_value:
		succ+=1;
	
print 'success_rate=',succ*100/seq,'%'
sys.exit();



