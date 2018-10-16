from PIL import Image
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import re
import inspect
import copy


STUDY_RATE=0.0015;

def relu(x):
	y=copy.deepcopy(x);
	y[y<0]*=0.1;
	return y;

def drelu(x):
	y=copy.deepcopy(x);
	y[y>=0]=1;
	y[y<0]=0.1;
	return y;

def active(x):
	return relu(x);
def dactive(x):
	return drelu(x);

def back_level(de_do,o,on,a,m,b):
	do_dn=dactive(on);
	de_da=np.dot(de_do*do_dn,b.T);
	de_dm=de_do*do_dn;

	a-=STUDY_RATE*de_da;
	m-=STUDY_RATE*de_dm;

	de_db=np.dot(a.T,de_do*do_dn);
	return de_db;	
def judge(p,w1,m1,w2,m2):
	on1=np.dot(w1,p)+m1;
	o1=active(on1);
	on2=np.dot(w2,o1)+m2;
	o2=active(on2);
	return o2;
	
def train(p,w1,m1,w2,m2,rv):
	on1=np.dot(w1,p)+m1;
	o1=active(on1);
	on2=np.dot(w2,o1)+m2;
	o2=active(on2);
	de_do2=o2-rv;	
	de_do1=back_level(de_do2,o2,on2,w2,m2,o1);
	de_dp=back_level(de_do1,o1,on1,w1,m1,p);
	de=de_do2*de_do2;
#	print 'o2=',o2.T
#	print 'e==================================================================',de.sum();

	
N=784;
H=100;
OUT=10;
w1=np.random.normal(0.0,pow(H,-0.5),(H,N));
w2=np.random.normal(0.0,pow(OUT,-0.5),(OUT,H));
m1=np.random.uniform(0.001,0.1,(H,1));
m2=np.random.uniform(0.001,0.1,(OUT,1));
print 'w1=',w1,'w2=',w2
print 'm1=',m1,'m2=',m2

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
		
		single=0
		if single>0:
			for i in range(single):
				train(p,w1,m1,w2,m2,rv);
			sys.exit();
		else:
			train(p,w1,m1,w2,m2,rv);

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
	o2=judge(p,w1,m1,w2,m2);
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





