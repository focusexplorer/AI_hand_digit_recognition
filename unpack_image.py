import os 
import pickle,gzip
from matplotlib import pyplot
print('Loadidng data from mnist.pkl.gz ...')
#with gzip.open('mnist.pkl.gz','rb') as f:
# train_set,valid_set,test_set=pickle.load(f)
f=gzip.open('mnist.pkl.gz','rb')
train_set,valid_set,test_set=pickle.load(f)

#create mnist folder
imgs_dir='mnist'
#os.system('mkdir -p {0}'.format(imgs_dir))
os.system('mkdir {0}'.format(imgs_dir))
datasets={'train':train_set,'val':valid_set,'test':test_set}

#transfer train,val and test
for dataname,dataset in datasets.items():
	print('Convering {0} dataset ...'.format(dataname))
	data_dir=os.sep.join([imgs_dir,dataname])
	 
#	os.system('mkdir -p {0}'.format(data_dir))
	os.system('mkdir {0}'.format(data_dir))
	 
	for i,(img,label) in enumerate(zip(*dataset)):
		filename='{0:0>6d}_{1}.jpg'.format(i,label)
#		filename='{0:0>6d}_{1}.png'.format(i,label)
		filepath=os.sep.join([data_dir,filename])

		img=img.reshape((28,28))
	  
		pyplot.imsave(filepath,img,cmap='gray')
		if (i+1)%10000==0:
			print('{0} imgaes converted!'.format(i+1))