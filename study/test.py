from PIL import Image
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

img=np.array(Image.open('./7.png'))
plt.figure("7")
plt.imshow(img);
plt.axis('off')
print img.shape  
print img.dtype 
print img.size 
print type(img)
print img[:,:,0];
#plt.show()

sys.exit();
x=np.linspace(0,5,100)
y=2*np.sin(x)+0.3*x**2;
plt.plot(x,y,'.');
plt.show();