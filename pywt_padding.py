# -*- coding:utf-8 -*-
import numpy as np
import pywt
import matplotlib.pyplot as plt
x = np.arange(0,2*np.pi,0.01)
y = np.sin(x)+0.25*x
yp1 = pywt.pad(y,(500,500),'symmetric')
yp2 = pywt.pad(y,(500,500),'reflect')
yp3 = pywt.pad(y,(500,500),'smooth')
yp4 = pywt.pad(y,(500,500),'constant')
yp5 = pywt.pad(y,(500,500),'zero')
yp6 = pywt.pad(y,(500,500),'periodic')
yp7 = pywt.pad(y,(500,500),'periodization')
yp8 = pywt.pad(y,(500,500),'antisymmetric')
yp9 = pywt.pad(y,(500,500),'antireflect')

plt.plot(y, '*', label='y')
plt.plot(yp1, label='yp1')
plt.plot(yp2, label='yp2')
plt.plot(yp3, label='yp3')
plt.plot(yp4, label='yp4')
plt.plot(yp5, label='yp5')
plt.plot(yp6, '-k',label='yp6')
plt.plot(yp7, '-.',label='yp7')
plt.plot(yp8, label='yp8')
plt.plot(yp9, '--',label='yp9')
plt.legend()
plt.show()
