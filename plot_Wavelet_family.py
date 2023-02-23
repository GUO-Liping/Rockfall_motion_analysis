#!/usr/bin/env python    
#encoding: utf-8 

# 该程序用于绘制各类小波家族函数，包括：
#Haar (haar)
#Daubechies (db)
#Symlets (sym)
#Coiflets (coif)
#Biorthogonal (bior)
#Reverse biorthogonal (rbio)
#“Discrete” FIR approximation of Meyer wavelet (dmey)
#Gaussian wavelets (gaus)
#Mexican hat wavelet (mexh)
#Morlet wavelet (morl)
#Complex Gaussian wavelets (cgau)
#Shannon wavelets (shan)
#Frequency B-Spline wavelets (fbsp)
#Complex Morlet wavelets (cmor)

import numpy as np
import pywt
import matplotlib.pyplot as plt
from user_func_package import *
import random

if __name__ == '__main__':
	print(pywt.families())
	print(pywt.families(short=False))
	print(pywt.wavelist('gaus'))
	wavelet = pywt.ContinuousWavelet('gaus2')
	psi, x = wavelet.wavefun(level=9)
	#plt.plot(phi)
	#plt.plot(psi)
	plt.plot(x,psi)
	plt.show()