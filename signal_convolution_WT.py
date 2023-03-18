#!/usr/bin/env python    
#encoding: utf-8 
# 该程序用于冲击试验速度时程曲线小波降噪，旨在获得可靠的加速度值用于预测落石冲击力
import numpy as np
import pywt
import matplotlib.pyplot as plt
from user_func_package import *
import random

if __name__ == '__main__':
	time_test = np.array([0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02,0.022,0.024,0.026,0.028,0.03,0.032,0.034,0.036,0.038,0.04,0.042,0.044,0.046,0.048,0.05,0.052,0.054,0.056,0.058,0.06,0.062,0.064,0.066,0.068,0.07,0.072,0.074,0.076,0.078,0.08,0.082,0.084,0.086,0.088,0.09,0.092,0.094,0.096,0.098,0.1,0.102,0.104,0.106,0.108,0.11,0.112,0.114,0.116,0.118,0.12,0.122,0.124,0.126,0.128,0.13,0.132,0.134,0.136,0.138,0.14,0.142,0.144,0.146,0.148,0.15,0.152,0.154,0.156,0.158,0.16,0.162,0.164,0.166,0.168,0.17,0.172,0.174,0.176,0.178,0.18,0.182,0.184,0.186,0.188,0.19,0.192,0.194,0.196,0.198,0.2,0.202,0.204,0.206,0.208,0.21,0.212,0.214,0.216,0.218,0.22,0.222,0.224,0.226,0.228,0.23,0.232,0.234,0.236,0.238,0.24,0.242,0.244,0.246,0.248,0.25,0.252,0.254,0.256,0.258,0.26,0.262,0.264,0.266,0.268,0.27,0.272,0.274,0.276,0.278,0.28,0.282,0.284,0.286,0.288,0.29,0.292,0.294,0.296,0.298,0.3,0.302,0.304,0.306,0.308,0.31,0.312,0.314,0.316,0.318,0.32,0.322,0.324,0.326,0.328,0.33,0.332,0.334,0.336,0.338,0.34,0.342,0.344,0.346,0.348,0.35,0.352,0.354])
	disp_test = np.array([0,-0.02274,-0.04048,-0.0614,-0.08732,-0.11097,-0.13462,-0.15736,-0.18283,-0.20648,-0.22922,-0.25196,-0.27607,-0.30063,-0.32518,-0.34974,-0.3693,-0.39068,-0.41114,-0.43434,-0.4589,-0.483,-0.50438,-0.52757,-0.55395,-0.5826,-0.60443,-0.62626,-0.65401,-0.6772,-0.69767,-0.72405,-0.74815,-0.76953,-0.79409,-0.81728,-0.84139,-0.8664,-0.89187,-0.91825,-0.94645,-0.96737,-0.98965,-1.01785,-1.04377,-1.0697,-1.09426,-1.11654,-1.13883,-1.15975,-1.18567,-1.20796,-1.23343,-1.25708,-1.28164,-1.3062,-1.32848,-1.35077,-1.37305,-1.39852,-1.42217,-1.44673,-1.47265,-1.49585,-1.51723,-1.54179,-1.56498,-1.58999,-1.61182,-1.63411,-1.65776,-1.68096,-1.70415,-1.7228,-1.74008,-1.76236,-1.78329,-1.80421,-1.8224,-1.84241,-1.85969,-1.87789,-1.89426,-1.90654,-1.92018,-1.94065,-1.95111,-1.96339,-1.97658,-1.98613,-1.99932,-2.0116,-2.02388,-2.03843,-2.04889,-2.05935,-2.07072,-2.083,-2.09437,-2.10665,-2.11711,-2.13121,-2.14531,-2.15668,-2.16896,-2.18124,-2.19306,-2.20398,-2.21489,-2.22808,-2.24036,-2.25173,-2.26538,-2.27948,-2.29267,-2.30677,-2.31859,-2.32996,-2.34269,-2.35634,-2.36816,-2.38044,-2.39591,-2.41046,-2.42092,-2.43411,-2.44912,-2.46094,-2.47368,-2.48732,-2.50233,-2.51325,-2.52553,-2.53962,-2.55372,-2.56737,-2.58283,-2.59602,-2.60875,-2.62194,-2.63468,-2.64696,-2.66151,-2.67561,-2.68926,-2.7029,-2.71836,-2.73155,-2.74565,-2.75975,-2.77294,-2.78795,-2.80114,-2.8166,-2.83297,-2.84434,-2.85844,-2.87118,-2.88573,-2.90074,-2.91575,-2.93121,-2.94622,-2.96259,-2.97578,-2.98988,-3.0058,-3.0199,-3.03354,-3.04901,-3.06219,-3.07902,-3.09312,-3.10767,-3.12223,-3.13769,-3.15179,-3.16771])
	sample_rate = 125
	time_updated, disp_updated = func_update_disp(time_test,disp_test, sample_rate)  # 更新采样频率至500Hz水平

	# scale = fc/f_pseudo*sample_rate，其中f_pseudo为傅里叶变换得到的伪频率
	scale =2  # 小波函数尺度参数 T=0.094s, fs=500Hz，伪中心频率0.12699对应的尺度参数为5.96853
	#key_i = int((len(time_updated)-2*n_add-1)*0.5)  # 关键索引，便于求解小波变换幅值参数0.918for12,0.79 fors=6

	# 边缘效应处理方法：pading，即向数据两段人工添加数据，小波变换后在除去这些数据
	#time_updated1 = pywt.pad(time_updated0,(0,500),'zero')

	n_fit = int(0.10*len(disp_updated))	# 第一个常数，表示用于待处理数据中可用于抛物线拟合的捕捉数据点数量
	n_add = int(0.50*len(disp_updated))	# 第二个常数，表示在信号首尾端需要添加的数据点数量

	plt.plot(time_updated, disp_updated,'*')
	time_updated, disp_updated = func_user_pad(time_updated, disp_updated, n_fit, 'before', n_add)
	time_updated, disp_updated = func_user_pad(time_updated, disp_updated, n_fit, 'after',  n_add)
	plt.plot(time_updated, disp_updated,'-')
	plt.show()

	total_time = np.max(time_updated)
	t_initial = np.arange(0,5.0,step=0.002)
	analy_t = func_analytical_signal_impact(0.002)[-1]
	analy_ut = func_analytical_signal_impact(0.002)[0]
	analy_vt = func_analytical_signal_impact(0.002)[1]
	analy_at = func_analytical_signal_impact(0.002)[2]

	white_noise = np.array([random.gauss(0.0, 1.0) for i in range(len(analy_ut))])  #是为了保证多次调用函数时，这一组选定的伪随机数不再改变
	add_SNR = func_get_SNR(analy_ut, white_noise, disp_updated)


	analy_utn = func_add_noise(analy_ut, white_noise, add_SNR)	
	analy_vtn = func_diff_2point(analy_t, analy_utn)
	analy_atn = func_diff_2point(analy_t, analy_vtn)
	
	print('analy_utn_SNR=', func_SNR(analy_utn))
	print('analy_vtn_SNR=', func_SNR(analy_vtn))
	print('analy_atn_SNR=', func_SNR(analy_atn))

	# 绘制数据延拓结果
	n_utn_pad = int(0.5/0.002)
	analy_utn_pad = np.concatenate((np.zeros(n_utn_pad), analy_utn),axis=0)
	plt.plot(analy_t, analy_utn,'-')
	plt.show()

	#plt.plot(time_updated,analy_ut-1.75, label="analy_ut")
	#plt.plot(time_updated,analy_uta-1.75, label="analy_uta")
	#plt.plot(time_updated, disp_updated, label="tracking Data")
	fc_gauss0 = 1/(2*np.pi)*np.sqrt(2/np.pi)
	fc_gauss1 = 1/(1*np.pi)*np.sqrt(2/np.pi)
	fc_gauss2 = 4/(3*np.pi)*np.sqrt(2/np.pi)
	fc_gauss3 = 8/(5*np.pi)*np.sqrt(2/np.pi)
###########################################################################################################################
# 该部分对解析信号进行小波卷积与幅值因子求解，由于解析解记为真实信号，故直接计算真实解与小波卷积、有限差分结果之间的欧氏距离
	analy_utn_conv0 = func_conv_gauss_wave(analy_utn_pad, scale*fc_gauss0/fc_gauss0)[0][n_utn_pad:-n_utn_pad]
	analy_utn_conv1 = func_conv_gauss_wave(analy_utn_pad, scale*fc_gauss1/fc_gauss0)[1][n_utn_pad:-n_utn_pad]
	analy_utn_conv2 = func_conv_gauss_wave(analy_utn_pad, scale*fc_gauss2/fc_gauss0)[2][n_utn_pad:-n_utn_pad]  # 手动生成高斯小波函数族,并与信号进行卷积

	Amp0_analy_utn, ED0_analy_utn, Amp0_analy = func_BinarySearch_ED(analy_ut[:-n_utn_pad], analy_utn_conv0, 1e-10)
	Amp1_analy_utn, ED1_analy_utn, Amp1_analy = func_BinarySearch_ED(analy_vt[:-n_utn_pad], analy_utn_conv1, 1e-10)
	Amp2_analy_utn, ED2_analy_utn, Amp2_analy = func_BinarySearch_ED(analy_at[:-n_utn_pad], analy_utn_conv2, 1e-10)
###########################################################################################################################

###########################################################################################################################
	# 处理试验捕捉的含噪信号
	# 对含噪信号进行高斯小波卷积

	test_utn = disp_updated
	test_vtn = func_diff_2point(time_updated, test_utn)
	test_atn = func_diff_2point(time_updated, test_vtn)

	SNR_utn = func_SNR(test_utn)
	SNR_vtn = func_SNR(test_vtn)
	SNR_atn = func_SNR(test_atn)

	test_time = time_updated[n_add:-n_add]
	test_utn_conv0 = func_conv_gauss_wave(test_utn, scale*fc_gauss0/fc_gauss0)[0][n_add:-n_add]
	test_utn_conv1 = func_conv_gauss_wave(test_utn, scale*fc_gauss1/fc_gauss0)[1][n_add:-n_add]
	test_utn_conv2 = func_conv_gauss_wave(test_utn, scale*fc_gauss2/fc_gauss0)[2][n_add:-n_add]  # 手动生成高斯小波函数族,并与信号进行卷积
	test_utn_conv3 = func_conv_gauss_wave(test_utn, scale*fc_gauss3/fc_gauss0)[3][n_add:-n_add]  # 手动生成高斯小波函数族,并与信号进行卷积
	print(len(time_updated),len(disp_updated))

	# 实际信号并无真实解，需要对含噪信号高斯小波卷积结果反向积分，通过积分-微分之间的自洽性验证结果的准确性，积分时需要输入初始条件
	integral_test_utn_conv1 = func_integral_trapozoidal_rule(test_time, test_utn_conv1, 0)  # 梯形法则一次积分，初始条件为0。
	integral_test_utn_conv2 = func_integral_trapozoidal_rule(test_time, test_utn_conv2, 0)  # 梯形法则再次积分，初始条件为0。
	integral_test_utn_conv3 = func_integral_trapozoidal_rule(test_time, test_utn_conv3, 0)  # 梯形法则再次积分，初始条件为0。

	test_source = test_utn[n_add:-n_add]

	Amp0_test_utn, ED0_test_utn, Amp0_convol = func_BinarySearch_ED(test_source, test_utn_conv0, 1e-10)
	Amp1_test_utn, ED1_test_utn, Amp1_convol = func_BinarySearch_ED(Amp0_convol, integral_test_utn_conv1, 1e-10)
	Amp2_test_utn, ED2_test_utn, Amp2_convol = func_BinarySearch_ED(Amp1_test_utn*test_utn_conv1, integral_test_utn_conv2, 1e-10)
	Amp3_test_utn, ED3_test_utn, Amp3_convol = func_BinarySearch_ED(Amp2_test_utn*test_utn_conv2, integral_test_utn_conv3, 1e-10)

	# 绘制试验数据小波微分-积分迭代结果
	plt.plot(Amp2_test_utn*test_utn_conv2,label = 'Amp2_test_utn*')
	plt.plot(Amp3_convol,label = 'Amp3_convol')
	plt.legend()
	plt.show()

###########################################################################################################################
	# 绘制数值微分与小波微分对比图
	smoothWT_ori_SNR = func_SNR(Amp0_analy_utn*analy_utn_conv0)
	smoothWT_1st_SNR = func_SNR(Amp1_analy_utn*analy_utn_conv1)
	smoothWT_2nd_SNR = func_SNR(Amp2_analy_utn*analy_utn_conv2)
	print('smoothWT_ori_SNR=', smoothWT_ori_SNR)
	print('smoothWT_1st_SNR=', smoothWT_1st_SNR)
	print('smoothWT_2nd_SNR=', smoothWT_2nd_SNR)

	smoothWT_ori_ED = np.linalg.norm(Amp0_analy_utn*analy_utn_conv0 - analy_ut[:-n_utn_pad])
	smoothWT_1st_ED = np.linalg.norm(Amp1_analy_utn*analy_utn_conv1 - analy_vt[:-n_utn_pad])
	smoothWT_2nd_ED = np.linalg.norm(Amp2_analy_utn*analy_utn_conv2 - analy_at[:-n_utn_pad])
	print('smoothWT_ori_ED=', smoothWT_ori_ED)
	print('smoothWT_1st_ED=', smoothWT_1st_ED)
	print('smoothWT_2nd_ED=', smoothWT_2nd_ED)

	smoothWT_ori_Err = np.amax(Amp0_analy_utn*analy_utn_conv0 - analy_ut[:-n_utn_pad])/np.amax(analy_ut[:-n_utn_pad])*100
	smoothWT_1st_Err = np.amax(Amp1_analy_utn*analy_utn_conv1 - analy_vt[:-n_utn_pad])/np.amax(analy_vt[:-n_utn_pad])*100
	smoothWT_2nd_Err = np.amax(Amp2_analy_utn*analy_utn_conv2 - analy_at[:-n_utn_pad])/np.amax(analy_at[:-n_utn_pad])*100
	print('smoothWT_ori_Err=', smoothWT_ori_Err)
	print('smoothWT_1st_Err=', smoothWT_1st_Err)
	print('smoothWT_2nd_Err=', smoothWT_2nd_Err)

	plt.subplot(3,3,1)
	plt.plot(analy_t[:-n_utn_pad], analy_utn[:-n_utn_pad],label = 'analy_utn')
	plt.plot(analy_t[:-n_utn_pad], Amp0_analy_utn*analy_utn_conv0,label = 'Amp*conv0')
	plt.plot(analy_t[:-n_utn_pad], analy_ut[:-n_utn_pad],label = 'analy_ut')
	plt.legend(loc="best",fontsize=8)

	plt.subplot(3,3,2)
	plt.plot(analy_t[:-n_utn_pad], analy_vtn[:-n_utn_pad],label = 'analy_vtn')
	plt.plot(analy_t[:-n_utn_pad], Amp1_analy_utn*analy_utn_conv1,label = 'Amp1*conv1')
	plt.plot(analy_t[:-n_utn_pad], analy_vt[:-n_utn_pad],label = 'analy_vt')
	plt.legend(loc="best",fontsize=8)

	plt.subplot(3,3,3)
	plt.plot(analy_t[:-n_utn_pad], analy_atn[:-n_utn_pad],label = 'analy_atn')
	plt.plot(analy_t[:-n_utn_pad], Amp2_analy_utn*analy_utn_conv2,label = 'Amp2*conv2')
	plt.plot(analy_t[:-n_utn_pad], analy_at[:-n_utn_pad],label = 'analy_at')
	plt.legend(loc="best",fontsize=8)

	plt.subplot(3,3,4)
	plt.plot(test_time, test_utn[n_add:-n_add])
	plt.legend(loc="best",fontsize=8)

	plt.subplot(3,3,5)
	plt.plot(test_time, test_vtn[n_add:-n_add])
	plt.legend(loc="best",fontsize=8)

	plt.subplot(3,3,6)
	plt.plot(test_time, test_atn[n_add:-n_add])
	plt.legend(loc="best",fontsize=8)

	plt.subplot(3,3,7)
	plt.plot(test_time, Amp0_test_utn*test_utn_conv0)
	plt.legend(loc="best",fontsize=8)

	plt.subplot(3,3,8)
	plt.plot(test_time, Amp1_test_utn*test_utn_conv1)
	plt.legend(loc="best",fontsize=8)

	plt.subplot(3,3,9)
	plt.plot(test_time, Amp2_test_utn*test_utn_conv2)
	plt.legend(loc="best",fontsize=8)
	plt.show()

	# Amp, DTWDist = func_BinarySearch_DTW(signal_handle,myconv0, 1e-5)
	# print('The Dynamic Time Warping distance between original signal and handled signal = ', EDist)	
	wave_use = 'gaus2'
	target_freq = np.arange(1/total_time,sample_rate/2, step=2)  # Nquist-Shannon采样定理
	freq_c = pywt.central_frequency(wave_use)
	scale_use = freq_c*sample_rate/target_freq
	# scale_use = scale_use[::-1]
	cwtmatr, freqs = pywt.cwt(disp_updated, scale_use, wave_use, sampling_period=0.002, method='conv')  #小波分解
	# print(type(cwtmatr),cwtmatr.shape)
	energys = []
	for i in range (cwtmatr.shape[0]):
		energys.append(np.linalg.norm(cwtmatr[i],ord=2))

	index = freqs  # 包含每个柱子下标的序列
	values = np.array(energys)  # 柱子的高度
	width = 5  # 柱子的宽度
	# 绘制柱状图, 每根柱子的颜色为紫罗兰色
	#plt.bar(index, values, width, label="num", color="#87CEFA")

	#tail_index = int(len(disp_updated)/100)
	#handle = signal_handle[tail_index:-tail_index]
	#handled = signal_handled[tail_index:-tail_index]

	#print('SNR of the handle signal is', func_SNR(disp_updated))
	#print('SNR of the handled signal is', func_SNR(Amp*myconv0))
########################################################################################
#### 打印及导出数据
	print('SNR of noise-added analytical displacement = ', func_SNR(analy_utn[n_add:-n_add]))
	print('SNR of noise-added analytical velocity = ', func_SNR(analy_vtn[n_add:-n_add]))
	print('SNR of noise-added analytical acceleration = ', func_SNR(analy_atn[n_add:-n_add]))
#
	#print('ED between analy disp and noise-added analy disp = ', np.linalg.norm(analy_utn[n_add:-n_add]-analy_ut[n_add:-n_add]))
	#print('ED between analy velo and noise-added analy velo = ', np.linalg.norm(analy_vtn[n_add:-n_add]-analy_vt[n_add:-n_add]))
	#print('ED between analy acce and noise-added analy acce = ', np.linalg.norm(analy_atn[n_add:-n_add]-analy_at[n_add:-n_add]))
#
	#print('SNR of tracked displacement = ', func_SNR(test_utn[n_add:-n_add]))
	#print('SNR of finite-diff velocity = ', func_SNR(test_vtn[n_add:-n_add]))
	#print('SNR of finite-diff acceleration = ', func_SNR(test_atn[n_add:-n_add]))
#
	#print('SNR of Gaussian displacement = ', func_SNR(Amp0_test_utn*test_utn_conv0))
	#print('SNR of Gaussian velocity = ', func_SNR(Amp1_test_utn*test_utn_conv1))
	#print('SNR of Gaussian acceleration = ', func_SNR(Amp2_test_utn*test_utn_conv2))
#
	#print('ED between test disp and gauss disp = ', ED0_test_utn)
	#print('ED between test velo and gauss velo = ', ED1_test_utn)
	#print('ED between test acce and gauss acce = ', ED2_test_utn)

	#plt.plot(analy_t,analy_ut)
	#plt.plot(analy_t,analy_ut)
	#plt.plot(analy_t,analy_vt)
	#plt.plot(analy_t,analy_at)

	np.savetxt('analy_t.txt', analy_t[:-n_utn_pad])
	np.savetxt('analy_ut.txt', analy_ut[:-n_utn_pad])
	np.savetxt('analy_vt.txt', analy_vt[:-n_utn_pad])
	np.savetxt('analy_at.txt', analy_at[:-n_utn_pad])

	np.savetxt('analy_utn.txt', analy_utn[:-n_utn_pad])
	np.savetxt('analy_vtn.txt', analy_vtn[:-n_utn_pad])
	np.savetxt('analy_atn.txt', analy_atn[:-n_utn_pad])

	np.savetxt('test_time.txt', test_time)
	np.savetxt('Amp0_test_utn.txt', Amp0_test_utn*test_utn_conv0)
	np.savetxt('Amp1_test_utn.txt', Amp1_test_utn*test_utn_conv1)
	np.savetxt('Amp2_test_utn.txt', Amp2_test_utn*test_utn_conv2)

	'''
	# 绘制卷积运算、pywt计算结果
	# 可以发现选择高斯小波家族时，计算结果差负号
	plt.plot(-cwtmatr[0],label = 'signal_handle')
	plt.plot(signal_handled,label = 'signal_handled')
	'''

	plt.show()
