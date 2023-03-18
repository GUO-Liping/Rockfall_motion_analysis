#!/usr/bin/env python    
#encoding: utf-8 
# 该程序用于冲击试验速度时程曲线小波降噪，旨在获得可靠的加速度值用于预测落石冲击力
import numpy as np
import pywt
import matplotlib.pyplot as plt
from user_func_package import *
import random

if __name__ == '__main__':
	time_test = np.array([0,0.008,0.016,0.024,0.032,0.04,0.048,0.056,0.064,0.072,0.08,0.088,0.096,0.104,0.112,0.12,0.128,0.136,0.144,0.152,0.16,0.168,0.176,0.184,0.192,0.2,0.208,0.216,0.224,0.232,0.24,0.248,0.256,0.264,0.272,0.28,0.288,0.296,0.304,0.312,0.32,0.328,0.336,0.344,0.352,0.36,0.368,0.376,0.384,0.392,0.4,0.408,0.416,0.424,0.432,0.44,0.448,0.456,0.464,0.472,0.48,0.488,0.496,0.504,0.512,0.52,0.528,0.536,0.544,0.552,0.56,0.568,0.576,0.584,0.592,0.6,0.608,0.616,0.624,0.632,0.64,0.648,0.656,0.664,0.672,0.68,0.688,0.696,0.704,0.712,0.72,0.728,0.736,0.744,0.752,0.76,0.768,0.776,0.784,0.792,0.8,0.808,0.816,0.824,0.832,0.84,0.848,0.856,0.864,0.872,0.88,0.888,0.896,0.904,0.912,0.92,0.928,0.936,0.944,0.952,0.96,0.968,0.976,0.984,0.992,1,1.008,1.016,1.024,1.032,1.04,1.048,1.056,1.064,1.072,1.08,1.088,1.096,1.104,1.112,1.12,1.128,1.136,1.144,1.152,1.16,1.168,1.176,1.184,1.192,1.2,1.208,1.216,1.224,1.232,1.24,1.248,1.256,1.264,1.272,1.28,1.288,1.296,1.304,1.312,1.32,1.328,1.336,1.344,1.352,1.36,1.368,1.376,1.384,1.392,1.4,1.408,1.416,1.424,1.432,1.44,1.448,1.456,1.464,1.472,1.48,1.488,1.496,1.504,1.512,1.52,1.528,1.536,1.544,1.552,1.56,1.568,1.576,1.584,1.592,1.6,1.608,1.616,1.624,1.632,1.64,1.648,1.656,1.664,1.672,1.68,1.688,1.696,1.704,1.712,1.72,1.728,1.736,1.744,1.752,1.76,1.768,1.776,1.784,1.792,1.8,1.808,1.816,1.824,1.832,1.84,1.848,1.856,1.864,1.872,1.88,1.888,1.896,1.904,1.912,1.92,1.928,1.936,1.944,1.952,1.96,1.968,1.976,1.984,1.992,2,2.008,2.016,2.024,2.032,2.04,2.048,2.056,2.064,2.072,2.08,2.088,2.096,2.104,2.112,2.12,2.128,2.136,2.144,2.152,2.16,2.168,2.176,2.184,2.192,2.2,2.208,2.216,2.224,2.232,2.24,2.248,2.256,2.264,2.272,2.28,2.288,2.296,2.304,2.312,2.32,2.328,2.336,2.344,2.352,2.36,2.368,2.376,2.384,2.392,2.4,2.408,2.416,2.424,2.432,2.44,2.448,2.456,2.464,2.472,2.48,2.488,2.496,2.504,2.512,2.52,2.528,2.536,2.544,2.552,2.56,2.568,2.576,2.584,2.592,2.6,2.608,2.616,2.624,2.632,2.64,2.648,2.656,2.664,2.672,2.68,2.688,2.696,2.704,2.712,2.72,2.728,2.736,2.744,2.752,2.76,2.768,2.776,2.784,2.792,2.8,2.808,2.816,2.824,2.832,2.84,2.848,2.856,2.864,2.872,2.88,2.888,2.896,2.904,2.912,2.92,2.928,2.936,2.944,2.952,2.96,2.968,2.976,2.984,2.992,3,3.008,3.016,3.024,3.032,3.04,3.048,3.056,3.064,3.072,3.08,3.088,3.096,3.104,3.112,3.12,3.128,3.136,3.144,3.152,3.16,3.168,3.176,3.184,3.192,3.2,3.208,3.216,3.224,3.232,3.24,3.248,3.256,3.264,3.272,3.28,3.288,3.296,3.304,3.312,3.32,3.328,3.336,3.344,3.352,3.36,3.368,3.376,3.384,3.392,3.4,3.408,3.416,3.424,3.432,3.44,3.448,3.456,3.464,3.472,3.48,3.488,3.496,3.504,3.512,3.52,3.528,3.536,3.544,3.552,3.56,3.568,3.576,3.584,3.592,3.6,3.608,3.616,3.624,3.632,3.64,3.648,3.656,3.664,3.672,3.68,3.688,3.696,3.704,3.712,3.72,3.728,3.736,3.744,3.752,3.76,3.768,3.776,3.784,3.792,3.8,3.808,3.816,3.824,3.832,3.84,3.848,3.856,3.864,3.872,3.88,3.888,3.896,3.904,3.912,3.92,3.928,3.936,3.944,3.952,3.96,3.968,3.976,3.984,3.992,4,4.008,4.016,4.024,4.032,4.04,4.048,4.056,4.064,4.072,4.08,4.088,4.096,4.104,4.112,4.12,4.128,4.136,4.144,4.152,4.16,4.168,4.176,4.184,4.192,4.2,4.208,4.216,4.224,4.232,4.24,4.248,4.256,4.264,4.272,4.28,4.288,4.296,4.304,4.312,4.32,4.328,4.336,4.344,4.352,4.36,4.368,4.376,4.384,4.392,4.4,4.408,4.416,4.424,4.426])
	disp_test = np.array([-0.00148,-0.07044,-0.13865,-0.20463,-0.27433,-0.33883,-0.41149,-0.48267,-0.54939,-0.61538,-0.68211,-0.75625,-0.83113,-0.90602,-0.97645,-1.04837,-1.12103,-1.19369,-1.26264,-1.33159,-1.39239,-1.45764,-1.52585,-1.58145,-1.63854,-1.68006,-1.71565,-1.74012,-1.74679,-1.74382,-1.72232,-1.6986,-1.67117,-1.64447,-1.61037,-1.577,-1.54438,-1.51028,-1.48359,-1.448,-1.41537,-1.3872,-1.35161,-1.32492,-1.29526,-1.26709,-1.24262,-1.21297,-1.18479,-1.15069,-1.12696,-1.09582,-1.08025,-1.05134,-1.03058,-1.00908,-0.98387,-0.96088,-0.94235,-0.91936,-0.90379,-0.88378,-0.86524,-0.8467,-0.83113,-0.81556,-0.79851,-0.78294,-0.76441,-0.7518,-0.73846,-0.72437,-0.71028,-0.69916,-0.69175,-0.67766,-0.66654,-0.65468,-0.64578,-0.63836,-0.62873,-0.62279,-0.61464,-0.60648,-0.60055,-0.59684,-0.59314,-0.59165,-0.58869,-0.58498,-0.58276,-0.5835,-0.58572,-0.58498,-0.58498,-0.5835,-0.58721,-0.59091,-0.59536,-0.60055,-0.59907,-0.60648,-0.61241,-0.61686,-0.62502,-0.63243,-0.64059,-0.65023,-0.66209,-0.67025,-0.68137,-0.69101,-0.70139,-0.7177,-0.73327,-0.74365,-0.75551,-0.77182,-0.78591,-0.8037,-0.81779,-0.83632,-0.85189,-0.86598,-0.88229,-0.90009,-0.91862,-0.94086,-0.96459,-0.98683,-1.00314,-1.02539,-1.05134,-1.07506,-1.09953,-1.1277,-1.15365,-1.1796,-1.20778,-1.23447,-1.26412,-1.29452,-1.31677,-1.34865,-1.37682,-1.4087,-1.44058,-1.46653,-1.49916,-1.51917,-1.54735,-1.58368,-1.60147,-1.62594,-1.6378,-1.6556,-1.66968,-1.67413,-1.67191,-1.66598,-1.64966,-1.63558,-1.61927,-1.60147,-1.58145,-1.56737,-1.54957,-1.52511,-1.50064,-1.48284,-1.45912,-1.44058,-1.42279,-1.39832,-1.38053,-1.35606,-1.33604,-1.32196,-1.30564,-1.29007,-1.27969,-1.2619,-1.24336,-1.22928,-1.21964,-1.20629,-1.19221,-1.18479,-1.16922,-1.1581,-1.1492,-1.13808,-1.13067,-1.12103,-1.1151,-1.10991,-1.10694,-1.10398,-1.09805,-1.09508,-1.08841,-1.08544,-1.08396,-1.08692,-1.08544,-1.08396,-1.08396,-1.08396,-1.08692,-1.08692,-1.0936,-1.09805,-1.09953,-1.10546,-1.10991,-1.10991,-1.11658,-1.124,-1.13215,-1.13808,-1.14327,-1.15217,-1.16774,-1.18034,-1.19147,-1.20036,-1.21148,-1.22335,-1.23521,-1.25004,-1.2619,-1.27747,-1.29378,-1.30787,-1.32344,-1.34049,-1.35606,-1.37534,-1.39684,-1.41315,-1.42946,-1.44948,-1.47024,-1.49248,-1.51176,-1.53178,-1.54587,-1.56144,-1.57181,-1.5859,-1.60147,-1.61037,-1.61852,-1.62446,-1.62594,-1.6289,-1.62742,-1.62446,-1.61185,-1.60444,-1.59332,-1.58442,-1.5733,-1.55476,-1.53919,-1.52214,-1.50954,-1.49767,-1.48507,-1.47098,-1.45541,-1.43539,-1.42501,-1.41537,-1.40425,-1.39387,-1.38275,-1.37386,-1.3657,-1.35829,-1.34716,-1.3353,-1.32937,-1.32492,-1.31602,-1.31306,-1.30787,-1.30416,-1.29971,-1.29823,-1.29601,-1.29601,-1.29378,-1.29526,-1.29378,-1.29601,-1.29971,-1.30416,-1.30564,-1.31009,-1.31306,-1.32121,-1.32492,-1.32863,-1.33678,-1.34123,-1.34865,-1.35532,-1.36496,-1.37237,-1.38201,-1.39017,-1.40203,-1.41463,-1.4265,-1.43465,-1.44503,-1.45615,-1.46802,-1.47988,-1.49174,-1.50583,-1.51398,-1.52362,-1.53474,-1.54142,-1.55106,-1.55699,-1.56514,-1.5733,-1.57849,-1.57923,-1.58219,-1.58294,-1.58071,-1.57849,-1.5733,-1.57107,-1.5644,-1.55699,-1.55106,-1.54142,-1.534,-1.52659,-1.51769,-1.50657,-1.49916,-1.49174,-1.48581,-1.4784,-1.47098,-1.46357,-1.45986,-1.45245,-1.44726,-1.44207,-1.43465,-1.4265,-1.42131,-1.42279,-1.41908,-1.41686,-1.41389,-1.41241,-1.41093,-1.40944,-1.4087,-1.41018,-1.41093,-1.41241,-1.41537,-1.41834,-1.41834,-1.42353,-1.4265,-1.43243,-1.43688,-1.44207,-1.44651,-1.4517,-1.45615,-1.46134,-1.4695,-1.47543,-1.47988,-1.48655,-1.49545,-1.4999,-1.50657,-1.51102,-1.51621,-1.52362,-1.52881,-1.53326,-1.53623,-1.54142,-1.53919,-1.54364,-1.54364,-1.54438,-1.54957,-1.54809,-1.54957,-1.54735,-1.54438,-1.5429,-1.54216,-1.53993,-1.53771,-1.53178,-1.52585,-1.51992,-1.51621,-1.51028,-1.50731,-1.50583,-1.50212,-1.49693,-1.49248,-1.48655,-1.48136,-1.47765,-1.47617,-1.47321,-1.47024,-1.46802,-1.46653,-1.46802,-1.46579,-1.46431,-1.46283,-1.46208,-1.46505,-1.46727,-1.4695,-1.47098,-1.47172,-1.47321,-1.47617,-1.47691,-1.47765,-1.4821,-1.48655,-1.48878,-1.491,-1.49248,-1.49693,-1.49916,-1.50286,-1.50509,-1.50954,-1.51176,-1.51473,-1.51769,-1.51917,-1.52214,-1.52511,-1.52585,-1.52659,-1.52807,-1.52733,-1.52881,-1.52955,-1.53178,-1.534,-1.53474,-1.53252,-1.52955,-1.52659,-1.52511,-1.52288,-1.5214,-1.51843,-1.51547,-1.5125,-1.51176,-1.51028,-1.50879,-1.50657,-1.50435,-1.50286,-1.50064,-1.49916,-1.49693,-1.49545,-1.49248,-1.49026,-1.48952,-1.48878,-1.48803,-1.48655,-1.49026,-1.48729,-1.491,-1.48878,-1.48878,-1.48878,-1.49026,-1.48952,-1.48729,-1.48952,-1.49174,-1.49174,-1.49322,-1.49471,-1.49693,-1.49916,-1.4999,-1.5036,-1.50583,-1.50805,-1.50954,-1.51176,-1.51473,-1.51324,-1.51398,-1.51473,-1.51695,-1.51992,-1.51992,-1.52288,-1.52066,-1.51992,-1.51992,-1.51992,-1.52066,-1.51917,-1.51769,-1.51621,-1.51695,-1.51547,-1.5125,-1.51102,-1.51102,-1.51028,-1.51028,-1.50805,-1.50657,-1.5036,-1.50138,-1.4999,-1.49916,-1.49841,-1.49841,-1.49693,-1.49471,-1.49174,-1.49248,-1.49174,-1.49248,-1.49174,-1.49322,-1.49248,-1.49248,-1.49026])
	sample_rate = 250
	time_updated, disp_updated = func_update_disp(time_test,disp_test, sample_rate)  # 更新采样频率至500Hz水平

	# scale = fc/f_pseudo*sample_rate，其中f_pseudo为傅里叶变换得到的伪频率
	scale =5  # 小波函数尺度参数 T=0.094s, fs=500Hz，伪中心频率0.12699对应的尺度参数为5.96853
	#key_i = int((len(time_updated)-2*n_add-1)*0.5)  # 关键索引，便于求解小波变换幅值参数0.918for12,0.79 fors=6

	# 边缘效应处理方法：pading，即向数据两段人工添加数据，小波变换后在除去这些数据
	#time_updated1 = pywt.pad(time_updated0,(0,500),'zero')

	n_fit = int(0.05*len(disp_updated))	# 第一个常数，表示用于待处理数据中可用于抛物线拟合的捕捉数据点数量
	n_add = int(0.15*len(disp_updated))	# 第二个常数，表示在信号首尾端需要添加的数据点数量

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
