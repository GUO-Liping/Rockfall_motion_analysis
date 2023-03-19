#!/usr/bin/env python    
#encoding: utf-8 
# 该程序用于冲击试验速度时程曲线小波降噪，旨在获得可靠的加速度值用于预测落石冲击力
import numpy as np
import pywt
import matplotlib.pyplot as plt
from user_func_package import *
import random

if __name__ == '__main__':
	time_test = np.array([0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02,0.022,0.024,0.026,0.028,0.03,0.032,0.034,0.036,0.038,0.04,0.042,0.044,0.046,0.048,0.05,0.052,0.054,0.056,0.058,0.06,0.062,0.064,0.066,0.068,0.07,0.072,0.074,0.076,0.078,0.08,0.082,0.084,0.086,0.088,0.09,0.092,0.094,0.096,0.098,0.1,0.102,0.104,0.106,0.108,0.11,0.112,0.114,0.116,0.118,0.12,0.122,0.124,0.126,0.128,0.13,0.132,0.134,0.136,0.138,0.14,0.142,0.144,0.146,0.148,0.15,0.152,0.154,0.156,0.158,0.16,0.162,0.164,0.166,0.168,0.17,0.172,0.174,0.176,0.178,0.18,0.182,0.184,0.186,0.188,0.19,0.192,0.194,0.196,0.198,0.2,0.202,0.204,0.206,0.208,0.21,0.212,0.214,0.216,0.218,0.22,0.222,0.224,0.226,0.228,0.23,0.232,0.234,0.236,0.238,0.24,0.242,0.244,0.246,0.248,0.25,0.252,0.254,0.256,0.258,0.26,0.262,0.264,0.266,0.268,0.27,0.272,0.274,0.276,0.278,0.28,0.282,0.284,0.286,0.288,0.29,0.292,0.294,0.296,0.298,0.3,0.302,0.304,0.306,0.308,0.31,0.312,0.314,0.316,0.318,0.32,0.322,0.324,0.326,0.328,0.33,0.332,0.334,0.336,0.338,0.34,0.342,0.344,0.346,0.348,0.35,0.352,0.354,0.356,0.358,0.36,0.362,0.364,0.366,0.368,0.37,0.372,0.374,0.376,0.378,0.38,0.382,0.384,0.386,0.388,0.39,0.392,0.394,0.396,0.398,0.4,0.402,0.404,0.406,0.408,0.41,0.412,0.414,0.416,0.418,0.42,0.422,0.424,0.426,0.428,0.43,0.432,0.434,0.436,0.438,0.44,0.442,0.444,0.446,0.448,0.45,0.452,0.454,0.456,0.458,0.46,0.462,0.464,0.466,0.468,0.47,0.472,0.474,0.476,0.478,0.48,0.482,0.484,0.486,0.488,0.49,0.492,0.494,0.496,0.498,0.5,0.502,0.504,0.506,0.508,0.51,0.512,0.514,0.516,0.518,0.52,0.522,0.524,0.526,0.528,0.53,0.532,0.534,0.536,0.538,0.54,0.542,0.544,0.546,0.548,0.55,0.552,0.554,0.556,0.558,0.56,0.562,0.564,0.566,0.568,0.57,0.572,0.574,0.576,0.578,0.58,0.582,0.584,0.586,0.588,0.59,0.592,0.594,0.596,0.598,0.6,0.602,0.604,0.606,0.608,0.61,0.612,0.614,0.616,0.618,0.62,0.622,0.624,0.626,0.628,0.63,0.632,0.634,0.636,0.638,0.64,0.642,0.644,0.646,0.648,0.65,0.652,0.654,0.656,0.658,0.66,0.662,0.664,0.666,0.668,0.67,0.672,0.674,0.676,0.678,0.68,0.682,0.684,0.686,0.688,0.69,0.692,0.694,0.696,0.698,0.7,0.702,0.704,0.706,0.708,0.71,0.712,0.714,0.716,0.718,0.72,0.722,0.724,0.726,0.728,0.73,0.732,0.734,0.736,0.738,0.74,0.742,0.744,0.746,0.748,0.75,0.752,0.754,0.756,0.758,0.76,0.762,0.764,0.766,0.768,0.77,0.772,0.774,0.776,0.778,0.78,0.782,0.784,0.786,0.788,0.79,0.792,0.794,0.796,0.798,0.8,0.802,0.804,0.806,0.808,0.81,0.812,0.814,0.816,0.818,0.82,0.822,0.824,0.826,0.828,0.83,0.832,0.834,0.836,0.838,0.84,0.842,0.844,0.846,0.848,0.85,0.852,0.854,0.856,0.858,0.86,0.862,0.864,0.866,0.868,0.87,0.872,0.874,0.876,0.878,0.88,0.882,0.884,0.886,0.888,0.89,0.892,0.894,0.896,0.898,0.9,0.902,0.904,0.906,0.908,0.91,0.912,0.914,0.916,0.918,0.92,0.922,0.924,0.926,0.928,0.93,0.932,0.934,0.936,0.938,0.94,0.942,0.944,0.946,0.948,0.95,0.952,0.954,0.956,0.958,0.96,0.962,0.964,0.966,0.968,0.97,0.972,0.974,0.976,0.978,0.98,0.982,0.984,0.986,0.988,0.99,0.992,0.994,0.996,0.998,1,1.002,1.004,1.006,1.008,1.01,1.012,1.014,1.016,1.018,1.02,1.022,1.024,1.026,1.028,1.03,1.032,1.034,1.036,1.038,1.04,1.042,1.044,1.046,1.048,1.05,1.052,1.054,1.056,1.058,1.06,1.062,1.064,1.066,1.068,1.07,1.072,1.074,1.076,1.078,1.08,1.082,1.084,1.086,1.088,1.09,1.092,1.094,1.096,1.098,1.1,1.102,1.104,1.106])
	disp_test = np.array([0,-0.02095,-0.0369,-0.05785,-0.0778,-0.09276,-0.10772,-0.12667,-0.15011,-0.17255,-0.1925,-0.20896,-0.22741,-0.24836,-0.26581,-0.28476,-0.30272,-0.32616,-0.34361,-0.35857,-0.37603,-0.39697,-0.41892,-0.43837,-0.45732,-0.47876,-0.49671,-0.51766,-0.5391,-0.55905,-0.57601,-0.59646,-0.61541,-0.63585,-0.6578,-0.68373,-0.70318,-0.72014,-0.74108,-0.75854,-0.77948,-0.79893,-0.81689,-0.83983,-0.86077,-0.88222,-0.90017,-0.92311,-0.94356,-0.9645,-0.98445,-1.0064,-1.02584,-1.04579,-1.06674,-1.08918,-1.11013,-1.13307,-1.15202,-1.17446,-1.19441,-1.21535,-1.2358,-1.25774,-1.27969,-1.30063,-1.32108,-1.34302,-1.36547,-1.38442,-1.40586,-1.4308,-1.45573,-1.47718,-1.50461,-1.53054,-1.54799,-1.56694,-1.59138,-1.61133,-1.63427,-1.65272,-1.67267,-1.69412,-1.71456,-1.73501,-1.75645,-1.7779,-1.79585,-1.8158,-1.83375,-1.85071,-1.86767,-1.88313,-1.89809,-1.91006,-1.92003,-1.93,-1.93848,-1.94746,-1.95245,-1.95843,-1.96292,-1.9689,-1.9719,-1.97788,-1.98337,-1.98835,-1.99284,-1.99833,-2.00481,-2.0103,-2.01528,-2.02077,-2.02725,-2.03423,-2.04172,-2.0477,-2.05269,-2.05967,-2.06366,-2.06964,-2.07762,-2.0846,-2.09009,-2.09807,-2.10405,-2.10954,-2.11552,-2.12101,-2.12849,-2.13497,-2.14096,-2.14644,-2.15343,-2.16141,-2.16889,-2.17637,-2.18534,-2.19282,-2.19981,-2.20629,-2.21377,-2.22025,-2.22724,-2.23422,-2.2412,-2.24818,-2.25417,-2.26065,-2.26763,-2.27411,-2.2811,-2.28858,-2.29556,-2.30404,-2.31401,-2.32149,-2.32997,-2.33795,-2.34693,-2.35341,-2.36139,-2.36687,-2.37336,-2.38134,-2.38981,-2.39829,-2.40577,-2.41525,-2.42423,-2.4347,-2.44368,-2.45215,-2.45963,-2.46811,-2.47709,-2.48457,-2.49255,-2.49953,-2.50801,-2.51599,-2.52397,-2.53195,-2.53971,-2.54773,-2.5607,-2.57018,-2.58015,-2.58863,-2.5981,-2.60608,-2.61356,-2.62254,-2.63152,-2.64099,-2.64997,-2.65945,-2.66792,-2.6764,-2.68488,-2.69336,-2.70283,-2.71231,-2.72278,-2.73375,-2.74273,-2.7537,-2.76318,-2.77066,-2.78013,-2.7916,-2.79859,-2.81006,-2.82053,-2.831,-2.83898,-2.85045,-2.85993,-2.8684,-2.88037,-2.88935,-2.89883,-2.9088,-2.92027,-2.93024,-2.94122,-2.95219,-2.96116,-2.96914,-2.97962,-2.98959,-3.00056,-3.01004,-3.01951,-3.03048,-3.04295,-3.05442,-3.0639,-3.07537,-3.08534,-3.09631,-3.11028,-3.12225,-3.13571,-3.14768,-3.15766,-3.16613,-3.17461,-3.18459,-3.19556,-3.20753,-3.2195,-3.22997,-3.23994,-3.25441,-3.26588,-3.27435,-3.28433,-3.2953,-3.30677,-3.31974,-3.33171,-3.34367,-3.35564,-3.37011,-3.38507,-3.39803,-3.412,-3.42397,-3.43594,-3.45169,-3.46765,-3.47863,-3.49309,-3.50705,-3.51902,-3.52949,-3.54246,-3.55243,-3.5669,-3.57986,-3.59034,-3.60031,-3.61228,-3.62425,-3.63672,-3.64868,-3.65966,-3.67312,-3.68758,-3.70205,-3.71701,-3.73197,-3.74444,-3.75591,-3.76788,-3.78134,-3.79281,-3.80428,-3.81825,-3.83221,-3.84568,-3.85864,-3.8741,-3.88457,-3.89754,-3.91001,-3.92497,-3.93744,-3.94741,-3.96138,-3.97384,-3.98581,-3.99978,-4.01524,-4.0282,-4.04566,-4.05912,-4.07109,-4.08555,-4.09852,-4.11099,-4.12495,-4.13493,-4.14839,-4.15837,-4.17482,-4.19278,-4.20824,-4.22619,-4.24165,-4.25711,-4.26609,-4.27955,-4.29252,-4.30848,-4.32843,-4.3369,-4.34837,-4.36383,-4.38079,-4.39426,-4.40722,-4.42218,-4.43665,-4.4531,-4.46707,-4.48103,-4.49599,-4.50896,-4.52392,-4.53738,-4.54886,-4.55783,-4.56631,-4.58127,-4.59075,-4.60022,-4.6092,-4.61768,-4.62715,-4.63264,-4.63613,-4.63962,-4.64112,-4.64361,-4.6471,-4.6486,-4.6476,-4.64959,-4.65009,-4.6486,-4.6476,-4.6486,-4.6476,-4.6476,-4.6486,-4.6486,-4.6486,-4.64959,-4.6476,-4.64511,-4.64511,-4.6471,-4.6461,-4.6476,-4.6461,-4.64461,-4.6461,-4.64361,-4.64511,-4.64511,-4.64361,-4.64211,-4.64461,-4.64211,-4.63862,-4.64112,-4.64062,-4.63713,-4.63812,-4.63713,-4.63613,-4.63613,-4.63364,-4.63114,-4.62965,-4.63364,-4.63364,-4.63114,-4.62665,-4.62965,-4.62965,-4.62715,-4.62416,-4.62416,-4.62416,-4.62167,-4.62067,-4.61917,-4.61668,-4.61718,-4.61319,-4.61818,-4.61967,-4.61818,-4.61419,-4.61419,-4.61169,-4.61468,-4.61169,-4.6102,-4.61169,-4.61169,-4.6092,-4.61219,-4.6102,-4.6092,-4.6102,-4.6077,-4.6077,-4.6082,-4.6077,-4.6082,-4.60571,-4.60671,-4.60321,-4.60571,-4.60571,-4.60172,-4.60172,-4.60421,-4.60321,-4.60321,-4.60072,-4.60072,-4.60022,-4.60072,-4.60272,-4.60022,-4.60022,-4.60022,-4.60172,-4.60022,-4.59922,-4.60022,-4.59773,-4.59823,-4.59922,-4.60022,-4.59922,-4.60022,-4.60072,-4.60072,-4.60172,-4.60022,-4.60072,-4.60022,-4.59922,-4.60022,-4.60072,-4.60072,-4.60022,-4.60172,-4.60321,-4.60172,-4.60072,-4.60072,-4.60072,-4.59823,-4.60172,-4.60072,-4.60172,-4.60321,-4.6082,-4.60671,-4.6082,-4.60421,-4.6082,-4.6107,-4.6082,-4.6077,-4.6082,-4.61319,-4.61419,-4.61319,-4.6102,-4.6092,-4.6102,-4.6102,-4.6092,-4.6107,-4.61568,-4.61468,-4.61668,-4.61917,-4.61818,-4.61818,-4.61668,-4.62067,-4.61917,-4.61718,-4.61967,-4.61967,-4.61818,-4.62217,-4.62466,-4.62167,-4.61917,-4.62067,-4.62566,-4.62067,-4.62217,-4.62217,-4.62217,-4.62217,-4.62167,-4.62067,-4.62217,-4.62067,-4.61967,-4.62167,-4.61917,-4.61718,-4.61917,-4.61917,-4.61967,-4.61568,-4.62217,-4.61917,-4.61967,-4.61668,-4.61818])
	sample_rate = 250
	time_updated, disp_updated = func_update_disp(time_test,disp_test, sample_rate)  # 更新采样频率至500Hz水平

	# scale = fc/f_pseudo*sample_rate，其中f_pseudo为傅里叶变换得到的伪频率
	scale =3  # 小波函数尺度参数 T=0.094s, fs=500Hz，伪中心频率0.12699对应的尺度参数为5.96853
	#key_i = int((len(time_updated)-2*n_add-1)*0.5)  # 关键索引，便于求解小波变换幅值参数0.918for12,0.79 fors=6

	# 边缘效应处理方法：pading，即向数据两段人工添加数据，小波变换后在除去这些数据
	#time_updated1 = pywt.pad(time_updated0,(0,500),'zero')

	n_fit = int(0.03*len(disp_updated))	# 第一个常数，表示用于待处理数据中可用于抛物线拟合的捕捉数据点数量
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
