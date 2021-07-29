#!/usr/bin/env python    
#encoding: utf-8 
import numpy as np
import matplotlib.pyplot as plt
import random

# 根据Fourier变换结果对频域能量段按照频域能量比例分别为33%，66%，99%进行分割，并返回分割点的索引值
def func_freqs_divide(signal_data):
	n = len(signal_data)
	power_signal = pow(signal_data,2)

	total_energy = np.sum(power_signal)
	print(total_energy - pow(np.linalg.norm(signal_data,ord=2),2))
	sum_energy = 0
	i50, i90, i99 = 0,0,0
	for i in range(n):
		sum_energy = np.sum(power_signal[:i+1])
		ratio_s = sum_energy/total_energy
		print('ratio_s=',ratio_s)
		if ratio_s*100>=0 and ratio_s*100<=50:
			i50 = i
			print('i50=',i50)
			print('ratio_s50=',ratio_s)
		elif ratio_s*100>50 and ratio_s*100<=90:
			i90 = i
			print('i90=',i90)
			print('ratio_s90=',ratio_s)
		elif ratio_s*100>90 and ratio_s*100<=99:
			i99 = i
			print('i99=',i99)
			print('ratio_s99=',ratio_s)
		else:
			pass
	return i50, i90, i99

# 给出一个单自由度低阻尼体系动力学时程解析信号，参考《结构动力学》
def func_analytical_signal(time_updated):
	exam_t = time_updated
	exam_m = 5
	exam_k = 600
	exam_xi = 0.08
	exam_u0 = 0
	exam_v0 = -12.5
	exam_omega = np.sqrt(exam_k/exam_m)
	exam_omega_D = exam_omega*np.sqrt(1-exam_xi**2)

	exam_A = exam_u0
	exam_B = (exam_v0+exam_u0*exam_xi*exam_omega)/exam_omega_D
	exam_C = np.exp(-exam_xi*exam_omega*exam_t)

	exam_ut = (exam_A*np.cos(exam_omega_D*exam_t)+exam_B * np.sin(exam_omega_D*exam_t))*exam_C
	exam_vt = exam_ut*(-exam_xi*exam_omega) + (-exam_A*np.sin(exam_omega_D*exam_t)+exam_B*np.cos(exam_omega_D*exam_t))*exam_C*exam_omega_D
	exam_at = exam_vt*(-exam_xi*exam_omega) + (-exam_A*np.sin(exam_omega_D*exam_t)+exam_B*np.cos(exam_omega_D*exam_t))*exam_C*exam_omega_D*(-exam_xi*exam_omega) - (exam_A*np.cos(exam_omega_D*exam_t)+exam_B * np.sin(exam_omega_D*exam_t))*exam_C*exam_omega_D**2
	exam_at1 = -2*exam_xi*exam_omega*exam_vt + (exam_xi**2*exam_omega**2 - exam_omega_D**2)*exam_ut
	return exam_ut, exam_vt, exam_at1


# 该函数用于计算原始数据data的信噪比
def func_SNR(data):
	if len(data)%2==0:
		A_signal = np.max(data) - np.min(data)
		sub_data1 = data[::2]
		sub_data11 = sub_data1[1:]
		sub_data12 = sub_data1[:-1]
		sub_data111 = (sub_data11 + sub_data12)/2

		sub_data2 = data[1:-1:2]
		sub_data = sub_data2 - sub_data111
		A_noise = np.max(sub_data) - np.min(sub_data)

	elif len(data)%2!=0:
		A_signal = np.max(data) - np.min(data)
		sub_data1 = data[::2]
		sub_data11 = sub_data1[1:]
		sub_data12 = sub_data1[:-1]
		sub_data111 = (sub_data11 + sub_data12)/2

		sub_data2 = data[1::2]
		sub_data = sub_data2 - sub_data111
		A_noise = np.max(sub_data) - np.min(sub_data)

	else:
		raise ValueError
	return 20*np.log10(A_signal/A_noise)


# 得到添加进入解析振动信号的噪声信噪比，使得添加的噪声水平与捕捉位移信号中的噪声水平相当
def func_get_SNR(analyze_Data, white_noise, tracking_Data):
	SNR_analyze = func_SNR(analyze_Data)
	SNR_tracking = func_SNR(tracking_Data)

	up_snr = 10*SNR_analyze
	low_snr = SNR_analyze/10
	snr_test = (up_snr+low_snr)/2

	err_snr = abs(SNR_analyze-SNR_tracking)
	count = 0

	while err_snr>1e-5 and count<500:
		
		if SNR_analyze>SNR_tracking:
			up_snr = (up_snr+low_snr)/2

		elif SNR_analyze<SNR_tracking:
			low_snr = (up_snr+low_snr)/2
		
		else:
			raise ValueError
		snr_test = (up_snr+low_snr)/2
		exam_uta = func_add_noise(analyze_Data, white_noise, snr_test)
		SNR_analyze = func_SNR(exam_uta)
		err_snr = abs(SNR_analyze-SNR_tracking)
		count = count + 1
	return snr_test


# 用于给信号添加指定信噪比水平的噪声干扰
def func_add_noise(signal, white_noise, SNR):

	A_signal = np.max(signal) - np.min(signal)
	A_noise = np.max(white_noise) - np.min(white_noise)
	k_gauss = A_signal/A_noise*np.sqrt(10**(-SNR/10))
	use_noise = k_gauss * white_noise
	return signal+use_noise


def func_conv_gauss_wave(data_array, scale):
	lower = -5 * scale # 这里取正负是因为本程序中的高斯函数对称轴为x=0
	upper = 5 * scale  # 这里取正负是因为本程序中的高斯函数对称轴为x=0
	
	s = scale  # s是scale的缩写，方便后续公式的简洁
	timestep = (upper-lower)/(s*10)
	t = np.arange(lower,upper+0.5*timestep,timestep)
	
	C = 1/np.sqrt(np.pi)  # 正则化系数
	theta_t = C * np.exp(-t**2)  # 一阶高斯母小波
	theta_st = C/np.sqrt(s) * np.exp(-t**2/(s**2))  # 一阶高斯小波族
	
	C1 = np.sqrt(np.sqrt(2/np.pi))
	psi_1st = C1 * (-2*t) * np.exp(-t**2)
	psi_1st_st = (-1)**1*C1 * np.exp(-t**2/s**2) * (2*t/s) / (np.sqrt(s))
	
	C2 = np.sqrt(np.sqrt(2/(9*np.pi)))
	psi_2nd = C2 * np.exp(-t**2) * (4*t**2 - 2)
	psi_2nd_st = (-1)**2*C2 * np.exp(-t**2/s**2) * (4*t**2/s**2 - 2) / (np.sqrt(s))

	plt.subplot(1,3,1)
	plt.plot(t,theta_st,label = 'gauss')
	plt.legend(loc="best",fontsize=8)

	plt.subplot(1,3,2)
	plt.plot(t,psi_1st_st,label = 'gauss1')
	plt.legend(loc="best",fontsize=8)

	plt.subplot(1,3,3)
	plt.plot(t,psi_2nd_st,label = 'gauss2')
	plt.legend(loc="best",fontsize=8)

	plt.show()

	data_conv0 = np.convolve(data_array, theta_st, 'same')  # 模仿python源码,卷积前后时间序列数量一致
	data_conv1 = np.convolve(data_array, psi_1st_st, 'same')  # 模仿python源码,卷积前后时间序列数量一致
	data_conv2 = np.convolve(data_array, psi_2nd_st, 'same')  # 模仿python源码,卷积前后时间序列数量一致

	return data_conv0, data_conv1, data_conv2


# 该函数用于四舍五入取整函数，并取到整数位
def func_round(number):
	if number >= 0:
		 if number - int(number) >= 0.5:
		 	return int(number) + 1
		 else:
		 	return int(number)
	elif number < 0:
		 if number - int(number)<=-0.5:
		 	return int(number) - 1
		 else:
		 	return int(number)
	else:
		print("请检查输入！")		


# 该函数采用有限差分法求解一组数组(data_x, data_y)的数值微分
def func_diff_2point(data_x, data_y):
	diff_data=np.zeros_like(data_y)
	for i in range(len(data_y)-1):
		timestep = data_x[i+1] - data_x[i]
		data_step = (data_y[i+1] - data_y[i])
		diff_data[i] = data_step/timestep
	diff_data[-1] = (data_y[-1]-data_y[-2])/timestep
	return diff_data

# 该函数为数值微分-2点中心差分：中间部分为相邻平均数值微分，两端点为相邻数值微分
def diff_2point_central(data, timestep):
	diff_data=np.zeros_like(data)
	diff_data[0] = (data[1]-data[0])/timestep
	for i in range(len(data)-3):
		data_former = 0.5*(data[i]+data[i+1])
		data_later = 0.5*(data[i+1]+data[i+2])
		diff_data[i+1] = (data_later-data_former)/timestep
	diff_data[len(data)-2] = (data[-2]-data[-3])/timestep
	diff_data[len(data)-1] = (data[-1]-data[-2])/timestep
	return diff_data

# 该函数为数值微分-2点中心差分：中间部分为相邻平均数值微分，两端点为相邻数值微分
def diff_8point_central(data, timestep):
	a1 = 0.8024
	a2 = -0.2022
	a3 = 0.03904
	a4 = -0.003732
	diff_data=np.zeros_like(data)
	for i in range(len(data)-9):
		delta4 = a4*(data[i+8] - data[i])
		delta3 = a3*(data[i+7] - data[i+1])
		delta2 = a2*(data[i+6] - data[i+2])
		delta1 = a1*(data[i+5] - data[i+3])
		diff_data[i] = (delta4+delta3+delta2+delta1)/timestep
		i = i + 20
	return diff_data

# 该函数是用于将采样频率混合125Hz，250Hz，500Hz的位移捕捉离散信号通过线性插值调整为采样频率统一为最大频率500Hz的采样信号
def func_update_disp(para_time, para_disp, target_freq):
	max_freq = 1000
	timestep = 1/max_freq
	time_maxnum = np.arange(para_time[0], para_time[-1]+0.5*timestep, timestep)
	disp_maxnum = np.zeros_like(time_maxnum)
	s = 0
	for i in range(len(para_time)-1):
		# python取整函数int为向零方向取整，一定要注意一下，round为四舍五入函数，可以取整到小数位
		count = func_round((para_time[i+1] - para_time[i])/timestep)
		disp_step = (para_disp[i+1] - para_disp[i])/count
		s = s + count
		for j in range (count):
			disp_maxnum[s-count+j] = para_disp[i]+j*disp_step*np.random.normal(loc=1.0,scale=0.1)
	disp_maxnum[-1] = para_disp[-1]

	if max_freq%target_freq==0:
		timestep = 1 / target_freq
		time_update = np.arange(para_time[0], para_time[-1], timestep)
		disp_update = np.zeros_like(time_update)
		amp = func_round(max_freq/target_freq)
		for k in range (len(time_update)):
			disp_update[k] = disp_maxnum[amp*k]
	else:
		print('最大采样频率与目标频率不为整数倍关系！')
	return [time_update, disp_update]


# Dynamic Time Warping动态时间规划距离，用于时间序列相似性，不等长度数据序列
# 该程序用于二分法求解两组离散数组DTW距离最小时的幅度参数-小波微分论文专用
# ！！！！！！！！！！！！！！需要调用dtw库！！！！！！！！！！！！！ #
def func_BinarySearch_DTW(source_array,target_array, para_threshold):
	import dtw
	tail_index = int(len(source_array)/100)
	source = source_array[tail_index:-tail_index]
	target = target_array[tail_index:-tail_index]

	num = 3
	low = 0
	maxs = np.amax(np.abs(source))
	maxt = np.amax(np.abs(target))
	if maxs > maxt:
		height = 2 * maxs / maxt
	elif maxs < maxt:
		height = 2 * maxt / maxs
	else:
		height = 1
	dAmp = (height-low)/num
	AmpArray = np.arange(low, height+0.5*dAmp, dAmp)
	dist = np.zeros(len(AmpArray))
	for i in range(num+1):
		para_dtw = dtw.dtw(source, AmpArray[i]*target)
		dist[i] = para_dtw.distance
	minDist = np.amin(dist)
	minDistIndex = np.argmin(dist)

	count = 0
	while height-low > para_threshold and count < 500:
		if minDist == dist[0]:
			low = AmpArray[0]
			height = AmpArray[1]
		elif minDist == dist[-1]:
			low = AmpArray[-2]
			height = AmpArray[-1]
		else:
			low = AmpArray[minDistIndex-1]
			height = AmpArray[minDistIndex+1]

		dAmp = (height-low)/num
		AmpArray = np.arange(low, height+0.5*dAmp, dAmp)	
		for i in range(num+1):
			para_dtw = dtw.dtw(source, AmpArray[i]*target)
			dist[i] = para_dtw.distance
			count = count + 1
			# print('It is the',count,'-th Iteration')

		print('It is the',count,'-th Iteration')
		minDist = np.amin(dist)
		minDistIndex = np.argmin(dist)

	Amp = (low+height)/2
	Dist = minDist
	return Amp, Dist


# The Euclidean distance欧拉距离，用于时间序列相似性，等长度数据序列
# 二分法求解两组离散数组欧拉距离最小时的幅度参数-小波微分论文专用
def func_BinarySearch_ED(source_array,target_array, para_threshold): 
	tail_index = int(len(source_array)/100)
	source = source_array[tail_index:-tail_index]
	target = target_array[tail_index:-tail_index]

	num = 3
	low = 0
	maxs = np.amax(np.abs(source))
	maxt = np.amax(np.abs(target))
	if maxs > maxt:
		height = 2 * maxs / maxt  # 三分法上边界取两数组峰值倍数的两倍
	elif maxs < maxt:
		height = 2 * maxt / maxs  # 三分法上边界取两数组峰值倍数的两倍
	else:
		height = 1
	dAmp = (height-low)/num
	AmpArray = np.arange(low, height+0.5*dAmp, dAmp)
	dist = np.zeros(len(AmpArray))
	for i in range(num+1):
		para_ed = np.sqrt(np.sum((source - AmpArray[i]*target)**2))
		dist[i] = para_ed
	minDist = np.amin(dist)
	minDistIndex = np.argmin(dist)

	count = 0
	while height-low > para_threshold and count < 500:
		if minDist == dist[0]:
			low = AmpArray[0]
			height = AmpArray[1]
		elif minDist == dist[-1]:
			low = AmpArray[-2]
			height = AmpArray[-1]
		else:
			low = AmpArray[minDistIndex-1]
			height = AmpArray[minDistIndex+1]

		dAmp = (height-low)/num
		AmpArray = np.arange(low, height+0.5*dAmp, dAmp)	
		for i in range(num+1):
			para_ed = np.sqrt(np.sum((source - AmpArray[i]*target)**2))
			dist[i] = para_ed
			count = count + 1
			#print('It is the',count,'-th Iteration, ', 'error = ', height-low)
		minDist = np.amin(dist)
		minDistIndex = np.argmin(dist)
	
	print('It is the',count,'-th Iteration, ', 'error = ', height-low)
	Amp = (low+height)/2
	Dist = minDist
	return Amp, Dist