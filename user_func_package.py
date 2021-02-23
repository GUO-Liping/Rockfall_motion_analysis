#!/usr/bin/env python    
#encoding: utf-8 
import numpy as np


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


# 该函数用于计算原始数据data的信噪比
def func_SNR(data):
	A_signal = np.max(data) - np.min(data)
	sub_data = data[1:] - data[:-1]
	A_noise = np.maximum(np.abs(np.max(sub_data)),np.abs(np.min(sub_data)))
	return 20*np.log10(A_signal/A_noise)


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
			print('It is the',count,'-th Iteration, ', 'error = ', height-low)

		minDist = np.amin(dist)
		minDistIndex = np.argmin(dist)

	Amp = (low+height)/2
	Dist = minDist
	return Amp, Dist