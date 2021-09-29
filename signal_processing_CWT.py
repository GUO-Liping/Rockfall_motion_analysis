# 该程序用于冲击试验速度时程曲线小波降噪，旨在获得可靠的加速度值用于预测落石冲击力
import numpy as np
import matplotlib.pyplot as plt
#import os
import pywt
from user_func_package import *
#import pywt.data
#import pandas as pd
# pywt.families()=['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']
# pywt.wavelist('sym')=['sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']
# pywt.wavelist(kind='continuous')
# ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan']

# 该函数为数值微分：中间部分为相邻平均数值微分，两端点为相邻数值微分
def diff_2point(data, timestep):
	diff_data=np.zeros_like(data)
	diff_data[0] = (data[1]-data[0])/timestep
	for i in range(len(data)-3):
		data_former = 0.5*(data[i]+data[i+1])
		data_later = 0.5*(data[i+1]+data[i+2])
		diff_data[i+1] = (data_later-data_former)/timestep
	diff_data[len(data)-2] = (data[-2]-data[-3])/timestep
	diff_data[len(data)-1] = (data[-1]-data[-2])/timestep
	return diff_data

def diff_8point(data, timestep):
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

# 该函数用于对处理小波分解后的小波高频系数进行阈值处理，低频系数部分保持不变，并更新小波系数
def  coeffs_handle(coeffs, level):
	sigma_value = np.zeros(level)
	threshold_value = np.zeros(level)
	for i in range(level):
		sigma_value[i] = np.median(np.abs(coeffs[i+1]))/0.6745
		threshold_value[i] = sigma_value[i]*np.sqrt(2*np.log(len(coeffs[i+1])))
		coeffs[i+1] = pywt.threshold(coeffs[i+1],threshold_value[i], 'soft')
	return coeffs

time_series = np.array([0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02,0.022,0.024,0.026,0.028,0.03,0.032,0.034,0.036,0.038,0.04,0.042,0.044,0.046,0.048,0.05,0.052,0.054,0.056,0.058,0.06,0.062,0.064,0.066,0.068,0.07,0.072,0.074,0.076,0.078,0.08,0.082,0.084,0.086,0.088,0.09,0.092,0.094,0.096,0.098,0.1,0.102,0.104,0.106,0.108,0.11,0.112,0.114,0.116,0.118,0.12,0.122,0.124,0.126,0.128,0.13,0.132,0.134,0.136,0.138,0.14,0.142,0.144,0.146,0.148,0.15,0.152,0.154,0.156,0.158,0.16,0.162,0.164,0.166,0.168,0.17,0.172,0.174,0.176,0.178,0.18,0.182,0.184,0.186,0.188,0.19,0.192,0.194,0.196,0.198,0.2,0.202,0.204,0.206,0.208,0.21,0.212,0.214,0.216,0.218,0.22,0.222,0.224,0.226,0.228,0.23,0.232,0.234,0.236,0.238,0.24,0.242,0.244,0.246,0.248,0.25,0.252,0.254,0.256,0.258,0.26,0.262,0.264,0.266,0.268,0.27,0.272,0.274,0.276,0.278,0.28,0.282,0.284,0.286,0.288,0.29,0.292,0.294,0.296,0.298,0.3,0.302,0.304,0.306,0.308,0.31,0.312,0.314,0.316,0.318,0.32,0.322,0.324,0.326,0.328,0.33,0.332,0.334,0.336,0.338,0.34,0.342,0.344,0.346,0.348,0.35,0.352,0.354,0.356,0.358,0.36,0.362,0.364,0.366,0.368,0.37,0.372,0.374,0.376,0.378,0.38,0.382,0.384,0.386,0.388,0.39,0.392,0.394,0.396,0.398,0.4,0.402,0.404,0.406,0.408,0.41,0.412,0.414,0.416,0.418,0.42,0.422,0.424,0.426,0.428,0.43,0.432,0.434,0.436,0.438,0.44,0.442,0.444,0.446,0.448,0.45,0.452,0.454,0.456,0.458,0.46,0.462,0.464,0.466,0.468,0.47,0.472,0.474,0.476,0.478,0.48,0.482,0.484,0.486,0.488,0.49,0.492,0.494,0.496,0.498,0.5,0.502,0.504,0.506,0.508,0.51,0.512,0.514,0.516,0.518,0.52,0.522,0.524,0.526,0.528,0.53,0.532,0.534,0.536,0.538,0.54,0.542,0.544,0.546,0.548,0.55,0.552,0.554,0.556,0.558,0.56,0.562,0.564,0.566,0.568,0.57,0.572,0.574,0.576,0.578,0.58,0.582,0.584,0.586,0.588,0.59,0.592,0.594,0.596,0.598,0.6,0.602,0.604,0.606,0.608,0.61,0.612,0.614,0.616,0.618,0.62,0.622,0.624,0.626,0.628,0.63,0.632,0.634,0.636,0.638,0.64,0.642,0.644,0.646,0.648,0.65,0.652,0.654,0.656,0.658,0.66,0.662,0.664,0.666,0.668,0.67,0.672,0.674,0.676,0.678,0.68,0.682,0.684,0.686,0.688,0.69,0.692,0.694,0.696,0.698,0.7,0.702,0.704,0.706,0.708,0.71,0.712,0.714,0.716,0.718,0.72,0.722,0.724,0.726,0.728,0.73,0.732,0.734,0.736,0.738,0.74,0.742,0.744,0.746,0.748,0.75,0.752,0.754,0.756,0.758,0.76,0.762,0.764,0.766,0.768,0.77,0.772,0.774,0.776,0.778,0.78,0.782,0.784,0.786,0.788,0.79,0.792,0.794,0.796,0.798,0.8,0.802,0.804,0.806,0.808,0.81,0.812,0.814,0.816,0.818,0.82,0.822,0.824,0.826,0.828,0.83,0.832,0.834,0.836,0.838,0.84,0.842,0.844,0.846,0.848,0.85,0.852,0.854,0.856,0.858,0.86,0.862,0.864,0.866,0.868,0.87,0.872,0.874,0.876,0.878,0.88,0.882,0.884,0.886,0.888,0.89,0.892,0.894,0.896,0.898,0.9,0.902,0.904,0.906,0.908,0.91,0.912,0.914,0.916,0.918,0.92,0.922,0.924,0.926,0.928,0.93,0.932,0.934,0.936,0.938,0.94,0.942,0.944,0.946,0.948,0.95,0.952,0.954,0.956,0.958,0.96,0.962,0.964,0.966,0.968,0.97,0.972,0.974,0.976,0.978,0.98,0.982,0.984,0.986,0.988,0.99,0.992,0.994,0.996,0.998,1,1.002,1.004,1.006,1.008,1.01,1.012,1.014,1.016,1.018,1.02,1.022,1.024,1.026,1.028,1.03,1.032,1.034,1.036,1.038,1.04,1.042,1.044,1.046,1.048,1.05,1.052,1.054,1.056,1.058,1.06,1.062,1.064,1.066,1.068,1.07,1.072,1.074,1.076,1.078,1.08,1.082,1.084,1.086,1.088,1.09,1.092,1.094,1.096,1.098,1.1,1.102,1.104,1.106,1.108,1.11,1.112,1.114,1.116,1.118,1.12,1.122,1.124,1.126,1.128,1.13,1.132,1.134,1.136,1.138,1.14,1.142,1.144,1.146,1.148,1.15,1.152,1.154,1.156,1.158,1.16,1.162,1.164,1.166,1.168,1.17,1.172,1.174,1.176,1.178,1.18,1.182,1.184,1.186,1.188,1.19,1.192,1.194,1.196,1.198,1.2,1.202,1.204,1.206,1.208,1.21,1.212,1.214,1.216,1.218,1.22,1.222,1.224,1.226,1.228,1.23,1.232,1.234,1.236,1.238,1.24,1.242,1.244,1.246,1.248,1.25,1.252,1.254,1.256,1.258,1.26,1.262,1.264,1.266,1.268,1.27,1.272,1.274,1.276,1.278,1.28,1.282,1.284,1.286,1.288,1.29,1.292,1.294,1.296,1.298,1.3,1.302,1.304,1.306,1.308,1.31,1.312,1.314,1.316,1.318,1.32,1.322,1.324,1.326,1.328,1.33,1.332,1.334,1.336,1.338,1.34,1.342,1.344,1.346,1.348,1.35,1.352,1.354,1.356,1.358,1.36,1.362,1.364,1.366,1.368,1.37,1.372,1.374,1.376,1.378,1.38,1.382,1.384,1.386,1.388,1.39,1.392,1.394,1.396,1.398,1.4,1.402,1.404,1.406,1.408,1.41,1.412,1.414,1.416,1.418,1.42,1.422,1.424,1.426,1.428,1.43,1.432,1.434,1.436,1.438,1.44,1.442,1.444,1.446,1.448,1.45,1.452,1.454,1.456,1.458,1.46,1.462,1.464,1.466,1.468,1.47,1.472,1.474,1.476,1.478,1.48,1.482,1.484,1.486,1.488,1.49,1.492,1.494,1.496,1.498,1.5,1.502,1.504,1.506,1.508,1.51,1.512,1.514,1.516,1.518,1.52,1.522,1.524,1.526,1.528,1.53,1.532,1.534,1.536,1.538,1.54,1.542,1.544,1.546,1.548,1.55,1.552,1.554,1.556,1.558,1.56,1.562,1.564,1.566,1.568,1.57,1.572,1.574,1.576,1.578,1.58,1.582,1.584,1.586,1.588,1.59,1.592,1.594,1.596,1.598,1.6,1.602,1.604,1.606,1.608,1.61,1.612,1.614,1.616,1.618,1.62,1.622,1.624,1.626,1.628,1.63,1.632,1.634,1.636,1.638,1.64,1.642,1.644,1.646,1.648,1.65,1.652,1.654,1.656,1.658,1.66,1.662,1.664,1.666,1.668,1.67,1.672,1.674,1.676,1.678,1.68,1.682,1.684,1.686,1.688,1.69,1.692,1.694,1.696,1.698,1.7,1.702,1.704,1.706,1.708,1.71,1.712,1.714,1.716,1.718,1.72,1.722,1.724,1.726,1.728,1.73,1.732,1.734,1.736,1.738,1.74,1.742,1.744,1.746,1.748,1.75,1.752,1.754,1.756,1.76,1.764,1.768,1.772,1.776,1.78,1.784,1.788,1.792,1.796,1.8,1.804,1.808,1.812,1.816,1.82,1.824,1.828,1.832,1.836,1.84,1.844,1.848,1.852,1.856,1.86,1.864,1.868,1.872,1.876,1.88,1.884,1.888,1.892,1.896,1.9,1.904,1.908,1.912,1.916,1.92,1.924,1.928,1.932,1.936,1.94,1.944,1.948,1.952,1.956,1.96,1.964,1.968,1.972,1.976,1.98,1.984,1.988,1.992,1.996,2,2.004,2.008,2.012,2.016,2.02,2.024,2.028,2.032,2.036,2.04,2.044,2.048,2.052,2.056,2.06,2.064,2.068,2.072,2.076,2.08,2.084,2.088,2.092,2.096,2.1,2.104,2.108,2.112,2.116,2.12,2.124,2.128,2.132,2.136,2.14,2.144,2.148,2.152,2.156,2.16,2.164,2.168,2.172,2.176,2.18,2.184,2.188,2.192,2.196,2.2,2.204,2.208,2.212,2.216,2.22,2.224,2.228,2.232,2.236,2.24,2.244,2.248,2.252,2.256,2.26,2.264,2.268,2.272,2.276,2.28,2.284,2.288,2.292,2.296,2.3,2.304,2.308,2.312,2.316,2.32,2.324,2.328,2.332,2.336,2.34,2.344,2.348,2.352,2.356,2.36,2.364,2.368,2.372,2.376,2.38,2.384,2.388,2.392,2.396,2.4,2.404,2.408,2.412,2.416,2.42,2.424,2.428,2.432,2.436,2.44,2.444,2.448,2.452,2.456,2.46,2.464,2.468,2.472,2.476,2.48,2.484,2.488,2.492,2.496,2.5,2.504,2.508,2.512,2.516,2.52,2.524,2.528,2.532,2.536,2.54,2.544,2.548,2.552,2.556,2.56,2.564,2.568,2.572,2.576,2.58,2.584,2.588,2.592,2.596,2.6,2.604,2.608,2.612,2.616,2.62,2.624,2.628,2.632,2.636,2.64,2.644,2.648,2.652,2.656,2.66,2.664,2.668,2.672,2.676,2.68,2.684,2.688,2.692,2.696,2.7,2.704,2.708,2.712,2.716,2.72,2.724,2.728,2.732,2.736,2.74,2.744,2.748,2.752,2.756,2.76,2.764,2.768,2.772,2.776,2.78,2.784,2.788,2.792,2.796,2.8,2.804,2.808,2.812,2.816,2.82,2.824,2.828,2.832,2.836,2.84,2.844,2.848,2.852,2.856,2.86,2.864,2.868,2.872,2.876,2.88,2.884,2.888,2.892,2.896,2.9,2.904,2.908,2.912,2.916,2.92,2.924,2.928,2.932,2.936,2.94,2.944,2.948,2.952,2.956,2.96,2.964,2.968,2.972,2.976,2.98,2.984,2.988,2.992,2.996,3,3.004,3.008,3.012,3.016,3.02,3.024,3.028,3.032,3.036,3.04,3.044,3.048,3.052,3.056,3.06,3.064,3.068,3.072,3.076,3.08,3.084,3.088,3.092,3.096,3.1,3.104,3.108,3.112,3.116,3.12,3.124,3.128,3.132,3.136,3.14,3.144,3.148,3.152,3.156,3.16,3.164,3.168,3.172,3.176,3.18,3.184,3.188,3.192,3.196,3.2,3.204,3.208,3.212,3.216,3.22,3.224,3.228,3.232,3.236,3.24,3.244,3.248,3.252,3.256,3.26,3.264,3.268,3.272,3.276,3.28,3.284,3.288,3.292,3.296,3.3,3.304,3.308,3.312,3.316,3.32,3.324,3.328,3.332,3.336,3.34,3.344,3.348,3.352,3.356,3.36,3.364,3.368,3.372,3.376,3.38,3.384,3.388,3.392,3.396,3.4,3.404,3.408,3.412,3.416,3.42,3.424,3.428,3.432,3.436,3.44,3.444,3.448,3.452,3.456,3.46,3.464,3.468,3.472,3.476,3.48,3.484,3.488,3.492,3.496,3.5,3.504,3.508,3.512,3.516,3.52,3.524,3.528,3.532,3.536,3.54,3.544,3.548,3.552,3.556,3.56,3.564,3.568,3.572,3.576,3.58,3.584,3.588,3.592,3.596,3.6,3.604,3.608,3.612,3.616,3.62,3.624,3.628,3.632,3.636,3.64,3.644,3.648,3.652,3.656,3.66,3.664,3.668,3.676,3.684,3.692,3.7,3.708,3.716,3.724,3.732,3.74,3.748,3.756,3.764,3.772,3.78,3.788,3.796,3.804,3.812,3.82,3.828,3.836,3.844,3.852,3.86,3.868,3.876,3.884,3.892,3.9,3.908,3.916,3.924,3.932,3.94,3.948,3.956,3.964,3.972,3.98,3.988,3.996,4.004,4.012,4.02,4.028,4.036,4.044,4.052,4.06,4.068,4.076,4.084,4.092,4.1,4.108,4.116,4.124,4.14,4.156,4.172,4.188,4.204,4.22,4.236,4.252,4.268,4.284,4.3,4.316,4.332,4.348,4.364,4.38,4.396,4.412,4.428,4.444,4.46,4.476,4.492,4.508,4.524,4.54,4.556,4.572,4.588,4.604,4.62])
disp_series = np.array([0,-0.016276871,-0.03410392,-0.055031325,-0.07595873,-0.099211403,-0.11858863,-0.137965857,-0.157343084,-0.175170133,-0.19454736,-0.213924587,-0.231751636,-0.243377972,-0.27050609,-0.292983673,-0.309260544,-0.334063395,-0.353440622,-0.370492581,-0.393745254,-0.413122481,-0.432499708,-0.451876935,-0.47280434,-0.494506835,-0.516984418,-0.534811467,-0.554188694,-0.580541723,-0.598368772,-0.621621444,-0.635573048,-0.654950275,-0.670452056,-0.693704729,-0.724708292,-0.753386588,-0.774313994,-0.793691221,-0.808417913,-0.830120407,-0.850272724,-0.871975218,-0.892902623,-0.915380207,-0.933982345,-0.95490975,-0.973511888,-0.994439293,-1.015366698,-1.038619371,-1.061872043,-1.082799448,-1.100626497,-1.120778813,-1.137055684,-1.158758178,-1.177360316,-1.197512633,-1.222315483,-1.244017978,-1.264170294,-1.28432261,-1.310675639,-1.330827955,-1.350980271,-1.372682765,-1.392059992,-1.409887041,-1.430039357,-1.450191674,-1.47034399,-1.491271395,-1.509873533,-1.534676384,-1.5548287,-1.572655749,-1.591257887,-1.609084935,-1.627687073,-1.645514122,-1.666441528,-1.68116822,-1.699770358,-1.713721962,-1.733874278,-1.750151149,-1.764877841,-1.781154712,-1.798981761,-1.811383186,-1.8245597,-1.840836571,-1.840836571,-1.856338353,-1.873390313,-1.889667183,-1.898193163,-1.903618787,-1.9098195,-1.913694945,-1.915245123,-1.91757039,-1.920670747,-1.920670747,-1.920670747,-1.920670747,-1.919120569,-1.916020212,-1.9098195,-1.902843698,-1.896642985,-1.889667183,-1.88424156,-1.879591025,-1.871840134,-1.861763976,-1.853237997,-1.847037284,-1.838511304,-1.828435146,-1.819134077,-1.810608097,-1.800531939,-1.792781048,-1.784255068,-1.777279266,-1.768753286,-1.761002396,-1.751701327,-1.744725525,-1.734649367,-1.727673565,-1.716047229,-1.70674616,-1.69822018,-1.693569645,-1.684268576,-1.676517686,-1.666441528,-1.65636537,-1.64783939,-1.63931341,-1.628462163,-1.620711272,-1.612185292,-1.603659312,-1.59668351,-1.587382441,-1.576531194,-1.568005214,-1.557929056,-1.553278522,-1.545527631,-1.540102007,-1.533901294,-1.524600225,-1.517624424,-1.508323355,-1.499022286,-1.492046484,-1.483520504,-1.477319791,-1.471119079,-1.464143277,-1.458717653,-1.452516941,-1.44476605,-1.437015159,-1.430814446,-1.425388823,-1.416087754,-1.409887041,-1.40213615,-1.39438526,-1.386634369,-1.376558211,-1.367257142,-1.358731162,-1.350980271,-1.346329736,-1.337028667,-1.326952509,-1.319976708,-1.313000906,-1.307575282,-1.30137457,-1.295948946,-1.288973144,-1.281997343,-1.275021541,-1.269595917,-1.263395205,-1.257194492,-1.248668512,-1.240142532,-1.23316673,-1.226190929,-1.220765305,-1.213789503,-1.206813702,-1.202163167,-1.195962454,-1.19131192,-1.181235762,-1.17425996,-1.168834337,-1.164183802,-1.155657822,-1.147906931,-1.141706219,-1.137830773,-1.134730417,-1.129304793,-1.126204437,-1.120778813,-1.113027923,-1.106052121,-1.100626497,-1.093650696,-1.086674894,-1.082024359,-1.077373825,-1.074273468,-1.070398023,-1.067297667,-1.062647132,-1.057996598,-1.052570974,-1.046370262,-1.040169549,-1.033193747,-1.025442856,-1.019242144,-1.01381652,-1.008390896,-1.002965273,-0.995989471,-0.990563848,-0.986688402,-0.980487689,-0.976612244,-0.97118662,-0.966536086,-0.96343573,-0.959560284,-0.95490975,-0.951034304,-0.945608681,-0.942508324,-0.93785779,-0.931657077,-0.927006543,-0.923906186,-0.919255652,-0.914605117,-0.908404405,-0.905304048,-0.902203692,-0.898328247,-0.892127534,-0.887477,-0.884376643,-0.878175931,-0.874300485,-0.869649951,-0.866549594,-0.862674149,-0.860348882,-0.856473436,-0.85337308,-0.849497635,-0.8448471,-0.842521833,-0.837871298,-0.834770942,-0.831670586,-0.82779514,-0.823919695,-0.81926916,-0.816168804,-0.813068448,-0.811518269,-0.807642824,-0.805317557,-0.801442111,-0.799116844,-0.796791577,-0.792916132,-0.789040686,-0.787490508,-0.783615063,-0.781289795,-0.77741435,-0.773538904,-0.76888837,-0.767338192,-0.764237835,-0.762687657,-0.76036239,-0.756486945,-0.754161677,-0.752611499,-0.752611499,-0.751061321,-0.747960965,-0.746410787,-0.744860608,-0.742535341,-0.742535341,-0.740985163,-0.738659896,-0.737884807,-0.73478445,-0.734009361,-0.731684094,-0.729358827,-0.729358827,-0.72625847,-0.723933203,-0.722383025,-0.720832847,-0.719282669,-0.71850758,-0.71773249,-0.715407223,-0.715407223,-0.713857045,-0.711531778,-0.7099816,-0.709206511,-0.708431421,-0.706106154,-0.703005798,-0.703005798,-0.703005798,-0.70145562,-0.700680531,-0.700680531,-0.700680531,-0.699130353,-0.698355263,-0.696805085,-0.695254907,-0.695254907,-0.693704729,-0.694479818,-0.695254907,-0.694479818,-0.696805085,-0.695254907,-0.697580174,-0.696805085,-0.695254907,-0.695254907,-0.695254907,-0.697580174,-0.694479818,-0.694479818,-0.69292964,-0.696805085,-0.695254907,-0.694479818,-0.695254907,-0.695254907,-0.695254907,-0.695254907,-0.695254907,-0.695254907,-0.696029996,-0.695254907,-0.69292964,-0.695254907,-0.696805085,-0.696805085,-0.696805085,-0.696805085,-0.698355263,-0.696805085,-0.697580174,-0.696029996,-0.699130353,-0.699905442,-0.699905442,-0.699905442,-0.699905442,-0.699905442,-0.699130353,-0.699905442,-0.702230709,-0.703005798,-0.705331065,-0.707656332,-0.710756689,-0.711531778,-0.711531778,-0.713857045,-0.714632134,-0.713857045,-0.713857045,-0.713857045,-0.713857045,-0.716182312,-0.71850758,-0.720832847,-0.720832847,-0.723933203,-0.725483381,-0.727808649,-0.727808649,-0.729358827,-0.729358827,-0.730909005,-0.734009361,-0.736334628,-0.737109718,-0.737884807,-0.740210074,-0.741760252,-0.742535341,-0.744860608,-0.748736054,-0.75183641,-0.753386588,-0.756486945,-0.758812212,-0.761137479,-0.764237835,-0.765788014,-0.768113281,-0.771213637,-0.774313994,-0.776639261,-0.778964528,-0.781289795,-0.783615063,-0.787490508,-0.790590864,-0.79446631,-0.798341755,-0.799891933,-0.803767379,-0.806092646,-0.807642824,-0.812293359,-0.815393715,-0.816943893,-0.81926916,-0.823919695,-0.827020051,-0.830120407,-0.832445675,-0.835546031,-0.837096209,-0.840196566,-0.8448471,-0.848722545,-0.85337308,-0.856473436,-0.859573793,-0.864224327,-0.867324683,-0.871975218,-0.875075574,-0.878175931,-0.882051376,-0.88670191,-0.889802267,-0.894452801,-0.898328247,-0.902203692,-0.906854227,-0.91227985,-0.916930385,-0.923906186,-0.932432166,-0.93785779,-0.944833592,-0.952584483,-0.960335373,-0.965760997,-0.970411531,-0.972736799,-0.975837155,-0.978937511,-0.981262779,-0.982037868,-0.983588046,-0.989788758,-0.994439293,-0.999089827,-1.002965273,-1.007615807,-1.012266342,-1.017691965,-1.0223425,-1.026217945,-1.031643569,-1.036294103,-1.042494816,-1.04792044,-1.054121152,-1.061872043,-1.067297667,-1.071948201,-1.076598736,-1.08124927,-1.086674894,-1.091325428,-1.09830123,-1.102951765,-1.109927566,-1.117678457,-1.12387917,-1.129304793,-1.134730417,-1.138605862,-1.141706219,-1.146356753,-1.150232199,-1.154107644,-1.159533268,-1.164183802,-1.169609426,-1.176585227,-1.18278594,-1.189761742,-1.198287722,-1.206038613,-1.213789503,-1.220765305,-1.228516196,-1.236267087,-1.243242888,-1.250993779,-1.257969581,-1.264170294,-1.271146095,-1.277346808,-1.286647877,-1.293623679,-1.299824392,-1.304474926,-1.310675639,-1.320751797,-1.328502688,-1.337803757,-1.344779558,-1.350980271,-1.357180984,-1.366482053,-1.374232943,-1.381983834,-1.388959636,-1.392835081,-1.399035794,-1.407561774,-1.416862843,-1.424613734,-1.433914803,-1.443990961,-1.451741852,-1.457942564,-1.467243633,-1.475769613,-1.481970326,-1.488171039,-1.494371751,-1.504447909,-1.5121988,-1.52072478,-1.526150404,-1.534676384,-1.542427274,-1.550953254,-1.560254323,-1.568005214,-1.575756105,-1.580406639,-1.587382441,-1.594358243,-1.600558956,-1.608309846,-1.616835826,-1.624586717,-1.636213053,-1.647064301,-1.654040102,-1.660240815,-1.669541884,-1.669541884,-1.681943309,-1.692019467,-1.701320536,-1.709071427,-1.71527214,-1.726123387,-1.736199545,-1.743950436,-1.751701327,-1.759452218,-1.767978197,-1.771853643,-1.777279266,-1.789680692,-1.795106315,-1.801307028,-1.809057919,-1.812933364,-1.820684255,-1.828435146,-1.836186037,-1.84161166,-1.846262195,-1.847812373,-1.851687818,-1.857113442,-1.857888531,-1.864864333,-1.869514867,-1.873390313,-1.876490669,-1.877265758,-1.880366114,-1.880366114,-1.880366114,-1.880366114,-1.880366114,-1.877265758,-1.874940491,-1.872615224,-1.871065045,-1.8671896,-1.863314155,-1.861763976,-1.859438709,-1.851687818,-1.846262195,-1.843161838,-1.835410948,-1.828435146,-1.823009522,-1.814483542,-1.80828283,-1.801307028,-1.795106315,-1.792005959,-1.786580335,-1.780379623,-1.773403821,-1.767203108,-1.762552574,-1.756351861,-1.749376059,-1.742400258,-1.736199545,-1.729223743,-1.722247942,-1.714497051,-1.710621605,-1.70674616,-1.702870714,-1.695119824,-1.688919111,-1.684268576,-1.677292775,-1.671092062,-1.66411626,-1.660240815,-1.653265013,-1.64783939,-1.641638677,-1.636213053,-1.633112697,-1.626136895,-1.620711272,-1.616060737,-1.609860025,-1.604434401,-1.599008777,-1.592032976,-1.585057174,-1.57963155,-1.574205927,-1.57110557,-1.567230125,-1.561029412,-1.554053611,-1.547077809,-1.541652185,-1.539326918,-1.536226562,-1.530800938,-1.526925493,-1.521499869,-1.515299156,-1.509098444,-1.505222998,-1.499797375,-1.496697019,-1.492046484,-1.48739595,-1.482745415,-1.47886997,-1.475769613,-1.474219435,-1.469568901,-1.467243633,-1.464918366,-1.460267832,-1.455617297,-1.450191674,-1.445541139,-1.440115515,-1.434689892,-1.430039357,-1.426163912,-1.423063556,-1.418413021,-1.414537576,-1.409887041,-1.408336863,-1.406011596,-1.401361061,-1.398260705,-1.392835081,-1.388959636,-1.385084191,-1.378883478,-1.375783122,-1.371132587,-1.365706964,-1.36028134,-1.357956073,-1.357956073,-1.355630805,-1.353305538,-1.349430093,-1.346329736,-1.341679202,-1.338578846,-1.336253578,-1.331603044,-1.328502688,-1.324627242,-1.320751797,-1.31765144,-1.31765144,-1.313775995,-1.312225817,-1.309125461,-1.306025104,-1.302924748,-1.299049302,-1.297499124,-1.295173857,-1.292073501,-1.290523323,-1.289748233,-1.287422966,-1.287422966,-1.28432261,-1.281997343,-1.281997343,-1.278121897,-1.273471363,-1.271921185,-1.269595917,-1.268045739,-1.268045739,-1.268820828,-1.265720472,-1.263395205,-1.261845026,-1.261845026,-1.260294848,-1.259519759,-1.25874467,-1.256419403,-1.253319047,-1.252543957,-1.25021869,-1.25021869,-1.247893423,-1.246343245,-1.246343245,-1.246343245,-1.242467799,-1.242467799,-1.242467799,-1.240142532,-1.240142532,-1.237817265,-1.237817265,-1.237817265,-1.237817265,-1.237817265,-1.235491998,-1.235491998,-1.235491998,-1.235491998,-1.23316673,-1.231616552,-1.231616552,-1.230841463,-1.229291285,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.227741107,-1.230066374,-1.230841463,-1.231616552,-1.23316673,-1.234716909,-1.236267087,-1.237817265,-1.237817265,-1.240142532,-1.24169271,-1.243242888,-1.245568156,-1.247118334,-1.250993779,-1.254094136,-1.255644314,-1.257194492,-1.257194492,-1.259519759,-1.261845026,-1.262620116,-1.263395205,-1.263395205,-1.264945383,-1.26727065,-1.268820828,-1.270371006,-1.272696274,-1.274246452,-1.27579663,-1.278896986,-1.281997343,-1.285097699,-1.287422966,-1.291298412,-1.295948946,-1.30137457,-1.306800193,-1.312225817,-1.316876351,-1.323077064,-1.328502688,-1.333928311,-1.341679202,-1.345554647,-1.352530449,-1.359506251,-1.365706964,-1.372682765,-1.379658567,-1.388184547,-1.397485616,-1.404461418,-1.412212309,-1.420738288,-1.42771409,-1.43624007,-1.443215872,-1.451741852,-1.458717653,-1.464918366,-1.474994524,-1.485070682,-1.497472108,-1.508323355,-1.516849335,-1.525375315,-1.535451473,-1.543202363,-1.552503432,-1.561804501,-1.57188066,-1.57963155,-1.588932619,-1.598233688,-1.606759668,-1.616835826,-1.626911984,-1.636988142,-1.646289211,-1.65636537,-1.667991706,-1.68116822,-1.692019467,-1.70674616,-1.719147585,-1.729223743,-1.736974634,-1.747050792,-1.758677128,-1.769528376,-1.775729088,-1.784255068,-1.792005959,-1.798981761,-1.805182473,-1.810608097,-1.817583899,-1.823009522,-1.827660057,-1.832310591,-1.836961126,-1.838511304,-1.838511304,-1.83308568,-1.827660057,-1.822234433,-1.819909166,-1.812933364,-1.805182473,-1.79975685,-1.793556137,-1.787355424,-1.781929801,-1.776504177,-1.771078554,-1.763327663,-1.750151149,-1.738524812,-1.731549011,-1.723023031,-1.717597407,-1.708296338,-1.699770358,-1.690469289,-1.685818755,-1.678067864,-1.669541884,-1.661015904,-1.65559028,-1.645514122,-1.635437964,-1.630012341,-1.62226145,-1.615285648,-1.609860025,-1.604434401,-1.595908421,-1.589707708,-1.582731907,-1.577306283,-1.57188066,-1.568005214,-1.561804501,-1.557153967,-1.552503432,-1.547077809,-1.543977453,-1.540102007,-1.534676384,-1.530800938,-1.526925493,-1.521499869,-1.517624424,-1.513748978,-1.508323355,-1.506773177,-1.50367282,-1.500572464,-1.499797375,-1.49514684,-1.492046484,-1.488946128,-1.48662086,-1.484295593,-1.482745415,-1.481195237,-1.479645059,-1.477319791,-1.475769613,-1.474994524,-1.474219435,-1.474219435,-1.472669257,-1.472669257,-1.472669257,-1.471894168,-1.471894168,-1.471894168,-1.471894168,-1.471894168,-1.471894168,-1.471894168,-1.471894168,-1.471894168,-1.474994524,-1.476544702,-1.477319791,-1.480420148,-1.483520504,-1.485070682,-1.48739595,-1.489721217,-1.492046484,-1.49514684,-1.500572464,-1.504447909,-1.509098444,-1.511423711,-1.513748978,-1.517624424,-1.52072478,-1.523825136,-1.527700582,-1.531576027,-1.534676384,-1.539326918,-1.542427274,-1.544752542,-1.547852898,-1.550178165,-1.554053611,-1.559479234,-1.566455036,-1.573430838,-1.57963155,-1.584282085,-1.588932619,-1.595133332,-1.600558956,-1.605984579,-1.610635114,-1.616835826,-1.623036539,-1.628462163,-1.633887786,-1.641638677,-1.64783939,-1.654815191,-1.661790993,-1.667991706,-1.674967508,-1.682718398,-1.690469289,-1.698995269,-1.70674616,-1.714497051,-1.720697763,-1.726898476,-1.733874278,-1.74007499,-1.745500614,-1.751701327,-1.758677128,-1.766428019,-1.771853643,-1.777279266,-1.781929801,-1.786580335,-1.792005959,-1.795106315,-1.798206672,-1.802082117,-1.803632295,-1.805182473,-1.805182473,-1.805182473,-1.805182473,-1.803632295,-1.802857206,-1.79975685,-1.796656493,-1.795106315,-1.79123087,-1.786580335,-1.778829445,-1.77417891,-1.770303465,-1.764877841,-1.761002396,-1.754801683,-1.749376059,-1.745500614,-1.74085008,-1.735424456,-1.731549011,-1.726123387,-1.719922674,-1.716047229,-1.709846516,-1.703645804,-1.69822018,-1.691244378,-1.685043666,-1.678067864,-1.671867151,-1.666441528,-1.66411626,-1.659465726,-1.65559028,-1.652489924,-1.64783939,-1.643963944,-1.63931341,-1.635437964,-1.629237252,-1.626136895,-1.62226145,-1.619161094,-1.616060737,-1.61373547,-1.610635114,-1.607534757,-1.604434401,-1.603659312,-1.600558956,-1.598233688,-1.597458599,-1.595908421,-1.595908421,-1.594358243,-1.592808065,-1.592808065,-1.592032976,-1.592808065,-1.592808065,-1.592808065,-1.592808065,-1.592032976,-1.592032976,-1.592032976,-1.591257887,-1.591257887,-1.591257887,-1.591257887,-1.591257887,-1.591257887,-1.591257887,-1.591257887,-1.591257887,-1.591257887,-1.591257887,-1.593583154,-1.594358243,-1.598233688,-1.601334045,-1.603659312,-1.605984579,-1.609084935,-1.612960381,-1.616835826,-1.619161094,-1.62226145,-1.625361806,-1.628462163,-1.632337608,-1.636988142,-1.640863588,-1.644739033,-1.64783939,-1.652489924,-1.660240815,-1.665666439,-1.671092062,-1.676517686,-1.679618042,-1.683493487,-1.688919111,-1.691244378,-1.695119824,-1.69822018,-1.702095625,-1.705971071,-1.710621605,-1.714497051,-1.719147585,-1.724573209,-1.729223743,-1.734649367,-1.739299901,-1.743175347,-1.747825881,-1.750926238,-1.755576772,-1.759452218,-1.763327663,-1.76565293,-1.767978197,-1.771853643,-1.774953999,-1.775729088,-1.776504177,-1.779604534,-1.781154712,-1.781154712,-1.781154712,-1.778829445,-1.777279266,-1.774953999,-1.773403821,-1.771078554,-1.768753286,-1.767203108,-1.764877841,-1.762552574,-1.759452218,-1.756351861,-1.754026594,-1.749376059,-1.745500614,-1.742400258,-1.74007499,-1.736974634,-1.733874278,-1.729998832,-1.725348298,-1.722247942,-1.716822318,-1.713721962,-1.708296338,-1.705971071,-1.702095625,-1.69822018,-1.693569645,-1.692019467,-1.688919111,-1.686593844,-1.684268576,-1.682718398,-1.680393131,-1.678842953,-1.676517686,-1.673417329,-1.671867151,-1.667991706,-1.665666439,-1.663341171,-1.660240815,-1.657140459,-1.654815191,-1.654040102,-1.654040102,-1.654040102,-1.652489924,-1.650939746,-1.650939746,-1.648614479,-1.648614479,-1.649389568,-1.649389568,-1.649389568,-1.649389568,-1.649389568,-1.649389568,-1.649389568,-1.649389568,-1.649389568,-1.649389568,-1.649389568,-1.649389568,-1.649389568,-1.650939746,-1.654040102,-1.65636537,-1.658690637,-1.663341171,-1.661790993,-1.664891349,-1.666441528,-1.667216617,-1.669541884,-1.671867151,-1.67264224,-1.676517686,-1.680393131,-1.68116822,-1.682718398,-1.686593844,-1.690469289,-1.693569645,-1.695894913,-1.699770358,-1.702870714,-1.705195982,-1.708296338,-1.710621605,-1.712946873,-1.716047229,-1.719147585,-1.721472852,-1.724573209,-1.727673565,-1.729998832,-1.733099189,-1.734649367,-1.736199545,-1.738524812,-1.74085008,-1.743175347,-1.746275703,-1.74860097,-1.750926238,-1.751701327,-1.753251505,-1.754026594,-1.754026594,-1.755576772,-1.758677128,-1.759452218,-1.759452218,-1.75712695,-1.754026594,-1.754026594,-1.754026594,-1.751701327,-1.750151149,-1.750926238,-1.749376059,-1.749376059,-1.747825881,-1.745500614,-1.743175347,-1.74007499,-1.734649367,-1.728448654,-1.723023031,-1.716822318,-1.712946873,-1.709071427,-1.705195982,-1.701320536,-1.697445091,-1.695894913,-1.692794556,-1.6896942,-1.686593844,-1.683493487,-1.680393131,-1.678842953,-1.677292775,-1.675742597,-1.675742597,-1.674967508,-1.675742597,-1.676517686,-1.678067864,-1.679618042,-1.681943309,-1.685043666,-1.687368933,-1.6896942,-1.691244378,-1.692794556,-1.695894913,-1.698995269,-1.702095625,-1.705971071,-1.709846516,-1.713721962,-1.718372496,-1.720697763,-1.725348298,-1.728448654,-1.733099189,-1.736199545,-1.739299901,-1.741625169,-1.743175347,-1.744725525,-1.744725525,-1.744725525,-1.744725525,-1.742400258,-1.739299901,-1.737749723,-1.736199545,-1.733874278,-1.730773921,-1.728448654,-1.722247942,-1.716047229,-1.709071427,-1.705195982,-1.701320536,-1.698995269,-1.696670002,-1.695119824,-1.695119824,-1.695894913,-1.699770358,-1.703645804,-1.705971071,-1.709071427,-1.712171783,-1.718372496,-1.72379812,-1.728448654,-1.733099189,-1.735424456,-1.735424456,-1.735424456,-1.735424456,-1.735424456,-1.729998832,-1.724573209,-1.719922674,-1.716822318,-1.712946873,-1.708296338,-1.708296338])
time_updated, disp_updated = func_update_disp(time_series,disp_series, 500)

signal_handle = disp_updated

wavelet_use = 'gaus1'
scale_use = np.arange(32,256,1)

wavelet_use = 'gaus2'
scale_use = np.arange(5,50,1)

totalscal = 256
fc = pywt.central_frequency('gaus2',precision=16)
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 1, -1)

cwtmatr, freqs = pywt.cwt(signal_handle, scale_use, wavelet_use)  #小波分解
print('fc=',cwtmatr.shape)

fig, ax = plt.subplots()
num = int(len(disp_updated)/10)
CS = ax.contour(time_updated[num:-num], freqs, abs(cwtmatr[:,num:-num]))
#CS2 = ax.contourf(time_updated, freqs, abs(cwtmatr))
ax.clabel(CS, CS.levels, inline=True, fontsize=8)

# reconstruct_wave = pywt.waverec(coeffs, wavelet_use)  # 小波重构

#####################################################################
#计算降噪前后数据的信噪比
'''
SNR = (np.mean(signal_handle))**2/(np.std(signal_handle,ddof=1))**2
SNR_dB = 10*np.log10(SNR)
SNR2 = (np.mean(reconstruct_wave))**2/np.var(reconstruct_wave)
SNR_dB2 = 10*np.log10(SNR2)
print('Before handle, SNRdB=', SNR_dB)
print('After handle, SNR_dB2=', SNR_dB2)
'''
#####################################################################

print(cwtmatr.shape)
print(freqs.shape)
print(time_updated.shape)
#for i in range(np.shape(cwtmatr)[0]):
#	plt.plot(cwtmatr[i,:])
#	plt.show()
##plt.plot(new_data)
plt.show()
