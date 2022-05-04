#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# NetPanelAnalysis_V1_0_2主函数

import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow,QTableWidget
from App_Ui_signal_processing import Ui_MainWindow
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
from user_func_package import *

#from PyQt5.QtWidgets import *
#from PyQt5.QtCore import *


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)

        self.pushButton_1.clicked.connect(self.draw_fourier)
        self.pushButton_2.clicked.connect(self.draw_wavelet)
        self.pushButton_3.clicked.connect(self.reset_cmd)
        self.slider_scale.valueChanged.connect(self.set_scale)


    def getTableData(self):
        xData = []
        yData = []
        # nrows = self.tableWidget.rowCount()
        row = 0
        item0 = self.tableWidget.item(0, 0)
        item1 = self.tableWidget.item(0, 1)

        while(item0!=None and item1!=None):
            text_str0 = item0.text()
            text_str1 = item1.text()
            xData.append(text_str0)
            yData.append(text_str1)
            row = row + 1
            item0 = self.tableWidget.item(row, 0)
            item1 = self.tableWidget.item(row, 1)
        self.input_xData = np.asarray(xData, dtype='float')
        self.input_yData = np.asarray(yData, dtype='float')

    def getFFT_TableData(self):
        N = len(self.input_yData)
        self.sample_rate = 250
        self.inputX_updated, self.inputY_updated = func_update_disp(self.input_xData,self.input_yData, self.sample_rate)
        freq_data = np.fft.fft(self.inputY_updated - np.average(self.inputY_updated))
        self.frequencies = np.linspace (0.0, self.sample_rate/2, int (N/2), endpoint=True)
        self.amp_frequencies = 2/N * abs(freq_data[0:N//2])

    def get_freqsEnergy(self):

        i_index = np.arange(4)
        j_index = 5**i_index

        i50, i90, i99 = func_freqs_divide(self.amp_frequencies)
        f_i = np.array([0.001,self.frequencies[i50],self.frequencies[i90],self.frequencies[i99],250])
        fc_g1 = 0.254  # 一阶高斯小波的中心频率
        fc_g2 = 0.339  # 二阶高斯小波的中心频率
        fc_g = fc_g2  # 高斯小波中心频率
        s_g = fc_g*self.sample_rate/f_i  # 高斯小波尺度参数
        E_total = pow(np.linalg.norm(self.amp_frequencies,ord=2),2)  # Fourier频域总能量

        i_0 = np.where((self.frequencies>=0) & (self.frequencies<=self.frequencies[i50]))
        i_1 = np.where((self.frequencies>self.frequencies[i50]) & (self.frequencies<=self.frequencies[i90]))
        i_2 = np.where((self.frequencies>self.frequencies[i90]) & (self.frequencies<=self.frequencies[i99]))
        i_3 = np.where((self.frequencies>self.frequencies[i99]))
        E_0 = np.sum(pow(self.amp_frequencies[i_0],2)) / E_total*100
        E_1 = np.sum(pow(self.amp_frequencies[i_1],2)) / E_total*100
        E_2 = np.sum(pow(self.amp_frequencies[i_2],2)) / E_total*100
        E_3 = np.sum(pow(self.amp_frequencies[i_3],2)) / E_total*100

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        self.labels = [str("%.1f" %f_i[0])+'Hz'+'~'+str("%.1f" %f_i[1])+'Hz',
                  str("%.1f" %f_i[1])+'Hz'+'~'+str("%.1f" %f_i[2])+'Hz',
                  str("%.1f" %f_i[2])+'Hz'+'~'+str("%.1f" %f_i[3])+'Hz',
                  str("%.1f" %f_i[3])+'Hz'+'~'+str("%.1f" %f_i[4])+'Hz']
        self.sizes = [E_0, E_1, E_2, E_3]
        self.explode = (0, 0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    def get_derivatives(self):

        self.time_updated, self.disp_updated = func_update_disp(self.input_xData,self.input_yData, self.sample_rate)

        n_fit = int(0.05*len(self.disp_updated)) # 第一个常数，表示用于待处理数据中可用于抛物线拟合的捕捉数据点数量
        n_add = int(0.10*len(self.disp_updated)) # 第二个常数，表示在信号首尾端需要添加的数据点数量

        self.time_updated, self.disp_updated = func_user_pad(self.time_updated, self.disp_updated, n_fit, 'before', n_add)
        self.time_updated, self.disp_updated = func_user_pad(self.time_updated, self.disp_updated, n_fit, 'after',  n_add)
        
        fc_gauss0 = 1/(2*np.pi)*np.sqrt(2/np.pi)
        fc_gauss1 = 1/(1*np.pi)*np.sqrt(2/np.pi)
        fc_gauss2 = 4/(3*np.pi)*np.sqrt(2/np.pi)
        fc_gauss3 = 8/(5*np.pi)*np.sqrt(2/np.pi)

        # 处理试验捕捉的含噪信号
        # 对含噪信号进行高斯小波卷积
        test_time = self.time_updated[n_add:-n_add]
        test_utn = self.disp_updated

        test_utn_conv0 = func_conv_gauss_wave(test_utn, self.scale_parameter*fc_gauss0/fc_gauss0)[0][n_add:-n_add]
        test_utn_conv1 = func_conv_gauss_wave(test_utn, self.scale_parameter*fc_gauss1/fc_gauss0)[1][n_add:-n_add]
        test_utn_conv2 = func_conv_gauss_wave(test_utn, self.scale_parameter*fc_gauss2/fc_gauss0)[2][n_add:-n_add]  # 手动生成高斯小波函数族,并与信号进行卷积
        test_utn_conv3 = func_conv_gauss_wave(test_utn, self.scale_parameter*fc_gauss3/fc_gauss0)[3][n_add:-n_add]  # 手动生成高斯小波函数族,并与信号进行卷积

        # 实际信号并无真实解，需要对含噪信号高斯小波卷积结果反向积分，通过积分-微分之间的自洽性验证结果的准确性，积分时需要输入初始条件
        integral_test_utn_conv1 = func_integral_trapozoidal_rule(test_time, test_utn_conv1, 0)  # 梯形法则一次积分，初始条件为0。
        integral_test_utn_conv2 = func_integral_trapozoidal_rule(test_time, test_utn_conv2, 0)  # 梯形法则再次积分，初始条件为0。
        integral_test_utn_conv3 = func_integral_trapozoidal_rule(test_time, test_utn_conv3, 0)  # 梯形法则再次积分，初始条件为0。

        test_source = test_utn[n_add:-n_add]

        Amp0_test_utn, ED0_test_utn, Amp0_convol = func_BinarySearch_ED(test_source, test_utn_conv0, 1e-10)
        Amp1_test_utn, ED1_test_utn, Amp1_convol = func_BinarySearch_ED(Amp0_convol, integral_test_utn_conv1, 1e-10)
        Amp2_test_utn, ED2_test_utn, Amp2_convol = func_BinarySearch_ED(Amp1_test_utn*test_utn_conv1, integral_test_utn_conv2, 1e-10)
        Amp3_test_utn, ED3_test_utn, Amp3_convol = func_BinarySearch_ED(Amp2_test_utn*test_utn_conv2, integral_test_utn_conv3, 1e-10)

        self.time_array = test_time
        self.zero_derivative = Amp0_test_utn*test_utn_conv0
        self.first_derivative = Amp1_test_utn*test_utn_conv1
        self.second_derivative = Amp2_test_utn*test_utn_conv2

    def draw_fourier(self):
        try:
            self.getTableData()
            self.getFFT_TableData()
            self.get_freqsEnergy()

            self.ax11.cla()
            self.ax11.set_title('Time-domain signal')
            self.ax11.plot(self.input_xData, self.input_yData)

            self.ax12.cla()
            self.ax12.set_title('Frequency-domain signal')
            self.ax12.stem(self.frequencies, self.amp_frequencies, linefmt=None, markerfmt='C0.', basefmt=None,bottom=0.0, use_line_collection=True)

            self.ax13.cla()
            self.ax13.set_title('Frequency energy distribute')
            self.ax13.pie(self.sizes, explode=self.explode, labels=self.labels, autopct='%1.1f%%',shadow=False, startangle=0)
            self.ax13.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            self.canvas1.draw()  # 绘制

        except Exception as e:
            print(e)

    def draw_wavelet(self):
        try:
            self.get_derivatives()

            self.ax21.set_title('zero-order derivative')
            self.ax21.plot(self.time_array, self.zero_derivative)

            self.ax22.set_title('First-order derivative')
            self.ax22.plot(self.time_array, self.first_derivative)

            self.ax23.set_title('Second-order derivative')
            self.ax23.plot(self.time_array, self.second_derivative)
            self.canvas2.draw()  # 绘制

        except Exception as e:
            print(e)

    def reset_cmd(self):
        try:
            # 清除内容
            self.ax11.cla()
            self.ax12.cla()
            self.ax13.cla()
            self.ax21.cla()
            self.ax22.cla()
            self.ax23.cla()
            # 重新设置标题
            self.ax11.set_title('Time-domain signal')
            self.ax12.set_title('Frequency-domain signal')
            self.ax13.set_title('Frequency energy distribute')
            self.ax21.set_title('zero-order derivative')
            self.ax22.set_title('First-order derivative')
            self.ax23.set_title('Second-order derivative')
            # 重新绘制
            self.canvas1.draw()
            self.canvas2.draw()
        except Exception as e:
            print(e)

    def set_scale(self, value):
        try:
        	font_time = QtGui.QFont()
        	font_time.setFamily("Times New Roman")
        	font_time.setPointSize(10)
        	self.label_scale.setFont(font_time)
        	self.label_scale.setText(str('Scale=')+str(value))
        	self.scale_parameter = float(value)

        except Exception as e:
            print(e)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())