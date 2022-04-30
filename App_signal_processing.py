#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# NetPanelAnalysis_V1_0_2主函数

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow,QTableWidget
from Ui_signal_processing import Ui_MainWindow
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
#from PyQt5.QtWidgets import *
#from PyQt5.QtCore import *


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)

        self.pushButton_1.clicked.connect(self.draw_fourier)
        self.pushButton_2.clicked.connect(self.draw_wavelet)
        self.pushButton_3.clicked.connect(self.reset_cmd)


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
        self.x1 = np.asarray(xData, dtype='float')
        self.y1 = np.asarray(yData, dtype='float')

    def draw_fourier(self):
        try:
            x = [i + 1 for i in range(5)]  # x轴
            y = np.random.randint(0, 10, 5)  # y轴
            self.getTableData()

            self.ax11.cla()
            self.ax11.set_title('Time-domain signal')
            self.ax11.plot(x, y)

            self.ax12.cla()
            self.ax12.set_title('Frequency-domain signal')
            self.ax12.plot(x, y)

            self.ax13.cla()
            self.ax13.set_title('Frequency energy distribute')
            self.ax13.plot(self.x1, self.y1)

            self.canvas1.draw()  # 绘制

        except Exception as e:
            print(e)

    def draw_wavelet(self):
        try:
            x = [i + 1 for i in range(5)]  # x轴
            y = np.random.randint(0, 10, 5)  # y轴

            self.ax21.set_title('zero-order derivative')
            self.ax21.plot(x, y)

            self.ax22.set_title('First-order derivative')
            self.ax22.plot(x, y)

            self.ax23.set_title('Second-order derivative')
            self.ax23.plot(x, y)
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())