#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# NetPanelAnalysis_V1_0_2主函数

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from Ui_signal_processing import Ui_MainWindow
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)

        self.pushButton_1.clicked.connect(self.draw_test)
        self.pushButton_2.clicked.connect(self.draw_test)

    def draw_test(self):
        try:
            # 如果是点击画一条线的按钮，就先清除内容
            if self.sender() == self.pushButton_1:
                ax = self.ax11
                ax.cla()
                self.ax11.set_title('draw a line')
            # 如果是点击重复画图的按钮，就不清除原先的内容
            elif self.sender() == self.pushButton_2:
                ax = self.ax12
            # 绘图部分
            x = [i + 1 for i in range(5)]  # x轴
            y = np.random.randint(0, 10, 5)  # y轴
            ax.plot(x, y)  # 折线图
            self.canvas1.draw()  # 绘制
        except Exception as e:
            print(e)

    def reset_cmd(self):
        try:
            # 清除内容
            self.ax.cla()
            self.ax11.cla()
            # 重新设置标题
            self.ax.set_title('draw a line')
            self.ax12.set_title('draw lines')
            # 重新绘制
            self.canvas.draw()
        except Exception as e:
            print(e)


        ## 设置画布部分
        #self.fig = plt.figure(figsize=(10, 4), dpi=80)
        #self.canvas = FC(self.fig)
        ## 添加第一个图
        #self.ax = self.fig.add_subplot(121)
        #self.ax.set_title('draw a line')
        ## 添加第二个图
        #self.ax1 = self.fig.add_subplot(122)
        #self.ax1.set_title('draw lines')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())