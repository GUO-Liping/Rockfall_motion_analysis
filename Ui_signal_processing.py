# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_signal_processing.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
import matplotlib.pyplot as plt
from table_userdefine import MyTable


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.tableWidget = MyTable(self.splitter)
        self.tableWidget.setMinimumSize(QtCore.QSize(200, 700))
        self.tableWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.tableWidget.setBaseSize(QtCore.QSize(200, 2000))
        self.tableWidget.setLineWidth(1)
        self.tableWidget.setMidLineWidth(1)
        self.tableWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tableWidget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.tableWidget.setAutoScrollMargin(15)
        self.tableWidget.setRowCount(100000)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(8)
        item.setFont(font)
        self.tableWidget.setItem(0, 0, item)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(80)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(30)
        self.tableWidget.verticalHeader().setDefaultSectionSize(30)
        self.widget = QtWidgets.QWidget(self.splitter)
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_up = QtWidgets.QGroupBox(self.widget)
        self.groupBox_up.setMinimumSize(QtCore.QSize(900, 300))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setItalic(True)
        self.groupBox_up.setFont(font)
        self.groupBox_up.setObjectName("groupBox_up")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.groupBox_up)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        # 设置画布部分
        self.fig1 = plt.figure(figsize=(12, 4), dpi=80)
        self.canvas1 = FC(self.fig1)
        # 添加第一个图
        self.ax11 = self.fig1.add_subplot(131)
        self.ax11.set_title('line 1')
        # 添加第二个图
        self.ax12 = self.fig1.add_subplot(132)
        self.ax12.set_title('line 2')
        # 添加第三个图
        self.ax13 = self.fig1.add_subplot(133)
        self.ax13.set_title('line 3')

        self.fig1.subplots_adjust(left=0.05,     # 调整最左边坐标系到画布左边框的间距（参数 0~1）
                            right=0.98,    # 调整最右边坐标系到画布右边框的间距（参数 0~1）
                            top=0.85,      # 调整最上面坐标系到画布顶部框的间距（参数 0~1）
                            bottom=0.15,   # 调整最下面坐标系到画布底部框的间距（参数 0~1）
                            wspace=0.2,   # 调整坐标系之间的水平间隔
                            hspace=0.2,   # 调整坐标系之间的垂直间隔
                            )

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.canvas1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_5.addWidget(self.canvas1)

        self.verticalLayout.addWidget(self.groupBox_up)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.pushButton_1 = QtWidgets.QPushButton(self.widget)
        self.pushButton_1.setMinimumSize(QtCore.QSize(300, 35))
        self.pushButton_1.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.pushButton_1.setFont(font)
        self.pushButton_1.setObjectName("pushButton_1")
        self.horizontalLayout.addWidget(self.pushButton_1)

        self.pushButton_2 = QtWidgets.QPushButton(self.widget)
        self.pushButton_2.setMinimumSize(QtCore.QSize(300, 35))
        self.pushButton_2.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)

        self.pushButton_3 = QtWidgets.QPushButton(self.widget)
        self.pushButton_3.setMinimumSize(QtCore.QSize(300, 35))
        self.pushButton_3.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)

        self.verticalLayout.addLayout(self.horizontalLayout)
        self.groupBox_low = QtWidgets.QGroupBox(self.widget)
        self.groupBox_low.setMinimumSize(QtCore.QSize(900, 300))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setItalic(True)
        self.groupBox_low.setFont(font)
        self.groupBox_low.setObjectName("groupBox_low")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_low)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        # 设置画布部分
        self.fig2 = plt.figure(figsize=(12, 4), dpi=80)
        self.canvas2 = FC(self.fig2)
        # 添加第一个图
        self.ax21 = self.fig2.add_subplot(131)
        self.ax21.set_title('line 1')
        # 添加第二个图
        self.ax22 = self.fig2.add_subplot(132)
        self.ax22.set_title('line 2')
        # 添加第三个图
        self.ax23 = self.fig2.add_subplot(133)
        self.ax23.set_title('line 3')

        self.fig2.subplots_adjust(left=0.05,     # 调整最左边坐标系到画布左边框的间距（参数 0~1）
                            right=0.98,    # 调整最右边坐标系到画布右边框的间距（参数 0~1）
                            top=0.85,      # 调整最上面坐标系到画布顶部框的间距（参数 0~1）
                            bottom=0.15,   # 调整最下面坐标系到画布底部框的间距（参数 0~1）
                            wspace=0.2,   # 调整坐标系之间的水平间隔
                            hspace=0.2,   # 调整坐标系之间的垂直间隔
                            )
        self.horizontalLayout_4.addWidget(self.canvas2)


        self.verticalLayout.addWidget(self.groupBox_low)
        self.horizontalLayout_2.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "1"))
        item = self.tableWidget.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "2"))
        item = self.tableWidget.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "3"))
        item = self.tableWidget.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "4"))
        item = self.tableWidget.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "5"))
        item = self.tableWidget.verticalHeaderItem(5)
        item.setText(_translate("MainWindow", "6"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Time"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Signal"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.setSortingEnabled(__sortingEnabled)
        self.groupBox_up.setTitle(_translate("MainWindow", "Fourier transform"))
        self.pushButton_1.setText(_translate("MainWindow", "Frequency analysis"))
        self.pushButton_2.setText(_translate("MainWindow", "Derivatives calculation"))
        self.pushButton_3.setText(_translate("MainWindow", "Reset"))
        self.groupBox_low.setTitle(_translate("MainWindow", "Gaussian wavelet transform"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
