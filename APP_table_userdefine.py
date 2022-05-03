# 重写tableWidget键盘复制、粘贴、删除快捷键
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class MyTable(QTableWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRowCount(10)
        self.setColumnCount(10)
        # etc.

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_C:  # Ctrl + C复制表格内容
            try:
                indexes = self.selectedIndexes()  # 获取表格对象中被选中的数据索引列表
                indexes_dict = {}
                for index in indexes:  # 遍历每个单元格
                    row, column = index.row(), index.column()  # 获取单元格的行号，列号
                    if row in indexes_dict.keys():
                        indexes_dict[row].append(column)
                    else:
                        indexes_dict[row] = [column]

                # 将数据表数据用制表符(\t)和换行符(\n)连接，使其可以复制到excel文件中
                text_str = ''
                for row, columns in indexes_dict.items():
                    row_data = ''
                    for column in columns:
                        data = self.item(row, column).text()
                        if row_data:
                            row_data = row_data + '\t' + data
                        else:
                            row_data = data

                    if text_str:
                        text_str = text_str + '\n' + row_data
                    else:
                        text_str = row_data

            except BaseException as e:
                print(e)
            clipboard = QApplication.clipboard()  # 获取剪贴板
            clipboard.setText(text_str)  # 内容写入剪贴板

        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_V:  # Ctrl + V粘贴表格内容
            try:
                indexes = self.selectedIndexes()
                for index in indexes:
                    index = index
                    break  # 用于获得选中表格中的第一个单元格（左上角）
                num_r, num_c = index.row(), index.column()
                text_str = QApplication.clipboard().text()
                ls_row = text_str.split('\n')[0:-1]
                ls_col = []
                for row in ls_row:
                    ls_col.append(row.split('\t'))
                rows = len(ls_row)  # 获取行数，每行以\n分隔
                columns = len(ls_col[0])
                for row in range(rows):
                    for column in range(columns):
                        item = QTableWidgetItem()
                        item.setText((str(ls_col[row][column])))
                        self.setItem(row + num_r, column + num_c, item)
            except Exception as e:
                print('粘贴时发生错误')

        elif event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            try:
                indexes = self.selectedIndexes()
                for index in indexes:
                    row, column = index.row(), index.column()
                    item = None
                    self.setItem(row, column, item)
            except Exception as e:
                print(e)

        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_X:  # Ctrl + X粘贴表格内容
            try:
                indexes = self.selectedIndexes()  # 获取表格对象中被选中的数据索引列表
                indexes_dict = {}
                for index in indexes:  # 遍历每个单元格
                    row, column = index.row(), index.column()  # 获取单元格的行号，列号
                    if row in indexes_dict.keys():
                        indexes_dict[row].append(column)
                    else:
                        indexes_dict[row] = [column]

                # 将数据表数据用制表符(\t)和换行符(\n)连接，使其可以复制到excel文件中
                text_str = ''
                for row, columns in indexes_dict.items():
                    row_data = ''
                    for column in columns:
                        data = self.item(row, column).text()
                        if row_data:
                            row_data = row_data + '\t' + data
                        else:
                            row_data = data

                    if text_str:
                        text_str = text_str + '\n' + row_data
                    else:
                        text_str = row_data

            except BaseException as e:
                print(e)
            clipboard = QApplication.clipboard()  # 获取剪贴板
            clipboard.setText(text_str)  # 内容写入剪贴板

            try:
                indexes = self.selectedIndexes()
                for index in indexes:
                    row, column = index.row(), index.column()
                    item = None
                    self.setItem(row, column, item)
            except Exception as e:
                print(e)
        else:
            pass
