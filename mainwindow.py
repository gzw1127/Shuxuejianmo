# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(947, 907)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 20, 851, 651))
        self.groupBox.setObjectName("groupBox")
        self.Imglabel = QtWidgets.QLabel(self.groupBox)
        self.Imglabel.setGeometry(QtCore.QRect(30, 50, 781, 571))
        self.Imglabel.setText("")
        self.Imglabel.setObjectName("Imglabel")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(350, 690, 521, 151))
        self.groupBox_2.setObjectName("groupBox_2")
        self.Infolabel = QtWidgets.QLabel(self.groupBox_2)
        self.Infolabel.setGeometry(QtCore.QRect(20, 40, 461, 61))
        self.Infolabel.setText("")
        self.Infolabel.setObjectName("Infolabel")
        self.fileBtn = QtWidgets.QPushButton(self.centralwidget)
        self.fileBtn.setGeometry(QtCore.QRect(40, 690, 271, 151))
        self.fileBtn.setObjectName("fileBtn")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 947, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "花卉识别"))
        self.groupBox.setTitle(_translate("MainWindow", "显示窗口"))
        self.groupBox_2.setTitle(_translate("MainWindow", "结果显示窗口"))
        self.fileBtn.setText(_translate("MainWindow", "选择文件"))
