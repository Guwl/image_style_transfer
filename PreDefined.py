from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGroupBox, QScrollArea, QPushButton, QProgressBar, QMessageBox, QInputDialog
from PyQt5.Qt import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QBasicTimer
from ImageViewer import *
from eval_pretrained import *
import subprocess
import os
from weibo import *
import requests
from requests_toolbelt import MultipartEncoder

global flag 
flag = 0    # The index (start from 1) of the selected label, 0 - nothing selected

class picLabel(QLabel):
    """ Define a label in the preStyle group
    """

    def __init__(self, parent, index, picName = "default.png", width = 200, height = 200):
        super(QLabel, self).__init__()
        self.clicked = False
        self.parent = parent
        self.index = index
        self.picName = picName
        self.width = width
        self.height = height
        self.initUI()

    def initUI(self):
        self.pic = QImage(self.picName)
        self.setFixedSize(self.width, self.height)
        self.pic = self.pic.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(self.pic))
        self.setStyleSheet("border: 3px solid rgba(0, 0, 255, 0%)") 

    def mousePressEvent(self, e):
        """ The behaviour of choosing a label
        """
        if self.index == 0: return
        global flag
        if flag == 0:
            flag = self.index
            self.setStyleSheet("border: 3px solid rgba(0, 0, 255, 100%)") 
        elif flag == self.index:
            flag = 0
            self.setStyleSheet("border: 3px solid rgba(0, 0, 255, 0%)") 
        else:
            self.parent.VBoxGroups[flag - 1].prevPic.setStyleSheet("border: 3px solid rgba(0, 0, 255, 0%)")
            self.setStyleSheet("border: 3px solid rgba(0, 0, 255, 100%)") 
            flag = self.index

class preStyle(QGroupBox):
    """ Defines a predefined style, save the previous selected one
    """

    def __init__(self, parent, index, picName = "default.png", width = 200, height = 200):
        super(QGroupBox, self).__init__()
        self.prevPic = picLabel(parent, index, picName, width, height)
        self.initUI()

    def initUI(self):
        self.vbox = QVBoxLayout()
        self.vbox.addStretch(1)
        self.vbox.addWidget(self.prevPic)
        self.vbox.addStretch(1)
        self.setLayout(self.vbox)

class myPreDefined(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__()
        self.picPath = ["pics/style1.jpg", "pics/style2.jpg", "pics/style3.jpg", "pics/style4.jpg", "pics/style5.jpg", "pics/style6.jpg", "pics/style7.jpg", "pics/style8.jpg", "pics/style9.jpg"]
        self.stylePath = ["pretrained_model/style1", "pretrained_model/style2", "pretrained_model/style3", "pretrained_model/style4", "pretrained_model/style5", "pretrained_model/style6", "pretrained_model/style7", "pretrained_model/style8", "pretrained_model/style9"]
        self.nStyle = len(self.picPath)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.tempInputPath = 'temp/input_temp.png'
        self.tempHumanPath = 'temp/human_temp.png'
        self.inputPic = myImageViewer("pics/default.jpg", 320, 240, self.tempInputPath)
        self.humanPic = myImageViewer("pics/default.jpg", 320, 240, self.tempHumanPath)
        # self.choosePos = myImagePicker("pics/null.png", 320, 240, self)
        self.outputPic = myImageResult("pics/default_null.jpg", 320, 240)
        self.defaultInputPath = self.inputPic.getImagePath()
        self.defaultHumanPath = self.humanPic.getImagePath()

        self.VBoxGroups = []
        for i in range(self.nStyle):
            self.VBoxGroups.append(preStyle(self, i + 1, self.picPath[i], 200, 200))

        self.HBoxGroupScroll = QGroupBox()
        self.HBox1 = QHBoxLayout()
        self.HBox1.addStretch(1)
        for group in self.VBoxGroups:
            self.HBox1.addWidget(group)
            self.HBox1.addStretch(1)
        self.HBoxGroupScroll.setLayout(self.HBox1)
        self.styleScroll = QScrollArea()
        self.styleScroll.setWidget(self.HBoxGroupScroll) 
        self.styleScroll.setAutoFillBackground(True)
        self.styleScroll.setWidgetResizable(True)

        self.configButton = QPushButton('配置')
        self.configButton.setFont(QFont("Roman times", 20, QFont.Bold))
        self.configButton.setFixedSize(160, 40)
        self.configButton.clicked.connect(self.configure)
        self.transButton = QPushButton('转换')
        self.transButton.setFont(QFont("Roman times", 20, QFont.Bold))
        self.transButton.setFixedSize(160, 40)
        self.transButton.clicked.connect(self.transfer)
        # self.shareButton = QPushButton('分享')
        # self.shareButton.setFont(QFont("Roman times", 20, QFont.Bold))
        # self.shareButton.setFixedSize(160, 40)
        # self.shareButton.clicked.connect(self.share)
        # self.shareButton.setDisabled(True)
        self.timer = QBasicTimer()
        self.step = 0
        self.HBoxGroupButton = QGroupBox()
        self.HBoxButton = QHBoxLayout()
        self.HBoxButton.addStretch(1)
        self.HBoxButton.addWidget(self.configButton)
        self.HBoxButton.addStretch(1)
        self.HBoxButton.addWidget(self.transButton)
        self.HBoxButton.addStretch(1)
        self.HBoxGroupButton.setLayout(self.HBoxButton)

        self.progBar = QProgressBar()
        self.progBar.setFixedSize(800, 10)
        self.progBar.setMinimum(0)
        self.progBar.setMaximum(100)
        self.HBoxGroupBar = QGroupBox()
        self.HBoxBar = QHBoxLayout()
        self.HBoxBar.addStretch(1)
        self.HBoxBar.addWidget(self.progBar)
        self.HBoxBar.addStretch(1)
        self.HBoxGroupBar.setLayout(self.HBoxBar)

        self.HBoxGroupDown = QGroupBox()
        self.HBox2 = QHBoxLayout()
        self.HBox2.addStretch(1)
        self.HBox2.addWidget(self.inputPic)
        self.HBox2.addStretch(2)
        self.HBox2.addWidget(self.humanPic)
        self.HBox2.addStretch(2)
        # self.HBox2.addWidget(self.choosePos)
        # self.HBox2.addStretch(2)
        self.HBox2.addWidget(self.outputPic)
        self.HBox2.addStretch(2)
        self.HBoxGroupDown.setLayout(self.HBox2)
        self.picScroll = QScrollArea()
        self.picScroll.setWidget(self.HBoxGroupDown) 
        self.picScroll.setAutoFillBackground(True)
        self.picScroll.setWidgetResizable(True)

        self.VBox = QVBoxLayout()
        self.VBox.addStretch(0.2)
        self.VBox.addWidget(self.styleScroll)
        self.VBox.addWidget(self.HBoxGroupBar)
        self.VBox.addWidget(self.picScroll)
        self.VBox.addStretch(0.2)
        self.VBox.addWidget(self.HBoxGroupButton)
        self.VBox.addStretch(0.2)
        self.setLayout(self.VBox)

    def configure(self):
        self.inputPath = self.inputPic.getImagePath()
        self.humanPath = self.humanPic.getImagePath()
        if self.inputPath == self.defaultInputPath:
            reply = QMessageBox.warning(self, "错误", "请选择内容图片！", QMessageBox.Ok, QMessageBox.Ok)
            return
        if self.humanPath == self.defaultHumanPath:
            reply = QMessageBox.warning(self, "错误", "请选择人物图片! ", QMessageBox.Ok, QMessageBox.Ok)
            return
        if not hasattr(self, 'confWidget') or self.inputPic.checkChange() or self.humanPic.checkChange():
            self.confWidget = confWidget(self.inputPath, self.humanPath)
        self.confWidget.show()

    def transfer(self):
        global flag
        if flag == 0:
            reply = QMessageBox.warning(self, "错误", "请选择一种风格！", QMessageBox.Ok, QMessageBox.Ok)
            return
        if self.inputPic.getImagePath() == self.defaultInputPath:
            reply = QMessageBox.warning(self, "错误", "请选择内容图片！", QMessageBox.Ok, QMessageBox.Ok)
            return
        if self.humanPic.getImagePath() == self.defaultHumanPath:
            reply = QMessageBox.warning(self, "错误", "请选择人物图片! ", QMessageBox.Ok, QMessageBox.Ok)
            return
        if not hasattr(self, 'confWidget') or self.inputPic.checkChange() or self.humanPic.checkChange():
            reply = QMessageBox.warning(self, "错误", "请先进行配置! ", QMessageBox.Ok, QMessageBox.Ok)
            return
        if not self.timer.isActive():
            self.inputPath = self.inputPic.getImagePath()
            self.humanPath = self.humanPic.getImagePath()
            self.outName = self.inputPic.getImageName() + "_transfered_style%d"%flag
            while os.path.exists("result/" + self.outName + ".jpg"):
                self.outName = self.outName + "_1"
            self.outPath = "result/" + self.outName + ".jpg"
            self.width = self.inputPic.getImageWidth()
            self.height = self.inputPic.getImageHeight()
            self.ratio = float(self.width) / float(self.height)
            self.maxheight = 600
            self.maxwidth = 800
            if self.height > self.maxheight:
                self.height = self.maxheight
                self.width = int(self.height * self.ratio)
            if self.width > self.maxwidth:
                self.width = self.maxwidth
                self.height = int(self.width / self.ratio)
            self.style = self.stylePath[flag - 1]
            # self.shareButton.setDisabled(True)
            self.timer.start(100, self)

            iters = self.confWidget.getConfiguration()
            self.ps = subprocess.Popen("python3 eval_pretrained.py " + self.inputPath + " " + self.humanPath + " " + self.outPath + " " + self.style \
                + " " + str(self.height) + " " + str(self.width) + " " + str(xpos) + " " + str(ypos) + " " + str(resizeRatio) + " " + str(alpha) + " " + str(iters), shell = True)

    def timerEvent(self, a):
        if self.step >= 100:
            self.timer.stop()
            # self.shareButton.setDisabled(False)
            self.step = 0
            return
        self.step += 0.01 
        if self.ps.poll() is not None:
            self.step = 100
            self.progBar.setMinimum(0)
            self.progBar.setMaximum(100)
            self.progBar.setValue(0)
            self.outputPic.changeImage(self.outPath)
            self.parent.newHistory()
            return
        self.progBar.setMaximum(0)

    def share(self):
        aFile = open("token", "r")
        aToken = aFile.read()
        aToken = aToken.strip('\n')
        aFile.close()
        if aToken == "0":
            openBrowser()
            code, ok1 = QInputDialog.getText(self, '关联微博账号', '请输入url中的code：')
            if ok1:
                ok2, token = get_token(code)
                if ok2:
                    tokenFile = open("token", "w")
                    tokenFile.write(token)
                    tokenFile.close()
                    text, ok3 = QInputDialog.getText(self, '分享到微博', '请输入您想说的话：')
                    if ok3:
                         post_a_pic(self.outPath, token, text)
                else:
                    reply = QMessageBox.warning(self, "错误", "请输入正确的code！", QMessageBox.Ok, QMessageBox.Ok)
        else:
            text, ok3 = QInputDialog.getText(self, '分享到微博', '请输入您想说的话：')
            if ok3:
                post_a_pic(self.outPath, aToken, text)
