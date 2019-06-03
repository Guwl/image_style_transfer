import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QTabWidget, QMainWindow, QApplication
import PreDefined
import UserDefined
import History

class myTabWidget(QTabWidget):
    
    def __init__(self):
        super(QTabWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.preDefined = PreDefined.myPreDefined(self)
        # self.userDefined = UserDefined.myUserDefined(self)
        self.history = History.myHistory()
        self.addTab(self.preDefined, '预设风格')
        # self.addTab(self.userDefined, '自定义风格')
        self.addTab(self.history, '历史记录')
    
    def newHistory(self):
        self.history = History.myHistory()
        self.removeTab(2)
        self.addTab(self.history, '历史记录')

class myMainWindow(QMainWindow):

    def __init__(self):
        super(QMainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.move(300, 100)
        self.resize(900, 600)
        self.setWindowTitle('Image Style Transfer')
        self.setCentralWidget(myTabWidget())

if __name__ == '__main__':
    with open("template.qss") as file:
        style = file.readlines()
        style = "".join(style).strip("\n")
    app = QApplication(sys.argv)
    app.setStyleSheet(style)
    window = myMainWindow()
    window.show()
    sys.exit(app.exec_())
