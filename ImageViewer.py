from PyQt5.QtWidgets import QLabel, QFileDialog, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.Qt import Qt
from NewWindow import *

def getName(path):
    name = ''
    path = path.split('/')[-1]
    path = path.split('.')[0:-1]
    for i in path:
        name += i
    return name

def setImage(label, newImage):
    label.imagePath = newImage
    label.name = getName(label.imagePath)
    label.image = QImage(newImage)
    label.origWidth = label.image.width()
    label.origHeight = label.image.height()
    label.image = label.image.scaled(label.width, label.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    label.setPixmap(QPixmap.fromImage(label.image))
    label.setAlignment(Qt.AlignCenter)


class myImageViewer(QLabel):

    def __init__(self, image, width, height, parent=None):
        super(QLabel, self).__init__()
        self.parent = parent
        self.imagePath = image
        self.name = getName(self.imagePath)
        self.image = QImage(image)
        self.origWidth = self.image.width()
        self.origHeight = self.image.height()
        self.width = width
        self.height = height
        self.setFixedSize(self.width, self.height)
        self.image = self.image.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(self.image)) 
        self.setAlignment(Qt.AlignCenter)
    
    def mousePressEvent(self, e):
        newImage = QFileDialog.getOpenFileName(self, 'open file', './', 'Images (*.png *.jpg)')[0]
        if len(newImage):
            setImage(self, newImage)
            if self.parent is not None:
                setImage(self.parent.choosePos, newImage)

    def getImagePath(self):
        return self.imagePath

    def getImageName(self):
        return self.name
    
    def getImageWidth(self):
        return self.origWidth
    
    def getImageHeight(self):
        return self.origHeight

class myImagePicker(QLabel):

    def __init__(self, image, width, height, parent=None):
        super(QLabel, self).__init__()
        self.parent = parent
        self.imagePath = image
        self.name = getName(self.imagePath)
        self.image = QImage(image)
        self.origWidth = self.image.width()
        self.origHeight = self.image.height()
        self.width = width
        self.height = height
        self.setFixedSize(self.width, self.height)
        self.image = self.image.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(self.image)) 
        self.setAlignment(Qt.AlignCenter)
    
    def mousePressEvent(self, e):
        if not hasattr(self, 'posWidget'):
            self.posWidget = posWidget(self.imagePath)
        self.posWidget.show()

    def getImagePath(self):
        return self.imagePath

    def getImageName(self):
        return self.name
    
    def getImageWidth(self):
        return self.origWidth
    
    def getImageHeight(self):
        return self.origHeight

class myImageResult(QLabel):

    def __init__(self, image, width, height, parent=None):
        super(QLabel, self).__init__()
        self.parent = parent
        self.imagePath = image
        self.name = getName(self.imagePath)
        self.origImage = QImage(image)
        self.origWidth = self.origImage.width()
        self.origHeight = self.origImage.height()
        self.width = width
        self.height = height
        self.setFixedSize(self.width, self.height)
        self.image = self.origImage.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.setAlignment(Qt.AlignCenter)
    
    def mousePressEvent(self, e):
        if not hasattr(self, 'editWidget'):
            self.editWidget = editWidget(self.imagePath)
        self.editWidget.show()
    
    def changeImage(self, image):
        self.imagePath = image
        self.name = getName(self.imagePath)
        self.origImage = QImage(image)
        self.origWidth = self.origImage.width()
        self.origHeight = self.origImage.height()
        self.setFixedSize(self.width, self.height)
        self.image = self.origImage.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.setAlignment(Qt.AlignCenter)

class myImageBlank(QLabel):

    def __init__(self, image, width, height):
        super(QLabel, self).__init__()
        self.imagePath = image
        self.name = getName(self.imagePath)
        self.origImage = QImage(image)
        self.origWidth = self.origImage.width()
        self.origHeight = self.origImage.height()
        self.width = width
        self.height = height
        self.setFixedSize(self.width, self.height)
        self.image = self.origImage.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.setAlignment(Qt.AlignCenter)
