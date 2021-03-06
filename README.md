# Image Style Transfer
一个图像风格转换软件

An image style transfer software

小组成员：刘昊伟(组长)，陈铭，刘晓明，许立，张舒楠

Members: Howie Lau, Ming Chen, Xiaoming Liu, Li Xu, Shunan Zhang

## 简介
这是一个转换图像风格的软件，主要提供以下两种转换风格：
+ 预设风格转换
+ 任意风格转换

预设风格转换类似滤镜，可以将图片转换为预先设定好的某种风格，时间较快。

任意风格转换则需要用户自定义输入风格图片，然后将内容图片转换为风格图片上的风格，时间较慢。

## Introduction

This software can change the style of an image with 2 main functions:
+ Predefined style transfer
+ Arbitary style transfer

Predefined style transfer is similar to image filter, which is changing the image style through a pre-trained model with fast speed.

Arbitary style transfer need the user to specify the style image, and transfer the content image with lower speed.

## 使用

本程序暂只支持Python3。

受GitHub文件上传大小限制，无法将预训练好的模型上传到这里。

完整软件大小约2GB，同时我们也提供了只有任意风格转换的程序，大小约1GB。

百度网盘链接: https://pan.baidu.com/s/1kVSqt3x  密码:snpq

直接下载后运行。

`cd /path/to/image_style_transfer`

`python3 MainWindow.py`

**注意**：使用本软件需要安装`PyQt5`，`Tensorflow`和`Pillow`。

**暂未测试Linux的使用**

在Windows下使用Python编译器(如PyCharm)直接运行`MainWindow.py`即可。

**若要在Windows上使用，需要将`PreDefined.py`中179行和`UserDefined.py`中118行命令的`python3`改为`python`，同时将`MainWindow.py`第40行改为：**

`with open("template.qss", encoding="UTF-8") as file:`

## Usage

This software only supports Python3 for now.

Limited by the file storage of GitHub, we can't upload the pre-trained model here.

The whole package is about 2GB, and we also provide arbitary-only package, which is about 1GB.

Baidu Netdisk Link: https://pan.baidu.com/s/1kVSqt3x Password: snpq

`cd /path/to/image_style_transfer`

`python3 MainWindow.py`

**Dependencies**: `PyQt5`, `Tensorflow`, `Pillow`

**Usage on Linux is not tested yet**

If used on Windows, execute `MainWindow.py` with Python complier (PyCharm, etc.).

**If you are using it on Windows, you need to change the command `python3` to `python` on line 179 of `PreDefined.py` and line 118 of `UserDefined.py`, and change line 40 of `MainWindow.py` to:**

`with open("template.qss", encoding="UTF-8") as file:`
