# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untixtledSXHoGZ.ui'
##
## Created by: Qt User Interface Compiler version 5.14.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
    QRadialGradient)
from PyQt5.QtWidgets import *


class Ui_ProcessingDialog(object):
    def setupUi(self, ProcessingDialog):
        if ProcessingDialog.objectName():
            ProcessingDialog.setObjectName(u"ProcessingDialog")
        ProcessingDialog.resize(178, 63)
        self.horizontalLayout = QHBoxLayout(ProcessingDialog)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.label = QLabel(ProcessingDialog)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)


        self.retranslateUi(ProcessingDialog)

        QMetaObject.connectSlotsByName(ProcessingDialog)
    # setupUi

    def retranslateUi(self, ProcessingDialog):
        ProcessingDialog.setWindowTitle(QCoreApplication.translate("ProcessingDialog", u"Processing", None))
        self.label.setText(QCoreApplication.translate("ProcessingDialog", u"Processing...", None))
    # retranslateUi


class ProcessingDialog(Ui_ProcessingDialog, QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)


