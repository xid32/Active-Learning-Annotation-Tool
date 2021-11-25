# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitledvPLiWS.ui'
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


from FrameBar import FrameBar

class Ui_FrameBarTestDialog(object):
    def setupUi(self, FrameBarTestDialog):
        if FrameBarTestDialog.objectName():
            FrameBarTestDialog.setObjectName(u"FrameBarTestDialog")
        FrameBarTestDialog.resize(717, 42)
        self.verticalLayout = QVBoxLayout(FrameBarTestDialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.u_frame_bar = FrameBar(FrameBarTestDialog)
        self.u_frame_bar.setObjectName(u"u_frame_bar")

        self.verticalLayout.addWidget(self.u_frame_bar)


        self.retranslateUi(FrameBarTestDialog)

        QMetaObject.connectSlotsByName(FrameBarTestDialog)
    # setupUi

    def retranslateUi(self, FrameBarTestDialog):
        FrameBarTestDialog.setWindowTitle(QCoreApplication.translate("FrameBarTestDialog", u"*", None))
    # retranslateUi

