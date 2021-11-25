# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitledCmUrFD.ui'
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


class Ui_ActiveLearningDialog(object):
    def setupUi(self, ActiveLearningDialog):
        if ActiveLearningDialog.objectName():
            ActiveLearningDialog.setObjectName(u"ActiveLearningDialog")
        ActiveLearningDialog.resize(619, 108)
        self.verticalLayout = QVBoxLayout(ActiveLearningDialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(ActiveLearningDialog)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.u_dataset_folder = QLineEdit(ActiveLearningDialog)
        self.u_dataset_folder.setObjectName(u"u_dataset_folder")

        self.horizontalLayout.addWidget(self.u_dataset_folder)

        self.u_choose_folder = QPushButton(ActiveLearningDialog)
        self.u_choose_folder.setObjectName(u"u_choose_folder")

        self.horizontalLayout.addWidget(self.u_choose_folder)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(ActiveLearningDialog)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.u_process = QProgressBar(ActiveLearningDialog)
        self.u_process.setObjectName(u"u_process")
        self.u_process.setValue(24)

        self.horizontalLayout_2.addWidget(self.u_process)

        self.u_refine = QPushButton(ActiveLearningDialog)
        self.u_refine.setObjectName(u"u_refine")

        self.horizontalLayout_2.addWidget(self.u_refine)


        self.verticalLayout.addLayout(self.horizontalLayout_2)


        self.retranslateUi(ActiveLearningDialog)

        QMetaObject.connectSlotsByName(ActiveLearningDialog)
    # setupUi

    def retranslateUi(self, ActiveLearningDialog):
        ActiveLearningDialog.setWindowTitle(QCoreApplication.translate("ActiveLearningDialog", u"Active Learning", None))
        self.label.setText(QCoreApplication.translate("ActiveLearningDialog", u"Dataset Folder: ", None))
        self.u_choose_folder.setText(QCoreApplication.translate("ActiveLearningDialog", u"Browser..", None))
        self.label_2.setText(QCoreApplication.translate("ActiveLearningDialog", u"Process: ", None))
        self.u_refine.setText(QCoreApplication.translate("ActiveLearningDialog", u"Refine", None))
    # retranslateUi

