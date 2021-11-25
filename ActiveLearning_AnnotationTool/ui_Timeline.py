# -*- coding: utf-8 -*-


from PyQt5.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
    QRadialGradient)
from PyQt5.QtWidgets import *


class Ui_Timeline(QDialog):
    def setupUi(self, Dialog):
        if Dialog.objectName():
            Dialog.setObjectName('Dialog')
        Dialog.resize(407, 512)
        self.verticalLayout = QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName('verticalLayout')
        self.u_timeline_table = QTableWidget(Dialog)
        self.u_timeline_table.setObjectName('u_timeline_table')

        self.verticalLayout.addWidget(self.u_timeline_table)


        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate('Timeline', 'Timeline', None))
    # retranslateUi
