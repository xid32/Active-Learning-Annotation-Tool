

from PyQt5.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
    QRadialGradient, QPen)
from PyQt5.QtWidgets import *





class FrameBar(QWidget):

    def __init__(self, parents=None):
        super(FrameBar, self).__init__(parents)

        self.setObjectName('FrameBar')
        self.setMinimumHeight(5)
        self.setMaximumWidth(4096)

    def paintEvent(self, event):
        
 
        painter = QPainter()
        painter.begin(self)
        pen = QPen()
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setWidth(2)
        pen.setStyle(Qt.SolidLine)
        pen.setColor("#fff")
        painter.setPen(pen)

        painter.drawRect(4,4,10,10)

        painter.end()