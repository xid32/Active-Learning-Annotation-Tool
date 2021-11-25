from PyQt5 import QtCore, QtGui, QtWidgets

class ProxyStyle(QtWidgets.QProxyStyle):
    def drawControl(self, element, option, painter, widget):
        if element == QtWidgets.QStyle.CE_ProgressBar:
            super(ProxyStyle, self).drawControl(element, option, painter, widget)
            if hasattr(option, 'alternative'):
                alternative = option.alternative

                last_value = option.progress
                last_pal = option.palette
                last_rect = option.rect

                option.progress = alternative
                pal = QtGui.QPalette()
                # alternative color
                pal.setColor(QtGui.QPalette.Highlight, QtCore.Qt.red)
                option.palette = pal
                option.rect = self.subElementRect(QtWidgets.QStyle.SE_ProgressBarContents, option, widget)
                self.proxy().drawControl(QtWidgets.QStyle.CE_ProgressBarContents, option, painter, widget)

                option.progress = last_value 
                option.palette = last_pal
                option.rect = last_rect
            return
        super(ProxyStyle, self).drawControl(element, option, painter, widget)

class ProgressBar(QtWidgets.QProgressBar):
    def paintEvent(self, event):
        painter =  QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionProgressBar()
        if hasattr(self, 'alternative'):
            opt.alternative = self.alternative()
        self.initStyleOption(opt)
        painter.drawControl(QtWidgets.QStyle.CE_ProgressBar, opt)

    @QtCore.pyqtSlot(int)
    def setAlternative(self, value):
        self._alternative = value
        self.update()

    def alternative(self):
        if not hasattr(self, '_alternative'):
            self._alternative = 0
        return self._alternative

class Actions(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Progress Bar')

        self.objectsToProcess = 100
        self.objectsProcessed = 0
        self.objectsVerified = 0

        self.progress_bar = ProgressBar(maximum=self.objectsToProcess)
        self.process_btn = QtWidgets.QPushButton('Process')
        self.verify_btn = QtWidgets.QPushButton('Verify')

        self.process_btn.clicked.connect(self.process)
        self.verify_btn.clicked.connect(self.verify)

        lay = QtWidgets.QGridLayout(self)
        lay.addWidget(self.progress_bar, 0, 0, 1, 2)
        lay.addWidget(self.process_btn, 1, 0)
        lay.addWidget(self.verify_btn, 1, 1)

    def process(self):
        if self.objectsProcessed + 1 < self.objectsToProcess:
            self.objectsProcessed += 1
            self.progress_bar.setValue(self.objectsProcessed)

    def verify(self):
        if self.objectsVerified < self.objectsProcessed:
            self.objectsVerified += 1
            self.progress_bar.setAlternative(self.objectsVerified)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(ProxyStyle(app.style()))
    w = Actions()
    w.show()
    sys.exit(app.exec_())