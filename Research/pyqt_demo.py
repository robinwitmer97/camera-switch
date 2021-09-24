import sys
import cv2
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QPushButton
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

    def terminate(self):
        self.running = False
        self.cap.release()
        super().terminate()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 500
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.webcam.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Webcam
        self.webcam = QLabel(self)
        self.webcam.move(150, 10)
        self.webcam.resize(640, 480)
        self.webcamthread = Thread(self)
        self.webcamthread.changePixmap.connect(self.setImage)
        self.webcamthread.start()

        # Button
        self.button = QPushButton(self)
        self.button.move(10, 10)
        self.button.resize(130, 30)
        self.button.setText('Test')
        self.button.clicked.connect(lambda:print('Test was pressed!'))

        self.show()
    
    def close(self) -> bool:
        self.webcamthread.terminate()

        return super().close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())