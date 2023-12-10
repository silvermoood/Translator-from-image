from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from neural import *
import sys
import onnxruntime as onnxr



class ProgramWindow(QMainWindow):
    def __init__(self):

        super().__init__()
        uic.loadUi('main.ui', self)

        # подключение кнопок
        self.image_load_button.clicked.connect(self.load_image)
        self.translate_button.clicked.connect(self.translate_text)
        self.phrase = ''

        # загрузка onnx model
        self.model = onnxr.InferenceSession("model.onnx")

    def load_image(self):
        dialog = QFileDialog()
        dialog.setNameFilter("Картинки (*.png)")
        dialog.exec_()
        selected_file = dialog.selectedFiles()
        if selected_file:
            pixmap = QPixmap(selected_file[0])
            pixmap = pixmap.scaled(self.picture.size(), aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.picture.setPixmap(pixmap)
            fraze = recognition(self.model, selected_file[0])
            self.phrase = fraze
            self.source_text_line.setText(fraze)

    def translate_text(self):
        self.translated_text_line.setText(translator(self.phrase))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ProgramWindow()
    win.show()
    sys.exit(app.exec_())
