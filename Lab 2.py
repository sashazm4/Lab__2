import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout, QSizePolicy, QFrame, QComboBox, QCheckBox
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
from PyQt5.QtCore import Qt, QBuffer
from PIL import Image
import numpy as np
import io
import cv2
import matplotlib.pyplot as plt

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('MyApp')
        self.setGeometry(100, 100, 800, 400)

        self.layout = QHBoxLayout()

        self.column0_layout = QVBoxLayout()
        self.image_button_1 = QPushButton('Choose source image', self)
        self.image_button_2 = QPushButton('Choose target image', self)
        self.image_button_3 = QPushButton('Get result', self)
        self.image_button_4 = QPushButton('Show histogram for source image', self)
        self.image_button_5 = QPushButton('Show histogram for target image', self)
        self.image_button_6 = QPushButton('Show histogram for result image', self)
        self.image_button_7 = QPushButton("Save result image", self)
        self.convert_type_2 = QComboBox(self)
        
        self.convert_type_2.addItem("RGB")
        self.convert_type_2.addItem("LAB")
        self.convert_type_2.addItem("HSL")
        self.convert_type_2.addItem("HSV")
        
        self.column0_layout.addWidget(self.convert_type_2, 1)
      
        self.result_1_channel = QCheckBox("R")
        self.result_2_channel = QCheckBox("G")
        self.result_3_channel = QCheckBox("B")
        self.result_1_channel.setChecked(True)
        self.result_2_channel.setChecked(True)
        self.result_3_channel.setChecked(True)
        self.convert_type_2.currentIndexChanged.connect(self.get_colorspace)

        self.column0_layout.addWidget(self.result_1_channel)
        self.column0_layout.addWidget(self.result_2_channel)
        self.column0_layout.addWidget(self.result_3_channel)

        self.column0_layout.addWidget(self.image_button_1)
        self.column0_layout.addWidget(self.image_button_2)
        self.column0_layout.addWidget(self.image_button_3)
        self.column0_layout.addWidget(self.image_button_4)
        self.column0_layout.addWidget(self.image_button_5)
        self.column0_layout.addWidget(self.image_button_6)
        self.column0_layout.addWidget(self.image_button_7)

        self.image_button_4.clicked.connect(self.Show_histogram_for_source_image)
        self.image_button_5.clicked.connect(self.Show_histogram_for_target_image)
        self.image_button_6.clicked.connect(self.Show_histogram_for_result_image)
        self.image_button_7.clicked.connect(self.save_image)

        self.layout.addLayout(self.column0_layout)

        line1 = QFrame()
        line1.setFrameShape(QFrame.VLine)
        line1.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(line1)


    
        self.column1_layout = QVBoxLayout()
        self.source_image_label = QLabel('Source Image', self)
        self.source_image_label.setAlignment(Qt.AlignHCenter)
        self.source_image_pixmap_label = QLabel('Waiting for image', self)
        self.source_image_pixmap_label.setAlignment(Qt.AlignCenter)
        self.source_image_pixmap_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_button_1.clicked.connect(self.load_source_image)

        self.column1_layout.addWidget(self.source_image_label)
        self.column1_layout.addWidget(self.source_image_pixmap_label)
    
        self.layout.addLayout(self.column1_layout)
        

        line2 = QFrame()
        line2.setFrameShape(QFrame.VLine)
        line2.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(line2)


        self.column2_layout = QVBoxLayout()
        self.target_image_label = QLabel('Target image', self)
        self.target_image_label.setAlignment(Qt.AlignHCenter)
        self.target_image_pixmap_label = QLabel('Waiting for image', self)
        self.target_image_pixmap_label.setAlignment(Qt.AlignCenter)
        self.target_image_pixmap_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_button_2.clicked.connect(self.load_target_image)

        self.column2_layout.addWidget(self.target_image_label)
        self.column2_layout.addWidget(self.target_image_pixmap_label)
    
        self.layout.addLayout(self.column2_layout)
        self.setLayout(self.layout)

        line3 = QFrame()
        line3.setFrameShape(QFrame.VLine)
        line3.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(line3)

        self.column3_layout = QVBoxLayout()
        self.result_image_label = QLabel('Result', self)
        self.result_image_label.setAlignment(Qt.AlignHCenter)
        self.result_image_pixmap_label = QLabel('Result', self)
        self.result_image_pixmap_label.setAlignment(Qt.AlignCenter)
        self.result_image_pixmap_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_button_3.clicked.connect(self.combineImages)
        self.column3_layout.addWidget(self.result_image_label)
        self.column3_layout.addWidget(self.result_image_pixmap_label)
    
        self.layout.addLayout(self.column3_layout)



  
    def load_source_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Выберите изображение', '', 'Изображения (*.png *.jpg *.bmp *.gif *.jpeg);;Все файлы (*)', options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.source_image_pixmap_label.setPixmap(pixmap.scaledToWidth(self.source_image_pixmap_label.width()))
    
    def load_target_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Выберите изображение', '', 'Изображения (*.png *.jpg *.bmp *.gif *.jpeg);;Все файлы (*)', options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.target_image_pixmap_label.setPixmap(pixmap.scaledToWidth(self.target_image_pixmap_label.width()))

    
    def pixmapToImage(self, pixmap):
        image = pixmap.toImage()
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        image.save(buffer, "PNG")
        pil_image = Image.open(io.BytesIO(buffer.data()))
        pil_image = pil_image.convert('RGB')
        image_array = np.array(pil_image)
        return image_array

    def convertToColorspace(self):
        pixmap = self.result_image_pixmap_label.pixmap()
        if pixmap is None:
            return

        image = self.pixmapToImage(pixmap)

        if self.convert_type_2.currentIndex() == 0:
            colorspace = 'RGB'
            converted_image = image.copy()
        if self.convert_type_2.currentIndex() == 1:
            colorspace = 'LAB'
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        if self.convert_type_2.currentIndex() == 2:
            colorspace = 'HLS'
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        if self.convert_type_2.currentIndex() == 3:
            colorspace = 'HSV'
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        i = np.hstack((converted_image[:, :, 0], converted_image[:, :, 1], converted_image[:, :, 2]))
        cv2.imshow(colorspace, cv2.resize(i, (i.shape[1] // 2, i.shape[0] // 2)))
    
    def get_colorspace(self):
        
        if self.convert_type_2.currentIndex() == 0:
            colorspace = 'RGB'
                
        if self.convert_type_2.currentIndex() == 1:
            colorspace = 'LAB'
                
        if self.convert_type_2.currentIndex() == 2:
            colorspace = 'HLS'
            
        if self.convert_type_2.currentIndex() == 3:
            colorspace = 'HSV'
        
        self.result_1_channel.setText(colorspace[0])
        self.result_2_channel.setText(colorspace[1])
        self.result_3_channel.setText(colorspace[2])
        
    def combineImages(self):
        palette_pixmap = self.source_image_pixmap_label.pixmap()
        source_pixmap = self.target_image_pixmap_label.pixmap()

        if source_pixmap is not None and palette_pixmap is not None:
            image_1 = self.pixmapToImage(palette_pixmap)
            if self.convert_type_2.currentIndex() == 0:
                image_1 = image_1.copy()
            if self.convert_type_2.currentIndex() == 1:
                image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2LAB)
            if self.convert_type_2.currentIndex() == 2:
                image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2HLS)
            if self.convert_type_2.currentIndex() == 3:
                image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2HSV)
            if self.convert_type_2.currentIndex() == 4:
                image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2YCrCb)

            image_2 = self.pixmapToImage(source_pixmap)
            if self.convert_type_2.currentIndex() == 0:
                image_2 = image_2.copy()
            if self.convert_type_2.currentIndex() == 1:
                image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2LAB)
            if self.convert_type_2.currentIndex() == 2:
                image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2HLS)
            if self.convert_type_2.currentIndex() == 3:
                image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2HSV)
            if self.convert_type_2.currentIndex() == 4:
                image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2YCrCb)
            
            image_3 = self.modify_image(image_1, image_2)

            if self.convert_type_2.currentIndex() == 0:
                image_3 = image_3.copy()
            if self.convert_type_2.currentIndex() == 1:
                image_3 = cv2.cvtColor(image_3, cv2.COLOR_LAB2RGB)
            if self.convert_type_2.currentIndex() == 2:
                image_3 = cv2.cvtColor(image_3, cv2.COLOR_HLS2RGB)
            if self.convert_type_2.currentIndex() == 3:
                image_3 = cv2.cvtColor(image_3, cv2.COLOR_HSV2RGB)
            if self.convert_type_2.currentIndex() == 4:
                image_3 = cv2.cvtColor(image_3, cv2.COLOR_YCrCb2RGB)

            pixmap = self.imageToPixmap(image_3)
            self.result_image_pixmap_label.setPixmap(pixmap.scaledToWidth(self.result_image_pixmap_label.width()))

    def imageToPixmap(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_image)
        return q_pixmap

    def pixmapToImage(self, pixmap):
        image = pixmap.toImage()
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        image.save(buffer, "PNG")
        pil_image = Image.open(io.BytesIO(buffer.data()))
        pil_image = pil_image.convert('RGB')
        image_array = np.array(pil_image)
        return image_array
    
    def get_modified_channel(self, source_image, target_image, channel_index):
        source_mean = np.mean(source_image[:, :, channel_index])
        target_mean = np.mean(target_image[:, :, channel_index])
        source_std = np.std(source_image[:, :, channel_index])
        target_std = np.std(target_image[:, :, channel_index])
        
        modified_channel = (target_image[:, :, channel_index] - target_mean) * (source_std / target_std) + source_mean
        return np.clip(modified_channel, 0, 255).astype(np.uint8)  
        
    def modify_image(self, source_image, target_image):
        if self.result_1_channel.isChecked():
            channel_0 = self.get_modified_channel(source_image, target_image, 0)    
        else:
            channel_0 = target_image[:, :, 0]
        if self.result_2_channel.isChecked():
            channel_1 = self.get_modified_channel(source_image, target_image, 1)    
        else:
            channel_1 = target_image[:, :, 1]
        if self.result_3_channel.isChecked():
            channel_2 = self.get_modified_channel(source_image, target_image, 2)    
        else:
            channel_2 = target_image[:, :, 2]

        
        modified_image = np.copy(target_image)
        modified_image = cv2.merge((channel_0, channel_1, channel_2))
        
        return modified_image.astype(np.uint8)
    
    def buildHistogram(self, image, colorspace):
        blue_channel, green_channel, red_channel = cv2.split(image)

        plt.figure(colorspace, figsize=(12, 4))

        plt.subplot(131)
        plt.hist(blue_channel.ravel(), bins=256, color='red', alpha=0.7, rwidth=0.9)
        plt.title(f'Histogram of {colorspace[0]} Channel')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.subplot(132)
        plt.hist(green_channel.ravel(), bins=256, color='green', alpha=0.7, rwidth=0.9)
        plt.title(f'Histogram of {colorspace[1]} Channel')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.subplot(133)
        plt.hist(red_channel.ravel(), bins=256, color='blue', alpha=0.7, rwidth=0.9)
        plt.title(f'Histogram of {colorspace[2]} Channel')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def Show_histogram_for_source_image(self):        
        pixmap = self.source_image_pixmap_label.pixmap()
        if pixmap is None:
            return

        image = self.pixmapToImage(pixmap)

        if self.convert_type_2.currentIndex() == 0:
            colorspace = 'RGB'
            converted_image = image.copy()
        if self.convert_type_2.currentIndex() == 1:
            colorspace = 'LAB'
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        if self.convert_type_2.currentIndex() == 2:
            colorspace = 'HLS'
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        if self.convert_type_2.currentIndex() == 3:
            colorspace = 'HSV'
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        self.buildHistogram(converted_image, colorspace)

    def Show_histogram_for_target_image(self):        
        pixmap = self.target_image_pixmap_label.pixmap()
        if pixmap is None:
            return

        image = self.pixmapToImage(pixmap)

        if self.convert_type_2.currentIndex() == 0:
            colorspace = 'RGB'
            converted_image = image.copy()
        if self.convert_type_2.currentIndex() == 1:
            colorspace = 'LAB'
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        if self.convert_type_2.currentIndex() == 2:
            colorspace = 'HLS'
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        if self.convert_type_2.currentIndex() == 3:
            colorspace = 'HSV'
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        self.buildHistogram(converted_image, colorspace)

    def Show_histogram_for_result_image(self):        
        pixmap = self.result_image_pixmap_label.pixmap()
        if pixmap is None:
            return

        image = self.pixmapToImage(pixmap)

        if self.convert_type_2.currentIndex() == 0:
            colorspace = 'RGB'
            converted_image = image.copy()
        if self.convert_type_2.currentIndex() == 1:
            colorspace = 'LAB'
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        if self.convert_type_2.currentIndex() == 2:
            colorspace = 'HLS'
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        if self.convert_type_2.currentIndex() == 3:
            colorspace = 'HSV'
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        self.buildHistogram(converted_image, colorspace)

    def save_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        image = self.result_image_pixmap_label.pixmap()
        if file_name:
            pixmap = image
            if pixmap:
                pixmap.save(file_name)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
