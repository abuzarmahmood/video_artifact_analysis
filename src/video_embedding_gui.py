import sys
import json
import numpy as np
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QSlider, QLabel)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class VideoEmbeddingViewer(QMainWindow):
    def __init__(self, video_path, embedding_path):
        super().__init__()
        self.video_path = video_path
        
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Load embedding
        self.embedding = np.load(embedding_path)
        
        # Initialize marked timepoints and current frame
        self.marked_timepoints = []
        self.current_frame = 0
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Video and Embedding Viewer')
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create video display
        self.video_label = QLabel()
        layout.addWidget(self.video_label)
        
        # Create matplotlib figure for embedding
        self.figure = Figure(figsize=(8, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.update_embedding_plot()
        layout.addWidget(self.canvas)
        
        # Create slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.total_frames - 1)
        self.slider.valueChanged.connect(self.slider_changed)
        layout.addWidget(self.slider)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)
        
        self.mark_button = QPushButton('Mark Timepoint')
        self.mark_button.clicked.connect(self.mark_timepoint)
        controls_layout.addWidget(self.mark_button)
        
        self.save_button = QPushButton('Save Timepoints')
        self.save_button.clicked.connect(self.save_timepoints)
        controls_layout.addWidget(self.save_button)
        
        layout.addLayout(controls_layout)
        
        # Setup timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False
        
        # Initial frame display
        self.current_frame = 0
        self.update_frame()
        
        self.setGeometry(100, 100, 800, 600)
        
    def update_embedding_plot(self):
        self.ax.clear()
        self.ax.plot(self.embedding)
        self.ax.set_yscale('symlog', linthresh=0.1)
        
        # Plot marked timepoints
        for timepoint in self.marked_timepoints:
            self.ax.axvline(x=timepoint, color='r', alpha=0.5)
            
        # Plot current position
        self.ax.axvline(x=self.current_frame, color='g', alpha=0.5)
        
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Embedding Value')
        self.canvas.draw()
        
    def update_frame(self):
        if self.current_frame >= self.total_frames:
            self.current_frame = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if ret:
            # Convert frame to RGB for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            
            # Convert to QImage and display
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
            
            # Update slider and embedding plot
            self.slider.setValue(self.current_frame)
            self.update_embedding_plot()
            
            if self.is_playing:
                self.current_frame += 1
        
    def slider_changed(self, value):
        self.current_frame = value
        self.update_frame()
        
    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.timer.start(int(1000 / self.fps))
            self.play_button.setText('Pause')
        else:
            self.timer.stop()
            self.play_button.setText('Play')
            
    def mark_timepoint(self):
        if self.current_frame not in self.marked_timepoints:
            self.marked_timepoints.append(self.current_frame)
            self.update_embedding_plot()
            
    def save_timepoints(self):
        output_path = self.video_path.rsplit('.', 1)[0] + '_timepoints.json'
        with open(output_path, 'w') as f:
            json.dump({
                'video_path': self.video_path,
                'timepoints': sorted(self.marked_timepoints)
            }, f, indent=2)
            
    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Video and Embedding Viewer')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('embedding_path', help='Path to the embedding NPY file')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    viewer = VideoEmbeddingViewer(args.video_path, args.embedding_path)
    viewer.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
