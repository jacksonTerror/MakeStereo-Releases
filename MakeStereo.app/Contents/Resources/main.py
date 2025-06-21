import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QProgressBar, 
                            QListWidget, QHBoxLayout, QStyle, QStyleOption,
                            QListWidgetItem, QComboBox, QFrame, QDialog, QTextEdit,
                            QAbstractItemView, QStyledItemDelegate, QGridLayout)
from PyQt6.QtCore import Qt, QMimeData, pyqtSignal, QThread, QSize, QRect
from PyQt6.QtGui import QPalette, QColor, QPainter, QFont, QPixmap, QPen
import soundfile as sf
import numpy as np
from typing import Union, Tuple, Optional
from audio_processor import AudioProcessor

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())

def resource_path(relative_path):
    """Get absolute path to resource, works for dev, py2app, and PyInstaller."""
    # PyInstaller
    if hasattr(sys, '_MEIPASS'):  # type: ignore[attr-defined]
        return os.path.join(sys._MEIPASS, relative_path)
    # py2app
    if getattr(sys, 'frozen', False):
        return os.path.join(os.environ.get('RESOURCEPATH', ''), relative_path)
    # Development
    return os.path.join(os.path.abspath("."), relative_path)

class OutputFormatFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #323232;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 10px;
            }
            QLabel {
                color: white;
            }
            QComboBox {
                background-color: #3b3b3b;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                min-width: 100px;
            }
            QPushButton {
                background-color: #2196F3;
                border: none;
                border-radius: 4px;
                color: white;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton.success {
                background-color: #4CAF50;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Output directory section
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("Output Directory:")
        self.dir_label.setFont(QFont("Arial", 11))
        self.dir_path = QLabel("Not set")
        self.dir_path.setFont(QFont("Arial", 11))
        self.same_as_input = QPushButton("Same as Input")
        self.same_as_input.clicked.connect(self.set_same_as_input)
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(self.dir_path, 1)  # 1 is stretch factor
        dir_layout.addWidget(self.same_as_input)
        layout.addLayout(dir_layout)
        
        # Format selection section
        format_layout = QHBoxLayout()
        self.format_label = QLabel("Output Format:")
        self.format_label.setFont(QFont("Arial", 11))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Same as Input", "WAV", "MP3"])
        self.format_combo.setFont(QFont("Arial", 11))
        format_layout.addWidget(self.format_label)
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        layout.addLayout(format_layout)

    def set_directory(self, path):
        self.dir_path.setText(path)
        self.dir_path.setStyleSheet("color: #4CAF50;")  # Green text when set
        
    def set_same_as_input(self):
        self.same_as_input_clicked.emit()
        
    def get_output_format(self):
        return self.format_combo.currentText()

    same_as_input_clicked = pyqtSignal()  # New signal for same as input button

class AudioProcessorThread(QThread):
    progress_updated = pyqtSignal(int, int)  # current_file_progress, overall_progress
    processing_complete = pyqtSignal()
    status_updated = pyqtSignal(str)
    failed_files_updated = pyqtSignal(list)  # New signal for failed files
    
    def __init__(self, mode, files, output_directory, output_format):
        super().__init__()
        self.mode = mode
        self.files = files
        self.output_directory = output_directory
        self.output_format = output_format
        self.is_cancelled = False
        self.failed_files = []  # Track failed files
        
    def process_audio_file(self, file_path):
        """Process audio file using soundfile"""
        try:
            self.status_updated.emit(f"Reading {os.path.basename(file_path)}...")
            data, rate = sf.read(file_path)
            info = sf.info(file_path)
            self.status_updated.emit(f"File info: {rate}Hz, format={info.format}, subtype={info.subtype}")
            self.status_updated.emit(f"Data shape: {data.shape}")
            return data, rate, info
        except Exception as e:
            self.status_updated.emit(f"Error reading {os.path.basename(file_path)}: {str(e)}")
            self.failed_files.append((file_path, str(e)))  # Add to failed files
            self.failed_files_updated.emit(self.failed_files)
            return None

    def save_audio_file(self, audio_data, output_path):
        """Save audio file using soundfile"""
        try:
            if not isinstance(audio_data, tuple) or len(audio_data) != 3:
                self.status_updated.emit("Invalid audio data format")
                return False
                
            data, rate, info = audio_data
            if not isinstance(data, np.ndarray):
                self.status_updated.emit("Invalid audio data type")
                return False
                
            self.status_updated.emit(f"Saving to: {output_path}")
            self.status_updated.emit(f"Audio data shape: {data.shape}, rate: {rate}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert to float32 if needed
            if data.dtype != np.float32:
                data = data.astype(np.float32)
                
            # Save the file
            sf.write(output_path, data, rate, format=info.format, subtype=info.subtype)
            self.status_updated.emit("Save successful")
            return True
        except Exception as e:
            self.status_updated.emit(f"Error saving {os.path.basename(output_path)}: {str(e)}")
            return False

    def get_output_extension(self, input_path):
        """Determine output file extension based on settings"""
        if self.output_format == "Same as Input":
            return os.path.splitext(input_path)[1]
        elif self.output_format == "WAV":
            return ".wav"
        else:  # MP3
            return ".mp3"

    def get_unique_output_path(self, base_path):
        """Get a unique output path by appending a number if file exists"""
        directory = os.path.dirname(base_path)
        basename = os.path.basename(base_path)
        name, ext = os.path.splitext(basename)
        
        # Add _Stereo suffix for stereo output
        if self.mode == "Mono to Stereo" and not name.endswith("_Stereo"):
            name = f"{name}_Stereo"
        
        counter = 0
        while True:
            if counter == 0:
                new_path = os.path.join(directory, f"{name}{ext}")
            else:
                new_path = os.path.join(directory, f"{name}.{counter:02d}{ext}")
            
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    def run(self):
        if self.mode == "Stereo to Mono":
            self.process_stereo_to_mono()
        else:  # Mono to Stereo
            self.process_mono_to_stereo()
        
        if not self.is_cancelled:
            self.processing_complete.emit()

    def process_stereo_to_mono(self):
        """Process stereo files to mono"""
        total_files = len(self.files)
        for i, file_path in enumerate(self.files):
            if self.is_cancelled:
                break
                
            self.status_updated.emit(f"\n=== Processing {os.path.basename(file_path)} ===")
            
            # Process the file
            result = self.process_audio_file(file_path)
            if result is None:
                continue
                
            data, rate, info = result
            
            # Check if it's stereo
            if len(data.shape) != 2 or data.shape[1] != 2:
                self.status_updated.emit("Not a stereo file, skipping...")
                self.failed_files.append((file_path, "Not a stereo file – will be ignored"))
                self.failed_files_updated.emit(self.failed_files)
                continue
            
            # Create output paths with unique names
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            ext = self.get_output_extension(file_path)
            left_path = self.get_unique_output_path(os.path.join(self.output_directory, f"{base_name}-L{ext}"))
            right_path = self.get_unique_output_path(os.path.join(self.output_directory, f"{base_name}-R{ext}"))
            
            try:
                # Save left channel
                self.status_updated.emit("\nProcessing left channel...")
                if not self.save_audio_file((data[:, 0], rate, info), left_path):
                    self.failed_files.append((file_path, "Failed to save left channel"))
                    self.failed_files_updated.emit(self.failed_files)
                    continue
                    
                # Save right channel
                self.status_updated.emit("\nProcessing right channel...")
                if not self.save_audio_file((data[:, 1], rate, info), right_path):
                    self.failed_files.append((file_path, "Failed to save right channel"))
                    self.failed_files_updated.emit(self.failed_files)
                    continue
                
                # Update progress
                self.progress_updated.emit(100, int((i + 1) * 100 / total_files))
            except Exception as e:
                self.failed_files.append((file_path, str(e)))
                self.failed_files_updated.emit(self.failed_files)
                continue
            
        self.status_updated.emit("\nProcessing complete!")
        if self.failed_files:
            self.status_updated.emit(f"\nFailed files: {len(self.failed_files)}")

    def process_mono_to_stereo(self):
        """Process mono files to stereo"""
        # Filter to only mono files
        mono_files = []
        for f in self.files:
            try:
                info = sf.info(f)
                if info.channels == 1:
                    mono_files.append(f)
                else:
                    print(f"[DEBUG] Skipping non-mono file for pairing: {f} (channels={info.channels})")
            except Exception as e:
                print(f"[DEBUG] Error reading file info for {f}: {e}")
        print("[DEBUG] Mono files for pairing:", mono_files)
        # Group mono files into pairs
        pairs = []
        processed = set()
        for file in mono_files:
            if file in processed:
                continue
            base_path = os.path.splitext(file)[0]
            potential_matches = []
            # Remove common L/R indicators to find base name
            base_name = base_path.replace('-L', '').replace('_L', '')\
                                    .replace(' L ', ' ').replace('-R', '')\
                                    .replace('_R', '').replace(' R ', ' ')\
                                    .replace('Left', '').replace('Right', '')\
                                    .replace('left', '').replace('right', '')
            # Find potential matches
            for other_file in mono_files:
                if other_file == file or other_file in processed:
                    continue
                other_base = os.path.splitext(other_file)[0]
                other_base = other_base.replace('-L', '').replace('_L', '')\
                                         .replace(' L ', ' ').replace('-R', '')\
                                         .replace('_R', '').replace(' R ', ' ')\
                                         .replace('Left', '').replace('Right', '')\
                                         .replace('left', '').replace('right', '')
                if base_name == other_base:
                    potential_matches.append(other_file)
            print(f"[DEBUG] File {file} potential mono matches: {potential_matches}")
            if len(potential_matches) == 1:
                # Check which is L and which is R
                if ('L' in base_path.upper() and 'R' in os.path.splitext(potential_matches[0])[0].upper()):
                    pairs.append((file, potential_matches[0]))
                    print(f"[DEBUG] Pair found: L={file}, R={potential_matches[0]}")
                elif ('R' in base_path.upper() and 'L' in os.path.splitext(potential_matches[0])[0].upper()):
                    pairs.append((potential_matches[0], file))
                    print(f"[DEBUG] Pair found: L={potential_matches[0]}, R={file}")
                processed.add(file)
                processed.add(potential_matches[0])
        print(f"[DEBUG] Total mono pairs found: {len(pairs)}")
        if not pairs:
            self.status_updated.emit("No valid mono pairs found for processing.")
            print("[DEBUG] No valid pairs found. Exiting.")
            return
        # Process each pair
        total_pairs = len(pairs)
        for i, (left_path, right_path) in enumerate(pairs):
            if self.is_cancelled:
                break
            self.status_updated.emit(f"\n=== Processing pair {i+1}/{total_pairs} ===")
            self.status_updated.emit(f"Left: {os.path.basename(left_path)}")
            self.status_updated.emit(f"Right: {os.path.basename(right_path)}")
            print(f"[DEBUG] Processing pair {i+1}/{total_pairs}: L={left_path}, R={right_path}")
            # Process left channel
            left_result = self.process_audio_file(left_path)
            if left_result is None:
                self.status_updated.emit(f"Failed to read left channel: {left_path}")
                print(f"[DEBUG] Failed to read left channel: {left_path}")
                continue
            # Process right channel
            right_result = self.process_audio_file(right_path)
            if right_result is None:
                self.status_updated.emit(f"Failed to read right channel: {right_path}")
                print(f"[DEBUG] Failed to read right channel: {right_path}")
                continue
            left_data, left_rate, info = left_result
            right_data, right_rate, _ = right_result
            # Check sample rates match
            if left_rate != right_rate:
                self.status_updated.emit(f"Sample rate mismatch: {left_rate} vs {right_rate}")
                print(f"[DEBUG] Sample rate mismatch: {left_rate} vs {right_rate}")
                continue
            # Create stereo array
            self.status_updated.emit("\nCombining channels...")
            stereo_data = np.column_stack((left_data, right_data))
            self.status_updated.emit(f"Stereo data shape: {stereo_data.shape}")
            # Create output path with unique name and _Stereo suffix
            base_name = os.path.splitext(os.path.basename(left_path))[0]
            base_name = base_name.replace('-L', '').replace('_L', '')\
                                    .replace(' L ', ' ').replace('Left', '')\
                                    .replace('left', '')
            ext = self.get_output_extension(left_path)
            output_path = self.get_unique_output_path(os.path.join(self.output_directory, f"{base_name}{ext}"))
            print(f"[DEBUG] Output path: {output_path}")
            # Save stereo file
            if not self.save_audio_file((stereo_data, left_rate, info), output_path):
                self.status_updated.emit(f"Failed to save stereo file: {output_path}")
                print(f"[DEBUG] Failed to save stereo file: {output_path}")
                continue
            print(f"[DEBUG] Successfully saved stereo file: {output_path}")
            # Update progress
            self.progress_updated.emit(100, int((i + 1) * 100 / total_pairs))
        self.status_updated.emit("\nProcessing complete!")
        print("[DEBUG] Processing complete!")

class DropArea(QWidget):
    filesDropped = pyqtSignal(list)  # Signal for dropped files
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 200)
        
        # Style
        self.normal_style = """
            QWidget {
                background-color: #2a2a2a;
                border: 2px dashed #666;
                border-radius: 12px;
            }
            QFrame#icon_frame {
                border: none;
                background: transparent;
            }
            QLabel#icon_label {
                background-color: transparent;
                border: none;
                padding: 0px;
                margin: 0px;
            }
            QLabel#text_label, QLabel#format_label {
                border: none;
                background: transparent;
            }
        """
        self.drag_style = """
            QWidget {
                background-color: #323232;
                border: 2px dashed #2196F3;
                border-radius: 12px;
            }
            QFrame#icon_frame {
                border: 2px solid #2196F3;
                border-radius: 62px;
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.5, stop:0 #2a2a2a, stop:1 #323232);
            }
            QLabel#icon_label {
                background-color: transparent;
                border: none;
                padding: 0px;
                margin: 0px;
            }
            QLabel#text_label, QLabel#format_label {
                border: none;
                background: transparent;
            }
        """
        self.setStyleSheet(self.normal_style)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Icon frame (for border effect)
        self.icon_frame = QFrame()
        self.icon_frame.setObjectName("icon_frame")
        self.icon_frame.setFixedSize(124, 124)
        frame_layout = QVBoxLayout(self.icon_frame)
        frame_layout.setContentsMargins(2, 2, 2, 2)
        frame_layout.setSpacing(0)
        
        # Icon label
        self.icon_label = QLabel()
        self.icon_label.setObjectName("icon_label")
        icon_path = resource_path("icon/Make120.png")
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            self.icon_label.setPixmap(pixmap)
        else:
            self.icon_label.setText("🎵")
            self.icon_label.setStyleSheet("font-size: 48px;")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setFixedSize(120, 120)
        frame_layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Text label
        self.text_label = QLabel("Drop stereo audio files here")
        self.text_label.setObjectName("text_label")
        self.text_label.setStyleSheet("color: #888; font-size: 14px;")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Format label
        self.format_label = QLabel("Supported formats: WAV, MP3")
        self.format_label.setObjectName("format_label")
        self.format_label.setStyleSheet("color: #666; font-size: 12px;")
        self.format_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add widgets to layout
        layout.addStretch()
        layout.addWidget(self.icon_frame, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.text_label)
        layout.addWidget(self.format_label)
        layout.addStretch()

    def dragEnterEvent(self, event):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.wav', '.mp3')):
                    event.acceptProposedAction()
                    self.setStyleSheet(self.drag_style)
                    self.text_label.setStyleSheet("color: #2196F3; font-size: 14px;")
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave event"""
        super().dragLeaveEvent(event)
        self.setStyleSheet(self.normal_style)
        self.text_label.setStyleSheet("color: #888; font-size: 14px;")

    def dropEvent(self, event):
        """Handle drop event"""
        files = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.wav', '.mp3')):
                files.append(file_path)
        
        if files:
            self.filesDropped.emit(files)
            
        self.setStyleSheet(self.normal_style)
        self.text_label.setStyleSheet("color: #888; font-size: 14px;")

class FileListItem(QWidget):
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)  # Adjusted horizontal padding
        layout.setSpacing(8)
        
        self.filename_label = QLabel(filename)
        self.filename_label.setFont(QFont("Arial", 12))
        
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setStyleSheet("color: #888;")
        self.status_label.setMinimumWidth(150)
        
        layout.addWidget(self.filename_label)
        layout.addStretch()
        layout.addWidget(self.status_label)
        
        self.setFixedHeight(30)  # Slightly reduced height for tighter look

    def set_status(self, status, is_error=False):
        self.status_label.setText(status)
        if is_error:
            self.status_label.setStyleSheet("color: #f44336; font-size: 12pt;")
        else:
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 12pt;")

class ClickableFrame(QFrame):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

class AudioConverterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_mode = "stereo_to_mono"  # Default mode
        self.dropped_files = []
        self.failed_files = []
        self.audio_processor = AudioProcessor()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Audio Channel Converter')
        # self.setMinimumSize(600, 400)  # Already commented out
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                color: white;
            }
            QPushButton {
                background-color: #2196F3;
                border: none;
                border-radius: 4px;
                color: white;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #666;
            }
            QPushButton.success {
                background-color: #4CAF50;
            }
            QPushButton.error {
                background-color: #f44336;
            }
            QProgressBar {
                border: 2px solid #666;
                border-radius: 5px;
                text-align: center;
                background-color: #2a2a2a;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
            }
            QListWidget {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 0px;
            }
            QListWidget::item {
                color: white;
                padding: 0px;
                margin: 1px;
                border: 1px solid transparent;
            }
            QListWidget::item:selected {
                background-color: rgba(33, 150, 243, 0.2);
                border: 1px solid #2196F3;
                border-radius: 2px;
            }
            QListWidget::item:hover:!selected {
                background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid #555;
                border-radius: 2px;
            }
            QLabel {
                color: white;
            }
        """)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Mode selection - Enhanced
        self.mode_frame = ClickableFrame()
        self.mode_frame.setStyleSheet(self.get_mode_frame_style("Stereo to Mono"))
        mode_layout = QHBoxLayout(self.mode_frame)
        mode_layout.setContentsMargins(10, 10, 10, 10)
        self.mode_label = QLabel("Mode Selector")
        self.mode_label.setFont(QFont("Arial", 14))
        self.mode_button = QPushButton()
        self.mode_button.setCheckable(True)
        self.mode_button.setMinimumHeight(40)
        self.mode_button.setMinimumWidth(220)
        self.mode_button.setStyleSheet(self.get_toggle_style("Stereo to Mono"))
        self.mode_button.clicked.connect(self.toggle_mode)
        self.update_mode_toggle_text("Stereo to Mono")
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.mode_button)
        mode_layout.addStretch()
        layout.addWidget(self.mode_frame)
        self.mode_frame.clicked.connect(self.toggle_mode)

        # Drop area
        self.drop_area = DropArea()
        # self.drop_area.setMinimumHeight(300)  # Removed enforced minimum height
        layout.addWidget(self.drop_area)
        
        # File list with clear buttons
        file_list_layout = QVBoxLayout()
        file_list_header = QHBoxLayout()
        
        file_list_label = QLabel("Files:")
        file_list_label.setFont(QFont("Arial", 12))  # Increased from 11
        
        # Add Clear Invalid, Clear Selected, and Clear All buttons (in that order)
        button_layout = QHBoxLayout()
        self.clear_invalid_btn = QPushButton("Clear Invalid")
        self.clear_invalid_btn.clicked.connect(self.clear_invalid_files)
        self.clear_selected_btn = QPushButton("Clear Selected")
        self.clear_selected_btn.clicked.connect(self.clear_selected_files)
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.clear_file_list)
        button_layout.addWidget(self.clear_invalid_btn)
        button_layout.addWidget(self.clear_selected_btn)
        button_layout.addWidget(self.clear_all_btn)
        
        file_list_header.addWidget(file_list_label)
        file_list_header.addStretch()
        file_list_header.addLayout(button_layout)
        
        file_list_layout.addLayout(file_list_header)
        
        self.file_list = QListWidget()
        self.file_list.setAcceptDrops(True)
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        font = QFont()
        font.setPointSize(12)
        self.file_list.setFont(font)
        self.file_list.setMinimumWidth(600)
        # self.file_list.setMinimumHeight(300)  # Removed enforced minimum height
        
        # Custom item delegate for two-column display
        class TwoColumnDelegate(QStyledItemDelegate):
            def paint(self, painter: QPainter, option: QStyleOption, index) -> None:
                if not index.isValid():
                    return
                
                # Get item data
                model = index.model()
                if not model:
                    return
                    
                filename = index.data(Qt.ItemDataRole.DisplayRole)
                status = index.data(Qt.ItemDataRole.UserRole + 2)
                is_mono_stereo = index.data(Qt.ItemDataRole.UserRole + 3)
                
                if not status:
                    status = ""
                
                # Calculate rectangles for both columns
                rect = option.rect
                filename_rect = QRect(rect.left() + 8, rect.top(), rect.width() - 250, rect.height())
                status_rect = QRect(rect.right() - 242, rect.top(), 234, rect.height())
                
                # Draw selection background if selected
                if option.state & QStyle.StateFlag.State_Selected:
                    painter.fillRect(rect, QColor(33, 150, 243, 51))  # Light blue background
                    painter.setPen(QColor("#2196F3"))  # Blue border
                    painter.drawRect(rect.adjusted(0, 0, -1, -1))
                
                # Get the color based on status
                if "will be ignored" in status:
                    color = QColor("#ff6b6b")  # Red for ignored files
                elif "Stereo file" in status or "Left channel" in status or "Right channel" in status:
                    color = QColor("#4CAF50")  # Green for valid files
                else:
                    color = QColor("#FFFFFF")  # White for other status
                
                # Always show filename in left column
                painter.setPen(QColor("#FFFFFF"))
                painter.drawText(filename_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, filename)
                
                # Show appropriate status in right column
                painter.setPen(color)
                if is_mono_stereo:
                    # For mono->stereo mode, show Left/Right channel status
                    if "Left channel" in status:
                        display_status = "Left channel"
                    elif "Right channel" in status:
                        display_status = "Right channel"
                    elif "No matching pair found" in status:
                        display_status = "No matching pair found – will be ignored"
                    else:
                        display_status = status
                else:
                    # For stereo->mono mode, show stereo status
                    if "will be ignored" in status:
                        display_status = "Not a stereo file – will be ignored"
                    else:
                        display_status = status
                
                painter.drawText(status_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, display_status)
        
        self.file_list.setItemDelegate(TwoColumnDelegate(self.file_list))
        
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 2px;
            }
            QListWidget::item {
                height: 30px;
                border: 1px solid transparent;
                border-radius: 2px;
            }
            QListWidget::item:selected {
                background-color: rgba(33, 150, 243, 0.2);
                border: 1px solid #2196F3;
            }
            QListWidget::item:hover:!selected {
                background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid #555;
            }
        """)
        file_list_layout.addWidget(self.file_list)
        
        layout.addLayout(file_list_layout)
        
        # Output format frame
        self.output_frame = OutputFormatFrame()
        self.output_frame.same_as_input_clicked.connect(self.set_output_to_input)
        layout.addWidget(self.output_frame)
        
        # Progress bars with aligned labels and equal widths
        progress_layout = QGridLayout()
        label_width = 110  # Fixed width for labels
        bar_width = 250    # Fixed/minimum width for progress bars

        self.current_progress_label = QLabel("Current File:")
        self.current_progress_label.setFont(QFont("Arial", 11))
        self.current_progress_label.setFixedWidth(label_width)
        self.current_progress = QProgressBar()
        self.current_progress.setMinimumWidth(bar_width)

        self.overall_progress_label = QLabel("Overall:")
        self.overall_progress_label.setFont(QFont("Arial", 11))
        self.overall_progress_label.setFixedWidth(label_width)
        self.overall_progress = QProgressBar()
        self.overall_progress.setMinimumWidth(bar_width)

        progress_layout.addWidget(self.current_progress_label, 0, 0)
        progress_layout.addWidget(self.current_progress, 0, 1)
        progress_layout.addWidget(self.overall_progress_label, 1, 0)
        progress_layout.addWidget(self.overall_progress, 1, 1)
        layout.addLayout(progress_layout)
        
        # Process status area, aligned like progress bars
        status_layout = QGridLayout()
        self.process_status_label = QLabel("Process Status:")
        self.process_status_label.setFont(QFont("Arial", 11))
        self.process_status_label.setFixedWidth(label_width)
        self.process_status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setStyleSheet("color: #888; padding: 10px 0px 10px 0px; min-height: 60px;")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.process_status_label, 0, 0)
        status_layout.addWidget(self.status_label, 0, 1)
        layout.addLayout(status_layout)
        
        # Buttons with new layout
        button_layout = QHBoxLayout()
        
        # Left side buttons
        left_buttons = QHBoxLayout()
        self.process_btn = QPushButton("Process")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_files)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        
        left_buttons.addWidget(self.process_btn)
        left_buttons.addWidget(self.cancel_btn)
        
        # Right side buttons
        right_buttons = QHBoxLayout()
        self.output_dir_btn = QPushButton("Select Output Directory")
        self.output_dir_btn.clicked.connect(self.select_output_directory)
        
        # Add Open Output Directory button
        self.open_output_btn = QPushButton("Open Output Directory")
        self.open_output_btn.clicked.connect(self.open_output_directory)
        self.open_output_btn.setEnabled(False)
        
        right_buttons.addWidget(self.output_dir_btn)
        right_buttons.addWidget(self.open_output_btn)
        
        button_layout.addLayout(left_buttons)
        button_layout.addStretch()
        button_layout.addLayout(right_buttons)
        
        layout.addLayout(button_layout)
        
        # Connect signals
        self.drop_area.filesDropped.connect(self.handle_dropped_files)
        
        # Initialize mode
        self.toggle_mode()
        
        # Center window
        self.center()
        
    def center(self):
        """Center the window on the screen"""
        frame_geometry = self.frameGeometry()
        screen = self.screen()
        if screen is not None:
            screen_center = screen.availableGeometry().center()
            frame_geometry.moveCenter(screen_center)
            self.move(frame_geometry.topLeft())
        
    def get_mode_frame_style(self, mode):
        if mode == "Stereo to Mono":
            return """
                QFrame {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f39c12, stop:1 #d35400);
                    border: 1px solid #444;
                    border-radius: 4px;
                    padding: 10px;
                }
                QLabel {
                    font-size: 14px;
                    font-weight: bold;
                }
                QComboBox {
                    background-color: #3b3b3b;
                    border: 1px solid #555;
                    border-radius: 3px;
                    padding: 8px;
                    min-width: 200px;
                    font-size: 14px;
                }
            """
        else:  # Mono to Stereo
            return """
                QFrame {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2980b9, stop:1 #3498db);
                    border: 1px solid #444;
                    border-radius: 4px;
                    padding: 10px;
                }
                QLabel {
                    font-size: 14px;
                    font-weight: bold;
                }
                QComboBox {
                    background-color: #3b3b3b;
                    border: 1px solid #555;
                    border-radius: 3px;
                    padding: 8px;
                    min-width: 200px;
                    font-size: 14px;
                }
            """

    def get_toggle_style(self, mode):
        if mode == "Stereo to Mono":
            return """
                QPushButton {
                    background-color: rgba(243, 156, 18, 0.7);
                    color: #fff;
                    border: 2px solid #d35400;
                    border-radius: 20px;
                    font-size: 18px;
                    font-weight: bold;
                    padding: 8px 24px;
                }
                QPushButton:checked {
                    background-color: #2980b9;
                    border: 2px solid #3498db;
                }
                QPushButton:hover {
                    background-color: rgba(243, 156, 18, 0.9);
                }
            """
        else:
            return """
                QPushButton {
                    background-color: rgba(41, 128, 185, 0.7);
                    color: #fff;
                    border: 2px solid #3498db;
                    border-radius: 20px;
                    font-size: 18px;
                    font-weight: bold;
                    padding: 8px 24px;
                }
                QPushButton:!checked {
                    background-color: rgba(243, 156, 18, 0.7);
                    border: 2px solid #d35400;
                }
                QPushButton:hover {
                    background-color: rgba(41, 128, 185, 0.9);
                }
            """

    def update_mode_toggle_text(self, mode):
        if mode == "Stereo to Mono":
            self.mode_button.setText("STEREO → 2 MONO")
            self.mode_button.setChecked(False)
        else:
            self.mode_button.setText("MONO → STEREO")
            self.mode_button.setChecked(True)
        self.mode_button.setStyleSheet(self.get_toggle_style(mode))

    def get_process_btn_style(self, state):
        if state == "green":
            return """
                QPushButton {
                    background-color: #4CAF50;
                    border: none;
                    border-radius: 4px;
                    color: white;
                    padding: 8px 16px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #388E3C;
                }
            """
        elif state == "red":
            return """
                QPushButton {
                    background-color: #f44336;
                    border: none;
                    border-radius: 4px;
                    color: white;
                    padding: 8px 16px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
            """
        else:  # grey
            return """
                QPushButton {
                    background-color: #666666;
                    border: none;
                    border-radius: 4px;
                    color: #999999;
                    padding: 8px 16px;
                    font-size: 14px;
                }
            """

    def update_process_button_state(self, valid_count=None):
        """Enable process button only if there are valid files to process in current mode, or red if valid in other mode. Also set tooltips and status messages."""
        has_valid_current = False
        has_valid_other = False
        valid_pairs = 0
        valid_count_local = 0
        # Use the valid_count if provided, otherwise analyze
        if valid_count is not None:
            if self.current_mode == "mono_to_stereo":
                valid_pairs = valid_count
                has_valid_current = valid_pairs > 0
            else:
                valid_count_local = valid_count
                has_valid_current = valid_count_local > 0
        else:
            if self.current_mode == "mono_to_stereo":
                valid_pairs = self.analyze_mono_pairs()
                has_valid_current = valid_pairs > 0
            else:
                valid_count_local = self.analyze_stereo_files()
                has_valid_current = valid_count_local > 0
        # Check for valid files in the other mode
        if not has_valid_current:
            other_mode = "mono_to_stereo" if self.current_mode == "stereo_to_mono" else "stereo_to_mono"
            if other_mode == "mono_to_stereo":
                processed = set()
                items_files = []
                for i in range(self.file_list.count()):
                    item = self.file_list.item(i)
                    if item is None:
                        continue
                    file = item.data(Qt.ItemDataRole.UserRole)
                    if not file:
                        continue
                    try:
                        info = sf.info(file)
                        if info.channels != 1:
                            continue
                    except Exception:
                        continue
                    items_files.append((item, file))
                pairs = 0
                for idx, (item, file) in enumerate(items_files):
                    if file in processed:
                        continue
                    base_path = os.path.splitext(file)[0]
                    base_name = base_path.replace('-L', '').replace('_L', '')\
                                        .replace(' L ', ' ').replace('-R', '')\
                                        .replace('_R', '').replace(' R ', ' ')\
                                        .replace('Left', '').replace('Right', '')\
                                        .replace('left', '').replace('right', '')
                    for jdx, (other_item, other_file) in enumerate(items_files):
                        if idx == jdx or other_file in processed:
                            continue
                        other_base = os.path.splitext(other_file)[0]
                        other_base = other_base.replace('-L', '').replace('_L', '')\
                                             .replace(' L ', ' ').replace('-R', '')\
                                             .replace('_R', '').replace(' R ', ' ')\
                                             .replace('Left', '').replace('Right', '')\
                                             .replace('left', '').replace('right', '')
                        if base_name == other_base:
                            if ('L' in base_path.upper() and 'R' in os.path.splitext(other_file)[0].upper()) or \
                               ('R' in base_path.upper() and 'L' in os.path.splitext(other_file)[0].upper()):
                                processed.add(file)
                                processed.add(other_file)
                                pairs += 1
                has_valid_other = pairs > 0
            else:
                for i in range(self.file_list.count()):
                    item = self.file_list.item(i)
                    if item is None:
                        continue
                    file = item.data(Qt.ItemDataRole.UserRole)
                    if not file:
                        continue
                    try:
                        info = sf.info(file)
                        if info.channels == 2:
                            has_valid_other = True
                            break
                    except Exception:
                        continue
        # Set button state, tooltip, and status message
        if has_valid_current:
            self.process_btn.setEnabled(True)
            self.process_btn.setStyleSheet(self.get_process_btn_style("green"))
            self.process_btn.setToolTip("Process all valid audio files.")
            self.status_label.setText("Ready to process valid audio files.")
        elif has_valid_other:
            self.process_btn.setEnabled(False)
            self.process_btn.setStyleSheet(self.get_process_btn_style("red"))
            other_mode_label = "MONO → STEREO" if self.current_mode == "stereo_to_mono" else "STEREO → 2 MONO"
            msg = f"No valid files for this mode. Switch to {other_mode_label} to process valid files."
            self.process_btn.setToolTip(msg)
            self.status_label.setText(msg)
        else:
            self.process_btn.setEnabled(False)
            self.process_btn.setStyleSheet(self.get_process_btn_style("grey"))
            self.process_btn.setToolTip("Add valid audio files to enable processing.")
            self.status_label.setText("No valid files. Add audio files to begin.")

    def clear_file_list(self):
        """Clear all files from the list"""
        self.file_list.clear()
        self.dropped_files = []
        self.update_process_button_state()

    def set_output_to_input(self):
        """Set output directory to the same as input directory"""
        if self.dropped_files:
            input_dir = os.path.dirname(self.dropped_files[0])
            self.output_frame.set_directory(input_dir)
            self.process_btn.setEnabled(self.file_list.count() > 0)
            self.output_dir_btn.setStyleSheet("background-color: #4CAF50;")  # Green when set
        
    def analyze_mono_pairs(self):
        """Analyze files for mono pairs and update status"""
        processed = set()
        items_files = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item is None:
                continue
            file = next((f for f in self.dropped_files if os.path.basename(f) == item.text()), None)
            if not file:
                continue
            try:
                info = sf.info(file)
                if info.channels != 1:
                    self.set_item_status(item, "Not a mono file – will be ignored", True)
                    processed.add(file)
                    continue
            except Exception as e:
                self.set_item_status(item, f"Error: {str(e)}", True)
                processed.add(file)
                continue
            items_files.append((item, file))
        pairs = []
        for idx, (item, file) in enumerate(items_files):
            if file in processed:
                continue
            base_path = os.path.splitext(file)[0]
            base_name = base_path.replace('-L', '').replace('_L', '')\
                                .replace(' L ', ' ').replace('-R', '')\
                                .replace('_R', '').replace(' R ', ' ')\
                                .replace('Left', '').replace('Right', '')\
                                .replace('left', '').replace('right', '')
            for jdx, (other_item, other_file) in enumerate(items_files):
                if idx == jdx or other_file in processed:
                    continue
                other_base = os.path.splitext(other_file)[0]
                other_base = other_base.replace('-L', '').replace('_L', '')\
                                     .replace(' L ', ' ').replace('-R', '')\
                                     .replace('_R', '').replace(' R ', ' ')\
                                     .replace('Left', '').replace('Right', '')\
                                     .replace('left', '').replace('right', '')
                if base_name == other_base:
                    if ('L' in base_path.upper() and 'R' in os.path.splitext(other_file)[0].upper()):
                        self.set_item_status(item, "Left channel", False)
                        self.set_item_status(other_item, "Right channel", False)
                        processed.add(file)
                        processed.add(other_file)
                        pairs.append((file, other_file))
                    elif ('R' in base_path.upper() and 'L' in os.path.splitext(other_file)[0].upper()):
                        self.set_item_status(item, "Right channel", False)
                        self.set_item_status(other_item, "Left channel", False)
                        processed.add(file)
                        processed.add(other_file)
                        pairs.append((other_file, file))
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item is None:
                continue
            file = next((f for f in self.dropped_files if os.path.basename(f) == item.text()), None)
            if not file or file in processed:
                continue
            try:
                info = sf.info(file)
                if info.channels == 1:
                    self.set_item_status(item, "No matching pair found – will be ignored", True)
            except Exception as e:
                self.set_item_status(item, f"Error: {str(e)}", True)
        return len(pairs)

    def analyze_stereo_files(self):
        """Analyze files for stereo format and update status"""
        valid_count = 0
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item is None:
                continue
            file = next((f for f in self.dropped_files if os.path.basename(f) == item.text()), None)
            if not file:
                continue
            try:
                info = sf.info(file)
                if info.channels == 2:
                    self.set_item_status(item, "Stereo file", False)
                    valid_count += 1
                else:
                    self.set_item_status(item, "Not a stereo file – will be ignored", True)
            except Exception as e:
                self.set_item_status(item, f"Error: {str(e)}", True)
        return valid_count

    def handle_dropped_files(self, files):
        """Handle dropped files and update the list"""
        # Filter out duplicates
        new_files = [f for f in files if f not in self.dropped_files]
        if not new_files:
            return
        # Add new files to list widget
        for file in new_files:
            self.add_file_to_list(file)
        # Always re-analyze all files and update process button
        valid_count = self.analyze_dropped_files()
        self.update_process_button_state(valid_count)
        # Check if we should set output directory to input directory
        if not self.output_frame.dir_path.text() or self.output_frame.dir_path.text() == "Not set":
            self.set_output_to_input()

    def open_output_directory(self):
        """Open the output directory in system file explorer"""
        output_path = self.output_frame.dir_path.text()
        if output_path and output_path != "Not set":
            import subprocess
            import platform
            
            try:
                if platform.system() == "Windows":
                    os.startfile(output_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", output_path])
                else:  # Linux
                    subprocess.run(["xdg-open", output_path])
            except Exception as e:
                self.status_label.setText(f"Error opening directory: {str(e)}")

    def process_files(self):
        """Process the audio files"""
        # Reset failed files list
        self.failed_files = []
        # Get full paths from filenames
        files = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item is None:
                continue
            filename = item.text()
            for file in self.dropped_files:
                if os.path.basename(file) == filename:
                    files.append(file)
                    break
        # Extra safety: prevent processing if no valid pairs/files
        can_process = False
        if self.current_mode == "mono_to_stereo":
            can_process = self.analyze_mono_pairs() > 0
        else:
            can_process = self.analyze_stereo_files() > 0
        if not can_process:
            self.status_label.setText("No valid files to process.")
            return
        # Create and start processor thread
        self.processor = AudioProcessorThread(
            "Stereo to Mono" if self.mode_button.isChecked() else "Mono to Stereo",
            files,
            self.output_frame.dir_path.text(),
            self.output_frame.get_output_format()
        )
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.status_updated.connect(self.update_status)
        self.processor.processing_complete.connect(self.processing_finished)
        self.processor.failed_files_updated.connect(self.update_failed_files)
        # Update UI
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("background-color: #2196F3;")  # Reset to blue
        self.cancel_btn.setEnabled(True)
        self.current_progress.setValue(0)
        self.overall_progress.setValue(0)
        # Enable open output directory button
        self.open_output_btn.setEnabled(True)
        # Start processing
        self.processor.start()

    def update_progress(self, current_progress, overall_progress):
        """Update progress bars"""
        self.current_progress.setValue(current_progress)
        self.overall_progress.setValue(overall_progress)
        
    def update_status(self, message):
        """Update status message with larger text"""
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setStyleSheet("color: #888; padding: 10px;")
        self.status_label.setText(message)
        
    def processing_finished(self):
        """Handle processing completion"""
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.current_progress.setValue(100)
        self.overall_progress.setValue(100)
        self.process_btn.setStyleSheet("background-color: #2196F3;")  # Reset to blue
        
    def cancel_processing(self):
        """Cancel the processing"""
        if hasattr(self, 'processor'):
            self.processor.is_cancelled = True
            self.cancel_btn.setEnabled(False)
            self.status_label.setText("Cancelling...")
            
    def select_output_directory(self):
        """Open directory selection dialog"""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_frame.set_directory(directory)
            self.process_btn.setEnabled(self.file_list.count() > 0)

    def handle_error(self, error_message):
        """Handle processing error"""
        self.process_btn.setStyleSheet("background-color: #f44336;")  # Red for error
        self.update_status(f"Error: {error_message}")

    def handle_success(self):
        """Handle processing success"""
        self.process_btn.setStyleSheet("background-color: #4CAF50;")  # Green for success

    def update_failed_files(self, failed_files):
        """Update the list of failed files and enable the show failed files button"""
        self.failed_files = failed_files
        self.process_btn.setEnabled(bool(failed_files))
        if failed_files:
            self.process_btn.setStyleSheet("background-color: #f44336;")  # Red for failed files

    def clear_selected_files(self):
        """Remove selected files from the list"""
        selected_items = self.file_list.selectedItems()
        for item in selected_items:
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
        # Update dropped_files to match the file list, skip None items
        new_dropped_files = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item is not None:
                new_dropped_files.append(item.data(Qt.ItemDataRole.UserRole))
        self.dropped_files = new_dropped_files
        self.update_process_button_state()

    def add_file_to_list(self, filepath):
        """Add a file to the list widget"""
        filename = os.path.basename(filepath)
        item = QListWidgetItem()
        
        # Store full path
        item.setData(Qt.ItemDataRole.UserRole, filepath)
        item.setText(filename)
        
        item.setSizeHint(QSize(0, 30))
        self.file_list.addItem(item)
        self.dropped_files.append(filepath)
        
        # Analyze the file immediately
        try:
            with sf.SoundFile(filepath) as audio_file:
                if audio_file.channels == 2:
                    self.set_item_status(item, "Stereo file", False)
                else:
                    self.set_item_status(item, "Not a stereo file – will be ignored", True)
        except Exception as e:
            self.set_item_status(item, f"Error: {str(e)}", True)
        
        return item

    def set_item_status(self, item, status, is_error=False):
        """Set the status for a list item"""
        filepath = item.data(Qt.ItemDataRole.UserRole)
        filename = os.path.basename(filepath)
        
        # Store filename and status separately using different display roles
        item.setText(filename)  # Main text is filename
        item.setData(Qt.ItemDataRole.UserRole + 2, status)  # Store status
        
        # Store whether this is a mono->stereo mode status
        item.setData(Qt.ItemDataRole.UserRole + 3, self.current_mode == "mono_to_stereo")
        
        # Set text alignment for the status (right-aligned)
        item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # Store status for process button state checking
        item.setData(Qt.ItemDataRole.UserRole + 1, status)

    def clear_invalid_files(self):
        """Remove all files from the list that are marked as invalid/ignored"""
        # Collect items to remove
        items_to_remove = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item is None:
                continue
            status = item.data(Qt.ItemDataRole.UserRole + 2)
            if status and ("will be ignored" in status or "Error" in status):
                items_to_remove.append(item)
        # Remove them
        for item in items_to_remove:
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
        # Update dropped_files to match
        self.dropped_files = [item.data(Qt.ItemDataRole.UserRole) for i in range(self.file_list.count()) if (item := self.file_list.item(i))]
        self.update_process_button_state()

    def toggle_mode(self):
        """Toggle between stereo->mono and mono->stereo modes"""
        if self.mode_button.text() == "STEREO → 2 MONO":
            self.mode_button.setText("MONO → STEREO")
            self.mode_button.setStyleSheet(self.get_toggle_style("Mono to Stereo"))
            self.current_mode = "mono_to_stereo"
            self.drop_area.text_label.setText("Drop mono audio files here (L/R pairs)")
            self.mode_frame.setStyleSheet(self.get_mode_frame_style("Stereo to Mono"))
            action_style = self.get_action_button_style("Mono to Stereo")
        else:
            self.mode_button.setText("STEREO → 2 MONO")
            self.mode_button.setStyleSheet(self.get_toggle_style("Stereo to Mono"))
            self.current_mode = "stereo_to_mono"
            self.drop_area.text_label.setText("Drop stereo audio files here")
            self.mode_frame.setStyleSheet(self.get_mode_frame_style("Mono to Stereo"))
            action_style = self.get_action_button_style("Stereo to Mono")
        self.clear_invalid_btn.setStyleSheet(action_style)
        self.clear_selected_btn.setStyleSheet(action_style)
        self.clear_all_btn.setStyleSheet(action_style)
        self.output_frame.same_as_input.setStyleSheet(action_style)
        valid_count = self.analyze_dropped_files()
        self.update_process_button_state(valid_count)

    def analyze_dropped_files(self):
        """Analyze dropped files and update their status in the list. Returns valid count/pairs."""
        if self.current_mode == "mono_to_stereo":
            return self.analyze_mono_pairs()
        else:
            return self.analyze_stereo_files()

    def get_action_button_style(self, mode):
        if mode == "Mono to Stereo":
            # Orange style
            return """
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #F57C00;
                }
            """
        else:
            # Blue style
            return """
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioConverterApp()
    window.show()
    sys.exit(app.exec()) 