"""PyQt6 GUI Main Window for Eco Video Upscaler."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QProgressBar,
    QTextEdit,
    QFileDialog,
    QMessageBox,
)

from ecoup.cli import check_gpu, _ensure_path_refreshed
from ecoup.config import (
    AppConfig,
    ModelConfig,
    ThermalConfig,
    FrameSavingConfig,
    CheckpointConfig,
    EncodeConfig,
    InferenceConfig,
    PreProcessConfig,
    PostProcessConfig,
)
from ecoup.engine.onnx_engine import OnnxUpscaleEngine
from ecoup.pipeline import UpscalePipeline
from ecoup.video_io import get_video_metadata, VideoMetadata


DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #18181b;
    color: #e4e4e7;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2e2e33;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    background-color: #202023;
}
QLineEdit, QComboBox, QSpinBox {
    background-color: #18181b;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 5px 8px;
    color: #f4f4f5;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border-color: #3b82f6;
}
QPushButton {
    background-color: #2e2e33;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
    color: #f4f4f5;
}
QPushButton:hover {
    background-color: #3f3f46;
}
QPushButton#primaryBtn {
    background-color: #2563eb;
    border: 1px solid #1d4ed8;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primaryBtn:hover {
    background-color: #1d4ed8;
}
QPushButton#dangerBtn {
    color: #ef4444;
}
QPushButton#dangerBtn:hover {
    background-color: #451a1a;
    border-color: #ef4444;
}
QProgressBar {
    background-color: #18181b;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    height: 12px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 3px;
}
QTextEdit {
    background-color: #121214;
    border: 1px solid #27272a;
    border-radius: 4px;
    color: #a1a1aa;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}
"""


class PipelineWorker(QThread):
    """Background worker thread for running the upscale pipeline."""
    progress_sig = pyqtSignal(int, int, float, float, object, float)
    log_sig = pyqtSignal(str)
    finished_sig = pyqtSignal(bool, str)

    def __init__(self, config: AppConfig, input_path: str, output_path: str, resume: bool = False):
        super().__init__()
        self.config = config
        self.input_path = input_path
        self.output_path = output_path
        self.resume = resume
        self.pipeline: Optional[UpscalePipeline] = None
        self.engine: Optional[OnnxUpscaleEngine] = None

    def run(self):
        try:
            model_path = Path(self.config.model.path)
            custom_fn = None
            if model_path.exists():
                self.log_sig.emit(f"Đang nạp mô hình AI: {model_path.name}...")
                self.engine = OnnxUpscaleEngine()
                self.engine.load(model_path, precision=self.config.model.precision, provider=self.config.inference.provider)
                if hasattr(self.engine, "scale") and self.engine.scale > 0:
                    self.config.model.scale = self.engine.scale
                    self.log_sig.emit(f"Mô hình đã sẵn sàng (Native Scale: {self.engine.scale}x).")
            else:
                self.log_sig.emit(f"[Thông báo] Không tìm thấy file model AI '{model_path.name}'. Đang chạy chế độ nội suy Lanczos4 chất lượng cao...")
                import cv2
                scale = self.config.model.scale
                def lanczos_fallback(tile: np.ndarray) -> np.ndarray:
                    th, tw = tile.shape[:2]
                    return cv2.resize(tile, (tw * scale, th * scale), interpolation=cv2.INTER_LANCZOS4)
                custom_fn = lanczos_fallback

            def on_progress(frame_idx, total_frames, fps, eta_sec, metrics, dedup_rate):
                self.progress_sig.emit(frame_idx, total_frames, fps, eta_sec, metrics, dedup_rate)

            self.pipeline = UpscalePipeline(
                config=self.config,
                engine=self.engine,
                custom_upscale_fn=custom_fn,
                on_progress=on_progress,
            )

            self.log_sig.emit(f"Bắt đầu xử lý: {Path(self.input_path).name} -> {Path(self.output_path).name}")
            self.pipeline.run(self.input_path, self.output_path, resume=self.resume)
            self.finished_sig.emit(True, "Hoàn tất upscale video thành công!")
        except Exception as e:
            self.finished_sig.emit(False, str(e))
        finally:
            if self.engine:
                self.engine.close()

    def cancel(self):
        if self.pipeline:
            self.pipeline.cancel()


class MainWindow(QMainWindow):
    """Main window for Eco Video Upscaler."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eco Video Upscaler")
        self.setMinimumSize(780, 680)
        self.setStyleSheet(DARK_STYLE)

        self.worker: Optional[PipelineWorker] = None
        self.is_paused = False
        self.meta: Optional[VideoMetadata] = None

        self._build_ui()
        self._init_hardware_info()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(12)
        root_layout.setContentsMargins(18, 16, 18, 16)

        # 1. Header Row
        header_layout = QHBoxLayout()
        title_label = QLabel("Eco Video Upscaler")
        title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.status_badge = QLabel("Sẵn sàng")
        self.status_badge.setStyleSheet("color: #10b981; font-weight: 500;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_badge)
        root_layout.addLayout(header_layout)

        # 2. File Selection Panel
        file_box = QWidget()
        file_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px; padding: 6px;")
        file_layout = QGridLayout(file_box)
        file_layout.setContentsMargins(10, 10, 10, 10)
        file_layout.setSpacing(8)

        file_layout.addWidget(QLabel("Video vào:"), 0, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Chọn video nguồn (.mp4, .mkv, .mov)...")
        file_layout.addWidget(self.input_edit, 0, 1)
        browse_in_btn = QPushButton("Chọn...")
        browse_in_btn.clicked.connect(self._browse_input)
        file_layout.addWidget(browse_in_btn, 0, 2)

        file_layout.addWidget(QLabel("Video ra:"), 1, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Đường dẫn lưu video sau khi upscale...")
        file_layout.addWidget(self.output_edit, 1, 1)
        browse_out_btn = QPushButton("Lưu tại...")
        browse_out_btn.clicked.connect(self._browse_output)
        file_layout.addWidget(browse_out_btn, 1, 2)

        self.meta_label = QLabel("Chưa chọn video")
        self.meta_label.setStyleSheet("color: #71717a; font-family: monospace; font-size: 11px;")
        file_layout.addWidget(self.meta_label, 2, 1, 1, 2)

        root_layout.addWidget(file_box)

        # 3. Settings Panel
        settings_box = QWidget()
        settings_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px; padding: 6px;")
        set_layout = QVBoxLayout(settings_box)
        set_layout.setContentsMargins(10, 10, 10, 10)
        set_layout.setSpacing(10)

        # Dropdowns row
        dropdown_row = QHBoxLayout()
        dropdown_row.setSpacing(12)

        # Model
        m_col = QVBoxLayout()
        m_col.addWidget(QLabel("Mô hình AI:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "NomosUni 2x (Siêu nét, video đời thực - Nhẹ 2.3MB)",
            "SPAN 2x (Siêu nhanh, mát GPU - Nhẹ 1.6MB)",
            "realesr-general-x4v3 (4x, Khử nhiễu mịn da - 4.9MB)",
            "realesr-animevideov3 (4x, Hoạt hình / 2D - 2.5MB)",
            "RealESRGAN x2plus (2x, Chi tiết cao cấp - Nặng 67MB)",
        ])
        m_col.addWidget(self.model_combo)
        dropdown_row.addLayout(m_col)

        # Scale
        s_col = QVBoxLayout()
        s_col.addWidget(QLabel("Độ phân giải đích:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems([
            "1080p (Scale 2x)",
            "1440p (Scale 2x)",
            "4K (Scale 4x)",
        ])
        s_col.addWidget(self.scale_combo)
        dropdown_row.addLayout(s_col)

        # Encoder
        e_col = QVBoxLayout()
        e_col.addWidget(QLabel("Bộ mã hóa (Encoder):"))
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems([
            "NVIDIA NVENC H.264 (Phần cứng)",
            "NVIDIA NVENC HEVC / H.265",
            "x264 (CPU)",
        ])
        e_col.addWidget(self.encoder_combo)
        dropdown_row.addLayout(e_col)

        # Tile size
        t_col = QVBoxLayout()
        t_col.addWidget(QLabel("Chia Tile (Tốc độ):"))
        self.tile_combo = QComboBox()
        self.tile_combo.addItems([
            "256x256 (Mát máy / Tiết kiệm VRAM)",
            "512x512 (Cân bằng)",
            "Toàn khung (Nhanh nhất)",
        ])
        t_col.addWidget(self.tile_combo)
        dropdown_row.addLayout(t_col)

        set_layout.addLayout(dropdown_row)

        # Checkboxes row
        chk_row = QHBoxLayout()
        self.thermal_chk = QCheckBox("Kiểm soát nhiệt độ: Giữ dưới")
        self.thermal_chk.setChecked(True)
        self.temp_spin = QSpinBox()
        self.temp_spin.setRange(50, 90)
        self.temp_spin.setValue(70)
        self.temp_spin.setSuffix(" °C")

        self.dedup_chk = QCheckBox("Bỏ qua khung hình trùng (Dedup)")
        self.dedup_chk.setChecked(True)

        self.checkpoint_chk = QCheckBox("Lưu checkpoint định kỳ (Resume)")
        self.checkpoint_chk.setChecked(True)

        chk_row.addWidget(self.thermal_chk)
        chk_row.addWidget(self.temp_spin)
        chk_row.addSpacing(16)
        chk_row.addWidget(self.dedup_chk)
        chk_row.addSpacing(16)
        chk_row.addWidget(self.checkpoint_chk)
        chk_row.addStretch()

        set_layout.addLayout(chk_row)

        # Realism & Artifact Control Row
        realism_row = QHBoxLayout()
        realism_row.setSpacing(14)

        # Pre-Denoise
        d_box = QHBoxLayout()
        d_box.addWidget(QLabel("Khử nhiễu gốc (Pre-Denoise):"))
        self.pre_denoise_spin = QSpinBox()
        self.pre_denoise_spin.setRange(0, 100)
        self.pre_denoise_spin.setValue(0)
        self.pre_denoise_spin.setSuffix(" %")
        self.pre_denoise_spin.setToolTip("Lọc sạch hạt nhiễu nén trước khi AI xử lý, ngăn chặn AI khuếch đại hạt rác và vẽ bậy")
        d_box.addWidget(self.pre_denoise_spin)
        realism_row.addLayout(d_box)

        # Recover Original Detail
        r_box = QHBoxLayout()
        r_box.addWidget(QLabel("Khôi phục gốc (Recover):"))
        self.recover_spin = QSpinBox()
        self.recover_spin.setRange(0, 50)
        self.recover_spin.setValue(0)
        self.recover_spin.setSuffix(" %")
        self.recover_spin.setToolTip("Trộn texture video gốc để xóa bỏ hiện tượng AI vẽ bậy hoa văn lạ hoặc bệt môi/da tượng sáp")
        r_box.addWidget(self.recover_spin)
        realism_row.addLayout(r_box)

        # Sharpening
        s_box = QHBoxLayout()
        s_box.addWidget(QLabel("Làm nét viền (Sharpen):"))
        self.sharpen_spin = QSpinBox()
        self.sharpen_spin.setRange(0, 100)
        self.sharpen_spin.setValue(25)
        self.sharpen_spin.setSuffix(" %")
        self.sharpen_spin.setToolTip("Tăng độ tương phản cạnh giúp viền rõ ràng, khử mờ nhòe")
        s_box.addWidget(self.sharpen_spin)
        realism_row.addLayout(s_box)

        # Film Grain
        g_box = QHBoxLayout()
        self.grain_chk = QCheckBox("Bù hạt tự nhiên (Film Grain)")
        self.grain_chk.setChecked(False)
        self.grain_chk.setToolTip("Phủ lớp hạt vi mô tự nhiên giúp da và môi sống động như quay máy ảnh thật, khử cảm giác búp bê nhựa")
        g_box.addWidget(self.grain_chk)
        realism_row.addLayout(g_box)

        realism_row.addStretch()
        set_layout.addLayout(realism_row)

        root_layout.addWidget(settings_box)

        # 4. Hardware Monitor Box
        hw_box = QWidget()
        hw_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px;")
        hw_layout = QGridLayout(hw_box)
        hw_layout.setContentsMargins(12, 10, 12, 10)

        # Col 0: Temp
        hw_layout.addWidget(QLabel("Nhiệt độ GPU"), 0, 0, Qt.AlignmentFlag.AlignCenter)
        self.val_temp = QLabel("-- °C")
        self.val_temp.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        self.val_temp.setStyleSheet("color: #10b981;")
        hw_layout.addWidget(self.val_temp, 1, 0, Qt.AlignmentFlag.AlignCenter)

        # Col 1: Power
        hw_layout.addWidget(QLabel("Công suất TDP"), 0, 1, Qt.AlignmentFlag.AlignCenter)
        self.val_power = QLabel("-- W")
        self.val_power.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        self.val_power.setStyleSheet("color: #3b82f6;")
        hw_layout.addWidget(self.val_power, 1, 1, Qt.AlignmentFlag.AlignCenter)

        # Col 2: VRAM
        hw_layout.addWidget(QLabel("VRAM sử dụng"), 0, 2, Qt.AlignmentFlag.AlignCenter)
        self.val_vram = QLabel("-- / 4.0 GB")
        self.val_vram.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        self.val_vram.setStyleSheet("color: #e4e4e7;")
        hw_layout.addWidget(self.val_vram, 1, 2, Qt.AlignmentFlag.AlignCenter)

        # Col 3: Dedup
        hw_layout.addWidget(QLabel("Khung trùng bỏ qua"), 0, 3, Qt.AlignmentFlag.AlignCenter)
        self.val_dedup = QLabel("0 (0.0%)")
        self.val_dedup.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        self.val_dedup.setStyleSheet("color: #a855f7;")
        hw_layout.addWidget(self.val_dedup, 1, 3, Qt.AlignmentFlag.AlignCenter)

        root_layout.addWidget(hw_box)

        # 5. Progress & Action Buttons
        prog_box = QWidget()
        prog_box.setStyleSheet("background-color: #202023; border: 1px solid #2e2e33; border-radius: 6px;")
        prog_layout = QVBoxLayout(prog_box)
        prog_layout.setContentsMargins(12, 12, 12, 12)
        prog_layout.setSpacing(8)

        # Stats line
        stats_line = QHBoxLayout()
        self.prog_text = QLabel("Tiến độ: 0% (0 / 0 khung)")
        self.prog_speed = QLabel("Tốc độ: -- fps • Còn lại: --:--")
        self.prog_speed.setStyleSheet("color: #71717a; font-family: monospace;")
        stats_line.addWidget(self.prog_text)
        stats_line.addStretch()
        stats_line.addWidget(self.prog_speed)
        prog_layout.addLayout(stats_line)

        # Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        prog_layout.addWidget(self.progress_bar)

        # Buttons
        btn_line = QHBoxLayout()
        self.start_btn = QPushButton("Bắt đầu")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.clicked.connect(self._start_pipeline)
        btn_line.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Tạm dừng")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._toggle_pause)
        btn_line.addWidget(self.pause_btn)

        self.cancel_btn = QPushButton("Hủy")
        self.cancel_btn.setObjectName("dangerBtn")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_pipeline)
        btn_line.addWidget(self.cancel_btn)

        btn_line.addStretch()

        open_folder_btn = QPushButton("Mở thư mục")
        open_folder_btn.clicked.connect(self._open_output_folder)
        btn_line.addWidget(open_folder_btn)

        prog_layout.addLayout(btn_line)
        root_layout.addWidget(prog_box)

        # 6. Real-time Console Log Box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(110)
        self.log_box.append("[Hệ thống] Giao diện Eco Video Upscaler đã sẵn sàng.")
        root_layout.addWidget(self.log_box)

    def _init_hardware_info(self):
        """Read initial hardware telemetry."""
        gpu = check_gpu()
        if gpu["available"]:
            self.val_temp.setText(f"{gpu['temp_c']} °C")
            if gpu["power_w"] is not None:
                self.val_power.setText(f"{gpu['power_w']} W")
            total_gb = round(gpu["vram_total_mb"] / 1024, 1)
            self.val_vram.setText(f"0.0 / {total_gb} GB")
            self.log_box.append(f"[Phần cứng] Phát hiện {gpu['name']} ({total_gb} GB VRAM).")

    def _browse_input(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Chọn Video Nguồn", "", "Video Files (*.mp4 *.mkv *.avi *.mov *.webm)"
        )
        if file:
            self.input_edit.setText(file)
            try:
                self.meta = get_video_metadata(file)
                self.meta_label.setText(
                    f"Gốc: {self.meta.width}x{self.meta.height} • {self.meta.fps:.2f} fps • Audio: {'Có' if self.meta.has_audio else 'Không'} • {self.meta.total_frames} khung"
                )
                # Auto set output path
                p = Path(file)
                default_out = str(p.parent / f"{p.stem}_upscaled{p.suffix}")
                self.output_edit.setText(default_out)
            except Exception as e:
                self.meta_label.setText(f"Lỗi đọc metadata: {e}")

    def _browse_output(self):
        file, _ = QFileDialog.getSaveFileName(
            self, "Lưu Video Đầu Ra", self.output_edit.text() or "", "MP4 Video (*.mp4)"
        )
        if file:
            self.output_edit.setText(file)

    def _start_pipeline(self):
        in_file = self.input_edit.text().strip()
        out_file = self.output_edit.text().strip()

        if not in_file or not os.path.exists(in_file):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn video đầu vào hợp lệ.")
            return

        model_text = self.model_combo.currentText().lower()
        if "nomosuni" in model_text:
            model_name = "2xNomosUni_compact_otf_medium"
            model_file = "models/2xNomosUni_compact_otf_medium.onnx"
            model_scale = 2
        elif "span" in model_text:
            model_name = "2x-spanx2-ch48"
            model_file = "models/2x-spanx2-ch48.onnx"
            model_scale = 2
        elif "x2plus" in model_text:
            model_name = "2x-realesrgan-x2plus"
            model_file = "models/2x-realesrgan-x2plus.onnx"
            model_scale = 2
        elif "anime" in model_text:
            model_name = "realesr-animevideov3"
            model_file = "models/realesr-animevideov3.onnx"
            model_scale = 4
        else:
            model_name = "realesr-general-x4v3"
            model_file = "models/realesr-general-x4v3.onnx"
            model_scale = 4

        target_scale = 4 if "4x" in self.scale_combo.currentText() else 2

        tile_text = self.tile_combo.currentText()
        if "Toàn khung" in tile_text:
            tile_sz = 0
            tile_ov = 0
        elif "512" in tile_text:
            tile_sz = 512
            tile_ov = 16
        else:
            tile_sz = 256
            tile_ov = 16

        cfg = AppConfig(
            model=ModelConfig(
                name=model_name,
                path=model_file,
                scale=model_scale,
                target_scale=target_scale,
            ),
            inference=InferenceConfig(
                provider="auto",
                tile_size=tile_sz,
                tile_overlap=tile_ov,
            ),
            thermal=ThermalConfig(
                enabled=self.thermal_chk.isChecked(),
                temp_soft_c=self.temp_spin.value(),
                temp_hard_c=self.temp_spin.value() + 10,
                temp_resume_c=self.temp_spin.value() - 5,
            ),
            frame_saving=FrameSavingConfig(
                enable_dedup=self.dedup_chk.isChecked(),
            ),
            checkpoint=CheckpointConfig(
                enabled=self.checkpoint_chk.isChecked(),
                interval_frames=300,
            ),
            encode=EncodeConfig(
                codec="h264_nvenc" if "NVENC H.264" in self.encoder_combo.currentText() else "libx264",
                copy_audio=True,
            ),
            pre_process=PreProcessConfig(
                denoise=self.pre_denoise_spin.value() / 100.0,
            ),
            post_process=PostProcessConfig(
                recover_detail=self.recover_spin.value() / 100.0,
                sharpen=self.sharpen_spin.value() / 100.0,
                film_grain=0.015 if self.grain_chk.isChecked() else 0.0,
            ),
        )

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.status_badge.setText("Đang chạy")
        self.status_badge.setStyleSheet("color: #3b82f6; font-weight: 500;")

        self.worker = PipelineWorker(config=cfg, input_path=in_file, output_path=out_file)
        self.worker.progress_sig.connect(self._on_worker_progress)
        self.worker.log_sig.connect(self.log_box.append)
        self.worker.finished_sig.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_progress(self, frame_idx: int, total_frames: int, fps: float, eta_sec: float, metrics, dedup_rate: float):
        pct = int((frame_idx / max(1, total_frames)) * 100)
        self.progress_bar.setValue(pct)
        self.prog_text.setText(f"Tiến độ: {pct}% ({frame_idx} / {total_frames} khung)")

        mins = int(eta_sec // 60)
        secs = int(eta_sec % 60)
        self.prog_speed.setText(f"Tốc độ: {fps:.1f} fps • Còn lại: {mins:02d}:{secs:02d}")

        if metrics:
            self.val_temp.setText(f"{metrics.temp_c} °C")
            if metrics.temp_c >= self.temp_spin.value():
                self.val_temp.setStyleSheet("color: #f59e0b;")
            else:
                self.val_temp.setStyleSheet("color: #10b981;")
            if metrics.power_w:
                self.val_power.setText(f"{round(metrics.power_w, 1)} W")
            used_gb = round(metrics.vram_used_mb / 1024, 1)
            total_gb = round(metrics.vram_total_mb / 1024, 1)
            self.val_vram.setText(f"{used_gb} / {total_gb} GB")

        self.val_dedup.setText(f"{dedup_rate:.1f}%")

    def _on_worker_finished(self, success: bool, message: str):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.status_badge.setText("Sẵn sàng")
        self.status_badge.setStyleSheet("color: #10b981; font-weight: 500;")

        if success:
            self.log_box.append(f"[Xong] {message}")
            QMessageBox.information(self, "Thành công", message)
        else:
            self.log_box.append(f"[Lỗi] {message}")
            QMessageBox.critical(self, "Lỗi xử lý", message)

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.worker and self.worker.pipeline:
            self.worker.pipeline.set_paused(self.is_paused)

        if self.is_paused:
            self.pause_btn.setText("Tiếp tục")
            self.status_badge.setText("Tạm dừng")
            self.status_badge.setStyleSheet("color: #f59e0b;")
            self.log_box.append("[Hệ thống] Đã tạm dừng xử lý (GPU đang nghỉ).")
        else:
            self.pause_btn.setText("Tạm dừng")
            self.status_badge.setText("Đang chạy")
            self.status_badge.setStyleSheet("color: #3b82f6;")
            self.log_box.append("[Hệ thống] Tiếp tục xử lý...")

    def _cancel_pipeline(self):
        if self.worker:
            self.worker.cancel()
            self.log_box.append("[Hệ thống] Đang yêu cầu hủy xử lý...")

    def _open_output_folder(self):
        out_file = self.output_edit.text().strip()
        folder = Path(out_file).parent if out_file else Path.cwd()
        if folder.exists():
            os.startfile(folder)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
