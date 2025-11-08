import os
import cv2
import sys
import threading
import tkinter as tk
from queue import Queue, Empty, Full
from threading import Thread, Event
from pathlib import Path
from tkinter import Tk, StringVar, Label, Button, messagebox, PhotoImage
from tkinter import ttk
from PIL import Image, ImageTk
import yaml
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, Tuple


# ============================================================================
# Configuration Management
# ============================================================================

def get_application_path() -> Path:
    """Get application directory (works for both script and compiled exe)."""
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        return Path(sys.executable).parent
    else:
        # Running as script
        return Path(__file__).parent


APP_DIR = get_application_path()
CONFIG_PATH = APP_DIR / "config.yaml"
LOG_DIR = APP_DIR / "logs"

# Default configuration
DEFAULT_CONFIG = {
    "paths": {
        "capture_dir": str(APP_DIR / "captures"),
        "reference_dir": str(APP_DIR / "references"),
        "database": str(APP_DIR / "database.db")
    },
    "camera": {
        "resolution": [1920, 1080],
        "fps": 30,
        "buffer_size": 1
    },
    "analysis": {
        "confidence_threshold": 0.75
    },
    "ui": {
        "preview_size": [960, 720],
        "window_size": [1200, 850],
        "fullscreen_preview": False,
        "result_image_size": [450, 340]
    }
}


def load_config() -> Dict[str, Any]:
    """Load configuration with fallback to defaults."""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
            # Merge with defaults (deep merge for nested dicts)
            config = DEFAULT_CONFIG.copy()
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
            logger.info("Configuration loaded successfully")
            return config
        else:
            # Create default config file
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
            logger.info("Default configuration created")
            return DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.error(f"Error loading config: {e}", exc_info=True)
        return DEFAULT_CONFIG.copy()


def setup_logging():
    """Setup logging system with rotation."""
    LOG_DIR.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


# Setup logging first
logger = setup_logging()

# Load configuration
config = load_config()

# Setup directories
CAPTURE_DIR = Path(config["paths"]["capture_dir"])
REF_DIR = Path(config["paths"]["reference_dir"])
DB_PATH = Path(config["paths"]["database"])
CAPTURE_PATH = CAPTURE_DIR / "capture.jpg"

# Create directories if not exist
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Application initialized")


# ============================================================================
# Import Core Modules (with error handling)
# ============================================================================

try:
    from core.camera import capture_sample
    from core.preprocessing import preprocess_image
    from core.features import extract_features
    from core.features import find_best_match
    from core.analysis import get_diagnosis_by_image
    logger.info("Core modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    messagebox.showerror(
        "Import Error",
        f"Gagal import modul core: {e} Pastikan semua file di folder 'core/' tersedia."
    )


# ============================================================================
# Base Page Class
# ============================================================================

class BasePage(ttk.Frame):
    """Base class untuk semua halaman aplikasi."""
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        logger.debug(f"Initialized page: {self.__class__.__name__}")
    
    def on_show(self):
        """Called when page is shown (override in subclasses if needed)."""
        pass
    
    def on_hide(self):
        """Called when page is hidden (override in subclasses if needed)."""
        pass


# ============================================================================
# Page 1: Home Page
# ============================================================================

class HomePage(BasePage):
    """Halaman utama aplikasi."""
    
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self._build_ui()
    
    def _build_ui(self):
        """Build UI components."""
        # Title
        title_frame = ttk.Frame(self)
        title_frame.pack(pady=30)
        
        ttk.Label(
            title_frame,
            text="🩸 Digital Blood Analyzer",
            font=("Arial", 20, "bold")
        ).pack()
        
        ttk.Label(
            title_frame,
            text="Aplikasi analisis darah berbasis gambar mikroskop",
            font=("Arial", 12)
        ).pack(pady=10)
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=20)
        
        ttk.Button(
            btn_frame,
            text="▶ Mulai Analisis",
            command=self._start_analysis,
            width=25
        ).pack(pady=10)
        
        ttk.Button(
            btn_frame,
            text="📁 Jelajahi Database",
            command=lambda: self.controller.show_page(DatabasePage),
            width=25
        ).pack(pady=5)
        
        ttk.Button(
            btn_frame,
            text="❌ Keluar",
            command=self._quit_app,
            width=25
        ).pack(pady=5)
        
        # Version info
        ttk.Label(
            self,
            text="Version 1.1 - Improved",
            font=("Arial", 8),
            foreground="gray"
        ).pack(side="bottom", pady=10)
    
    def _start_analysis(self):
        """Start analysis workflow."""
        logger.info("User started analysis workflow")
        self.controller.show_page(CameraSelectionPage)
    
    def _quit_app(self):
        """Quit application with confirmation."""
        if messagebox.askyesno("Konfirmasi", "Yakin ingin keluar dari aplikasi?"):
            logger.info("Application closed by user")
            self.controller.quit_app()


# ============================================================================
# Page 2: Camera Selection Page
# ============================================================================

class CameraSelectionPage(BasePage):
    """Halaman pemilihan kamera."""
    
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.device_var = tk.StringVar()
        self.devices_mapping = {}
        self._build_ui()
    
    def _build_ui(self):
        """Build UI components."""
        ttk.Label(
            self,
            text="📹 Pilih Perangkat Kamera",
            font=("Arial", 16, "bold")
        ).pack(pady=20)
        
        # Dropdown
        self.device_dropdown = ttk.Combobox(
            self,
            textvariable=self.device_var,
            state="readonly",
            width=60
        )
        self.device_dropdown.pack(pady=10)
        
        # Status label
        self.status_label = ttk.Label(
            self,
            text="Memuat daftar perangkat...",
            font=("Arial", 10)
        )
        self.status_label.pack(pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=15)
        
        ttk.Button(
            btn_frame,
            text="🔄 Refresh",
            command=self.load_devices
        ).grid(row=0, column=0, padx=5)
        
        ttk.Button(
            btn_frame,
            text="✅ Gunakan Terpilih",
            command=self.use_selected
        ).grid(row=0, column=1, padx=5)
        
        ttk.Button(
            btn_frame,
            text="⬅️ Kembali",
            command=lambda: self.controller.show_page(HomePage)
        ).grid(row=0, column=2, padx=5)
    
    def on_show(self):
        """Load devices when page is shown."""
        self.load_devices()
    
    def load_devices(self):
        """Load available camera devices."""
        logger.info("Loading camera devices...")
        self.status_label.config(text="🔍 Mencari kamera...", foreground="blue")
        
        try:
            from core.camera import list_available_camera_devices
            self.devices_mapping = list_available_camera_devices()
            
            if self.devices_mapping:
                device_names = list(self.devices_mapping.keys())
                self.device_dropdown["values"] = device_names
                
                # Select previously used or first device
                current = self.device_var.get()
                if current in device_names:
                    self.device_dropdown.set(current)
                else:
                    # Try to auto-select microscope
                    microscope = self._find_microscope()
                    if microscope:
                        self.device_dropdown.set(microscope)
                        self.status_label.config(
                            text=f"✅ Mikroskop terdeteksi: {microscope}",
                            foreground="green"
                        )
                    else:
                        self.device_dropdown.current(0)
                        self.status_label.config(
                            text=f"✅ {len(device_names)} kamera ditemukan",
                            foreground="green"
                        )
                
                logger.info(f"Found {len(device_names)} cameras: {device_names}")
            else:
                self.device_dropdown["values"] = ["Tidak ada kamera ditemukan"]
                self.status_label.config(
                    text="❌ Tidak ada kamera aktif",
                    foreground="red"
                )
                logger.warning("No cameras found")
                
        except Exception as e:
            error_msg = f"⚠️ Gagal memuat perangkat: {str(e)}"
            self.status_label.config(text=error_msg, foreground="red")
            logger.error(f"Error loading devices: {e}", exc_info=True)
            messagebox.showerror("Error", f"Gagal memuat daftar kamera:{e}")
    
    def _find_microscope(self) -> Optional[str]:
        """Find microscope camera by name keywords."""
        microscope_keywords = ['usb', 'microscope', 'mikroskop', 'external', 'pc cam']
        laptop_keywords = ['integrated', 'built-in', 'laptop', 'webcam', 'facecam']
        
        for name in self.devices_mapping.keys():
            name_lower = name.lower()
            
            # Check if contains microscope keyword
            has_microscope_keyword = any(kw in name_lower for kw in microscope_keywords)
            has_laptop_keyword = any(kw in name_lower for kw in laptop_keywords)
            
            if has_microscope_keyword and not has_laptop_keyword:
                logger.info(f"Auto-detected microscope: {name}")
            return name
        
        return None
    
    def use_selected(self):
        """Use selected camera device."""
        selected = self.device_var.get()
        
        if not selected or "Tidak ada" in selected:
            messagebox.showwarning("Peringatan", "Pilih perangkat kamera yang valid!")
            return
        
        logger.info(f"User selected camera: {selected}")

        self.controller.selected_device_name = selected  
        self.controller.selected_device_index = self.devices_mapping.get(selected)  
        self.controller.show_page(CapturePage)  


# ============================================================================
# Page 3: Capture Page (Improved)
# ============================================================================

class CapturePage(BasePage):
    """Halaman capture gambar dengan live preview."""
    
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        
        # Thread management
        self.running = False
        self.stop_event = Event()
        self.camera_thread: Optional[Thread] = None
        self.frame_lock = threading.Lock()
        
        # Camera resources
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue = Queue(maxsize=2)
        self.current_frame = None
        
        # UI state
        self.is_capturing = False
        self.is_fullscreen = False
        
        self._build_ui()
    
    def _build_ui(self):
        """Build UI components."""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(pady=10, fill="x")
        
        ttk.Label(
            header_frame,
            text="📷 Live Preview Mikroskop",
            font=("Arial", 16, "bold")
        ).pack()
        
        # Device info label
        self.device_info_label = ttk.Label(
            header_frame,
            text="",
            font=("Arial", 9),
            foreground="gray"
        )
        self.device_info_label.pack()
        
        # Preview area dengan frame
        preview_container = ttk.Frame(self, relief="solid", borderwidth=2)
        preview_container.pack(pady=10, padx=10, expand=True, fill="both")
        
        preview_width, preview_height = config["ui"]["preview_size"]
        
        self.preview_label = Label(
            preview_container,
            text="⏳ Mengaktifkan kamera...\n\nMohon tunggu sebentar",
            bg="#2c3e50",
            fg="white",
            font=("Arial", 12),
            width=int(preview_width/10),  # Approximate character width
            height=int(preview_height/20)  # Approximate character height
        )
        self.preview_label.pack(padx=5, pady=5, expand=True)
        
        # Controls frame
        controls_frame = ttk.Frame(self)
        controls_frame.pack(pady=10)
        
        # Main buttons
        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack()
        
        self.capture_btn = ttk.Button(
            btn_frame,
            text="📸 Ambil Gambar (Space)",
            command=self.capture_frame,
            state="disabled",
            width=25
        )
        self.capture_btn.grid(row=0, column=0, padx=5)
        
        self.fullscreen_btn = ttk.Button(
            btn_frame,
            text="⛶ Fullscreen (F)",
            command=self.toggle_fullscreen,
            state="disabled",
            width=20
        )
        self.fullscreen_btn.grid(row=0, column=1, padx=5)
        
        ttk.Button(
            btn_frame,
            text="⬅️ Kembali (Esc)",
            command=self._go_back,
            width=20
        ).grid(row=0, column=2, padx=5)
        
        # Status frame with progress
        status_frame = ttk.Frame(self)
        status_frame.pack(pady=5, fill="x", padx=20)
        
        self.status_label = ttk.Label(
            status_frame,
            text="⏳ Status: Memuat kamera...",
            font=("Arial", 10),
            foreground="blue"
        )
        self.status_label.pack()
        
        # Progress bar (hidden by default)
        self.progress_bar = ttk.Progressbar(
            status_frame,
            mode='indeterminate',
            length=300
        )
        
        # Tips label
        tips_frame = ttk.Frame(self)
        tips_frame.pack(pady=5)
        
        ttk.Label(
            tips_frame,
            text="💡 Tips: Gunakan Space untuk capture, F untuk fullscreen, Esc untuk kembali",
            font=("Arial", 8),
            foreground="gray"
        ).pack()
        
        # Bind keyboard shortcuts
        self.bind_all('<space>', lambda e: self.capture_frame() if self.capture_btn['state'] == 'normal' else None)
        self.bind_all('<f>', lambda e: self.toggle_fullscreen() if self.fullscreen_btn['state'] == 'normal' else None)
        self.bind_all('<F>', lambda e: self.toggle_fullscreen() if self.fullscreen_btn['state'] == 'normal' else None)
        self.bind_all('<Escape>', lambda e: self._go_back())
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode for preview."""
        self.is_fullscreen = not self.is_fullscreen
        
        if self.is_fullscreen:
            preview_width = int(config["ui"]["window_size"][0] * 0.9)
            preview_height = int(config["ui"]["window_size"][1] * 0.85)
            self.fullscreen_btn.config(text="⛶ Normal View (F)")
        else:
            preview_width, preview_height = config["ui"]["preview_size"]
            self.fullscreen_btn.config(text="⛶ Fullscreen (F)")
        
        # Store new size in config temporarily
        config["ui"]["current_preview_size"] = [preview_width, preview_height]
        
        logger.info(f"Preview mode changed: fullscreen={self.is_fullscreen}, size={preview_width}x{preview_height}")
    
    def on_show(self):
        """Start camera when page is shown."""
        # Update device info
        device_name = getattr(self.controller, 'selected_device_name', 'Default Camera')
        self.device_info_label.config(text=f"🎥 Device: {device_name}")
        
        # Reset fullscreen state
        self.is_fullscreen = False
        config["ui"]["current_preview_size"] = config["ui"]["preview_size"]
        
        self.start_camera()
    
    def on_hide(self):
        """Stop camera when page is hidden."""
        self.stop_camera()
    
    def start_camera(self):
        """Start camera in background thread."""
        if self.running:
            logger.warning("Camera already running")
            return
        
        # Show progress bar
        self.progress_bar.pack(pady=5)
        self.progress_bar.start(10)
        
        # Get camera device
        device_name = getattr(self.controller, 'selected_device_name', None)
        device_index = getattr(self.controller, 'selected_device_index', 0)
        
        if device_name:
            logger.info(f"Starting camera: {device_name} (index {device_index})")
            self.status_label.config(
                text=f"⏳ Memulai kamera: {device_name}...",
                foreground="blue"
            )
        else:
            logger.info(f"Starting default camera (index {device_index})")
            device_index = 0
            self.status_label.config(
                text="⏳ Memulai kamera default...",
                foreground="blue"
            )
        
        self.running = True
        self.stop_event.clear()
        self.camera_thread = Thread(
            target=self._camera_worker,
            args=(device_index,),
            daemon=True,
            name="CameraThread"
        )
        self.camera_thread.start()
        
        # Start preview updates
        self.update_preview()
    
    def stop_camera(self):
        """Stop camera and cleanup resources."""
        logger.info("Stopping camera...")
        
        self.running = False
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=5)
            if self.camera_thread.is_alive():
                logger.warning("Camera thread did not stop gracefully")
        
        # Cleanup camera resource
        if self.cap is not None:
            try:
                self.cap.release()
                self.cap = None
                logger.info("Camera released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
        
        # Final OpenCV cleanup
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        logger.info("Camera stopped successfully")
    
    def _camera_worker(self, camera_index: int):
        """Camera worker thread (runs in background)."""
        logger.info(f"Camera worker started for index {camera_index}")
        
        try:
            # Cleanup before opening
            cv2.waitKey(1)
            time.sleep(0.3)
            
            # Try to open camera with retries
            camera_opened = False
            
            for attempt in range(3):
                logger.info(f"Opening camera attempt {attempt + 1}/3...")
                
                # Release previous attempt if exists
                if self.cap is not None:
                    self.cap.release()
                    cv2.waitKey(1)
                    time.sleep(0.2)
                
                # Open camera
                self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                
                if self.cap.isOpened():
                    # Test reading frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        logger.info(f"✅ Camera {camera_index} opened successfully")
                        camera_opened = True
                        break
                    else:
                        logger.warning(f"Camera opened but cannot read frame (attempt {attempt + 1})")
                        self.cap.release()
                        self.cap = None
                        time.sleep(0.5)
                else:
                    logger.warning(f"Cannot open camera (attempt {attempt + 1})")
                    time.sleep(0.5)
            
            if not camera_opened or self.cap is None:
                error_msg = f"Gagal membuka kamera setelah 3x percobaan"
                logger.error(error_msg)
                self.frame_queue.put({"error": error_msg})
                return
            
            # Configure camera
            try:
                width, height = config["camera"]["resolution"]
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config["camera"]["buffer_size"])
                logger.info(f"Camera configured: {width}x{height}")
            except Exception as e:
                logger.warning(f"Cannot configure camera: {e}")
            
            # Notify UI that camera is ready
            self.after(0, self._on_camera_ready)
            
            # Main streaming loop
            frame_count = 0
            consecutive_errors = 0
            max_consecutive_errors = 5
            target_fps = config["camera"]["fps"]
            frame_time = 1.0 / target_fps
            last_frame_time = time.time()
            
            logger.info("Camera streaming started")
            
            while self.running and not self.stop_event.is_set():
                try:
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    
                    # Frame rate control
                    if elapsed < frame_time:
                        time.sleep(0.001)
                        continue
                    
                    ret, frame = self.cap.read()
                    
                    if not ret or frame is None:
                        consecutive_errors += 1
                        logger.warning(f"Failed to read frame (error count: {consecutive_errors})")
                        
                        if consecutive_errors >= max_consecutive_errors:
                            error_msg = "Kamera berhenti merespon"
                            logger.error(error_msg)
                            self.frame_queue.put({"error": error_msg})
                            break
                        
                        time.sleep(0.1)
                        continue
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                    frame_count += 1
                    last_frame_time = current_time
                    
                    # Store current frame (thread-safe)
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                    
                    # Put frame to queue (non-blocking)
                    try:
                        self.frame_queue.put_nowait(frame)  # Tidak pakai timeout
                    except Full:
                        # Jika penuh, buang frame lama dan masukkan baru
                        try:
                            self.frame_queue.get_nowait()  # Buang frame lama
                        except Empty:
                            pass
                        try:
                            self.frame_queue.put_nowait(frame)  # Masukkan frame baru
                        except Full:
                            pass
                
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Error in streaming loop: {e}", exc_info=True)
                    
                    if consecutive_errors >= max_consecutive_errors:
                        error_msg = f"Error streaming: {str(e)}"
                        self.frame_queue.put({"error": error_msg})
                        break
                    
                    time.sleep(0.1)
            
            logger.info(f"Camera worker finished (captured {frame_count} frames)")
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.critical(f"Camera worker crash: {e}", exc_info=True)
            self.frame_queue.put({"error": error_msg})
        
        finally:
            # Cleanup
            logger.info("Cleaning up camera resources...")
            if self.cap is not None:
                try:
                    for _ in range(3):
                        self.cap.release()
                        cv2.waitKey(1)
                        time.sleep(0.1)
                    self.cap = None
                except Exception as e:
                    logger.error(f"Error during camera cleanup: {e}")
            
            cv2.waitKey(1)
            logger.info("Camera worker cleanup complete")
    
    def _on_camera_ready(self):
        """Called when camera is ready (runs in main thread)."""
        # Hide progress bar
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        
        # Enable buttons
        self.capture_btn.config(state="normal")
        self.fullscreen_btn.config(state="normal")
        
        # Update status with more details
        device_name = getattr(self.controller, 'selected_device_name', 'Default')
        self.status_label.config(
            text=f"✅ Kamera aktif - Siap mengambil gambar | Resolution: {config['camera']['resolution'][0]}x{config['camera']['resolution'][1]}",
            foreground="green"
        )
        logger.info("Camera ready for capture")
    
    def update_preview(self):
        """Update preview with latest frame (runs in main thread)."""
        if not self.running:
            return
        
        try:
            # Get frame from queue
            if not self.frame_queue.empty():
                data = self.frame_queue.get()
                
                # Check for error message
                if isinstance(data, dict) and "error" in data:
                    self._show_error(data['error'])
                    return
                
                # Display frame
                frame = data
                self._display_frame(frame)
        
        except Exception as e:
            logger.error(f"Error updating preview: {e}", exc_info=True)
            self.status_label.config(
                text=f"Error preview: {str(e)}",
                foreground="red"
            )
        
        # Schedule next update
        if self.running:
            self.after(30, self.update_preview)
    
    def _display_frame(self, frame):
        """Display frame on preview label."""
        try:
            # Cleanup old image
            if hasattr(self.preview_label, 'image') and self.preview_label.image:
                old_image = self.preview_label.image
                self.preview_label.image = None
                del old_image
            
            # Convert and resize
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            
            # Use current preview size (can be fullscreen or normal)
            preview_width, preview_height = config["ui"].get("current_preview_size", config["ui"]["preview_size"])
            img = img.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
            
            img_tk = ImageTk.PhotoImage(img)
            
            # Update label
            self.preview_label.configure(
                image=img_tk, 
                text="",
                bg="black"  # Black background for better contrast
            )
            self.preview_label.image = img_tk
            
        except Exception as e:
            logger.error(f"Error displaying frame: {e}", exc_info=True)
    
    def _show_error(self, error_msg: str):
        """Show error on preview."""
        # Hide progress bar if showing
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        
        self.preview_label.config(
            text=f"❌ ERROR\n\n{error_msg}\n\nCoba pilih kamera lain atau periksa koneksi",
            bg="#e74c3c",
            fg="white",
            font=("Arial", 11, "bold")
        )
        self.status_label.config(text=f"❌ Error: {error_msg}", foreground="red")
        self.capture_btn.config(state="disabled")
        self.fullscreen_btn.config(state="disabled")
        logger.error(f"Camera error: {error_msg}")
        
        # Offer recovery
        self.after(500, self._offer_recovery)
    
    def _offer_recovery(self):
        """Offer recovery options to user."""
        response = messagebox.askyesnocancel(
            "Kamera Error",
            f"Kamera gagal dimulai.\n\n"
            f"Opsi pemulihan:\n"
            f"• Klik 'Yes' untuk mencoba kamera lain\n"
            f"• Klik 'No' untuk kembali ke menu utama\n"
            f"• Klik 'Cancel' untuk retry kamera ini"
        )
        
        if response is True:  # Yes
            self.controller.show_page(CameraSelectionPage)
        elif response is False:  # No
            self.controller.show_page(HomePage)
        else:  # Cancel (retry)
            self.stop_camera()
            self.after(500, self.start_camera)
    
    def capture_frame(self):
        """Capture current frame and save."""
        if self.is_capturing:
            return
        
        self.is_capturing = True
        self.capture_btn.config(state="disabled", text="⏳ Menyimpan...")
        
        try:
            # Get current frame (thread-safe)
            with self.frame_lock:
                if self.current_frame is None:
                    raise ValueError("Tidak ada frame yang tersedia")
                frame = self.current_frame.copy()
            
            # Save frame
            cv2.imwrite(str(CAPTURE_PATH), frame)
            logger.info(f"Image captured and saved: {CAPTURE_PATH}")
            
            # Flash effect for visual feedback
            original_bg = self.preview_label.cget("bg")
            self.preview_label.config(bg="white")
            self.after(100, lambda: self.preview_label.config(bg=original_bg))
            
            # Show success
            self.status_label.config(
                text=f"✅ Gambar berhasil disimpan ke: {CAPTURE_PATH.name}",
                foreground="green"
            )
            
            # Play a subtle audio cue if possible (optional)
            self.bell()  # System beep
            
            # Show success dialog
            messagebox.showinfo(
                "✅ Berhasil",
                f"Gambar berhasil disimpan!\n\n"
                f"Lokasi: {CAPTURE_PATH}\n"
                f"Ukuran: {frame.shape[1]}x{frame.shape[0]} pixels"
            )
            
            # Ask user what to do next
            response = messagebox.askyesno(
                "Lanjutkan?",
                "Gambar berhasil diambil!\n\nLanjutkan ke analisis?"
            )
            
            if response:
                self.controller.show_page(AnalysisPage)
            else:
                # Re-enable capture button
                self.capture_btn.config(state="normal", text="📸 Ambil Gambar (Space)")
                self.is_capturing = False
        
        except Exception as e:
            logger.error(f"Error capturing frame: {e}", exc_info=True)
            messagebox.showerror(
                "❌ Error", 
                f"Gagal menyimpan gambar!\n\nDetail error:\n{str(e)}\n\nCoba lagi atau restart kamera."
            )
            self.capture_btn.config(state="normal", text="📸 Ambil Gambar (Space)")
            self.status_label.config(text=f"❌ Error: {str(e)}", foreground="red")
            self.is_capturing = False
    
    def _go_back(self):
        """Go back to camera selection."""
        if messagebox.askyesno("Konfirmasi", "Kembali ke pemilihan kamera?"):
            self.controller.show_page(CameraSelectionPage)


# ============================================================================
# Page 4: Analysis Page
# ============================================================================

class AnalysisPage(BasePage):
    """Halaman analisis hasil."""
    
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.status = StringVar(value="Klik 'Mulai Analisis' untuk memproses.")
        self.analysis_result = None
        self._build_ui()
    
    def _build_ui(self):
        """Build UI components."""
        ttk.Label(
            self,
            text="🔍 Hasil Analisis",
            font=("Arial", 16, "bold")
        ).pack(pady=15)
        
        # Image comparison frame
        image_frame = ttk.Frame(self)
        image_frame.pack(pady=10)
        
        # Sample image
        sample_frame = ttk.LabelFrame(image_frame, text="Sampel", padding=10)
        sample_frame.grid(row=0, column=0, padx=10)
        
        self.sample_view = Label(
            sample_frame,
            text="(Sampel)",
            bg="#e6e6e6",
            width=35,
            height=15
        )
        self.sample_view.pack()
        
        # Reference image
        ref_frame = ttk.LabelFrame(image_frame, text="Referensi", padding=10)
        ref_frame.grid(row=0, column=1, padx=10)
        
        self.match_view = Label(
            ref_frame,
            text="(Referensi)",
            bg="#e6e6e6",
            width=35,
            height=15
        )
        self.match_view.pack()
        
        # Result info
        self.result_frame = ttk.LabelFrame(self, text="Informasi Diagnosis", padding=15)
        self.result_frame.pack(pady=10, padx=20, fill="x")
        
        self.result_text = tk.Text(
            self.result_frame,
            height=8,
            width=70,
            wrap="word",
            font=("Arial", 10)
        )
        self.result_text.pack(fill="both", expand=True)
        self.result_text.config(state="disabled")
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        
        self.analyze_btn = ttk.Button(
            btn_frame,
            text="🔬 Mulai Analisis",
            command=self.run_analysis
        )
        self.analyze_btn.grid(row=0, column=0, padx=5)
        
        ttk.Button(
            btn_frame,
            text="📸 Ambil Ulang",
            command=lambda: controller.show_page(CapturePage)
        ).grid(row=0, column=1, padx=5)
        
        ttk.Button(
            btn_frame,
            text="🏠 Kembali",
            command=lambda: self.controller.show_page(HomePage)
        ).grid(row=0, column=2, padx=5)
        
        # Status
        status_label = ttk.Label(self, textvariable=self.status, font=("Arial", 10))
        status_label.pack(pady=5)
    
    def on_show(self):
        """Load sample image when page is shown."""
        self._load_sample_image()
    
    def _load_sample_image(self):
        """Load and display sample image."""
        if CAPTURE_PATH.exists():
            self._load_image_to_label(str(CAPTURE_PATH), self.sample_view)
            logger.info("Sample image loaded")
        else:
            self.sample_view.config(text="(Tidak ada sampel)")
            logger.warning("No sample image found")
    
    def run_analysis(self):
        """Run image analysis in background thread."""
        if not CAPTURE_PATH.exists():
            messagebox.showwarning("Peringatan", "Ambil gambar terlebih dahulu!")
            return
        
        logger.info("Starting analysis...")
        self.analyze_btn.config(state="disabled")
        self.status.set("⏳ Sedang menganalisis gambar...")
        
        def analysis_task():
            """Analysis task (runs in background)."""
            try:
                # Preprocess image
                self.status.set("⏳ Preprocessing gambar...")
                img = preprocess_image(str(CAPTURE_PATH))
                logger.debug("Image preprocessed")
                
                # Extract features
                self.status.set("⏳ Mengekstrak fitur...")
                features = extract_features(img)
                logger.debug("Features extracted")
                
                # Find best match
                self.status.set("⏳ Mencari kecocokan...")
                result = find_best_match(features)
                logger.debug(f"Match result: {result}")
                
                # Process result
                if result and result.get("score", 0) >= config["analysis"]["confidence_threshold"]:
                    self.after(0, lambda: self._display_success(result))
                else:
                    self.after(0, self._display_no_match)
            
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Analysis error: {e}", exc_info=True)
                self.after(0, lambda: self._display_error(error_msg))
        
        # Run in background thread
        Thread(target=analysis_task, daemon=True, name="AnalysisThread").start()
    
    def _display_success(self, result: Dict[str, Any]):
        """Display successful analysis result."""
        self.analysis_result = result
        
        # Display reference image
        if "image_path" in result:
            self._load_image_to_label(result["image_path"], self.match_view)
        
        # Display result information
        result_text = (
            f"🩸 Penyakit: {result.get('name', 'Unknown')}"
            f"🔬 Penyebab:{result.get('cause', 'N/A')}"
            f"💊 Solusi:{result.get('solution', 'N/A')}"
            f"📊 Tingkat Keyakinan: {result.get('score', 0)*100:.1f}%"
        )
        
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, result_text)
        self.result_text.config(state="disabled")
        
        self.status.set(f"✅ Diagnosis: {result.get('name')} ({result.get('score', 0)*100:.1f}%)")
        self.analyze_btn.config(state="normal")
        
        logger.info(f"Analysis complete: {result.get('name')} ({result.get('score', 0)*100:.1f}%)")
        
        messagebox.showinfo(
            "Analisis Selesai",
            f"Diagnosis: {result.get('name')}"
            f"Keyakinan: {result.get('score', 0)*100:.1f}%"
        )
    
    def _display_no_match(self):
        """Display no match result."""
        self.clear_views()
        
        result_text = (
            "❌ Tidak ditemukan kecocokan yang meyakinkan."
            "Kemungkinan penyebab:"
            "• Sampel darah tidak jelas"
            "• Belum ada referensi untuk penyakit ini"
            "• Gambar terlalu gelap atau terang"
            "Saran:"
            "• Ambil gambar ulang dengan pencahayaan lebih baik"
            "• Pastikan fokus mikroskop tepat"
        )
        
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, result_text)
        self.result_text.config(state="disabled")
        
        self.status.set("❌ Tidak ada kecocokan ditemukan")
        self.analyze_btn.config(state="normal")
        
        logger.warning("No match found")
        messagebox.showinfo("Hasil Analisis", "Tidak ditemukan kecocokan yang meyakinkan.")
    
    def _display_error(self, error_msg: str):
        """Display error message."""
        self.status.set(f"❌ Error: {error_msg}")
        self.analyze_btn.config(state="normal")
        
        messagebox.showerror("Error Analisis", f"Gagal menganalisis gambar:{error_msg}")
    
    def _load_image_to_label(self, path: str, label: Label):
        """Load and display image on label."""
        try:
            img = Image.open(path)
            img = img.resize((300, 220), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            label.configure(image=img_tk, text="")
            label.image = img_tk  # type: ignore  # Keep reference
            logger.debug(f"Image loaded: {path}")
        except Exception as e:
            label.configure(text="(Gambar tidak tersedia)")
            logger.error(f"Error loading image {path}: {e}")
    
    def clear_views(self):
        """Clear image views."""
        self.sample_view.configure(image="", text="(Sampel)")
        self.match_view.configure(image="", text="(Referensi)")
        
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state="disabled")


# ============================================================================
# Page 5: Database Page
# ============================================================================

class DatabasePage(BasePage):
    """Halaman database penyakit."""
    
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self._build_ui()
    
    def _build_ui(self):
        """Build UI components."""
        ttk.Label(
            self,
            text="📁 Database Penyakit",
            font=("Arial", 16, "bold")
        ).pack(pady=15)
        
        info_text = (
            "Fitur ini akan menampilkan:"
            "• Daftar penyakit yang dikenali sistem"
            "• Gambar referensi untuk setiap penyakit"
            "• Informasi detail (penyebab & solusi)"
            "• Statistik analisis"
            "Status: 🚧 Dalam Pengembangan"
        )
        
        info_frame = ttk.Frame(self)
        info_frame.pack(pady=30)
        
        ttk.Label(
            info_frame,
            text=info_text,
            justify="left",
            font=("Arial", 11)
        ).pack()
        
        ttk.Button(
            self,
            text="⬅️ Kembali",
            command=lambda: self.controller.show_page(HomePage)
        ).pack(pady=20)


# ============================================================================
# Main Application Controller
# ============================================================================

class BloodAnalyzerApp:
    """Main application controller."""
    
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Digital Blood Analyzer v2.0")
        self.root.geometry("900x700")
        self.root.resizable(False, False)
        
        # Application state
        self.selected_device_name: Optional[str] = None
        self.selected_device_index: Optional[int] = None
        self.current_page = None
        
        # Setup UI
        self._setup_ui()
        
        # Setup cleanup handler
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        
        # Show home page
        self.show_page(HomePage)
        
        logger.info("Application started")
    
    def _setup_ui(self):
        """Setup main UI container."""
        # Main container
        self.container = ttk.Frame(self.root)
        self.container.pack(fill="both", expand=True)
        
        # Configure grid
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        # Create all pages
        self.frames = {}
        all_pages = (
            HomePage,
            CameraSelectionPage,
            CapturePage,
            AnalysisPage,
            DatabasePage
        )
        
        for PageClass in all_pages:
            try:
                frame = PageClass(self.container, self)
                self.frames[PageClass] = frame
                frame.grid(row=0, column=0, sticky="nsew")
                logger.debug(f"Created page: {PageClass.__name__}")
            except Exception as e:
                logger.error(f"Error creating page {PageClass.__name__}: {e}", exc_info=True)
    
    def show_page(self, page_class):
        """Show specified page."""
        try:
            # Hide current page
            if self.current_page:
                self.current_page.on_hide()
            
            # Show new page
            frame = self.frames[page_class]
            frame.tkraise()
            frame.on_show()
            
            self.current_page = frame
            logger.info(f"Showing page: {page_class.__name__}")
        
        except Exception as e:
            logger.error(f"Error showing page {page_class.__name__}: {e}", exc_info=True)
            messagebox.showerror("Error", f"Gagal menampilkan halaman:{e}")
    
    def quit_app(self):
        """Quit application with cleanup."""
        logger.info("Application closing...")
        
        try:
            # Cleanup current page
            if self.current_page:
                self.current_page.on_hide()
            
            # Cleanup OpenCV resources
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
            logger.info("Application closed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
        finally:
            self.root.destroy()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""
    try:
        logger.info("=" * 60)
        logger.info("Digital Blood Analyzer v2.0 - Starting")
        logger.info("=" * 60)
        
        root = Tk()
        app = BloodAnalyzerApp(root)
        root.mainloop()
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        messagebox.showerror(
            "Fatal Error",
            f"Aplikasi mengalami error fatal:{e}Lihat log untuk detail."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()