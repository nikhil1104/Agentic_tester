# modules/voice_backends/local_backend.py
"""
Local Whisper backend (FREE)
Uses open-source Whisper models
"""
import logging
from pathlib import Path
from typing import Dict, Any

from .base import VoiceBackend

logger = logging.getLogger(__name__)

# Try faster-whisper first
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER = True
except ImportError:
    FASTER_WHISPER = False
    try:
        import whisper
        STANDARD_WHISPER = True
    except ImportError:
        STANDARD_WHISPER = False


class LocalWhisperBackend(VoiceBackend):
    """FREE local Whisper backend"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.model_size = config.get('model_size', 'base')
        self.device = config.get('device', 'cpu')
        self.use_faster = config.get('use_faster_whisper', True) and FASTER_WHISPER
        
        if self.is_available():
            self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            if self.use_faster:
                # Faster-whisper (5-10x faster)
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.config.get('compute_type', 'int8')
                )
                logger.info(f"✅ Loaded faster-whisper: {self.model_size}")
            else:
                # Standard whisper
                self.model = whisper.load_model(self.model_size, device=self.device)
                logger.info(f"✅ Loaded whisper: {self.model_size}")
        except Exception as e:
            logger.error(f"Failed to load local Whisper: {e}")
            raise
    
    def transcribe(self, audio_file: Path) -> str:
        """Transcribe audio file (100% FREE)"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            language = self.config.get('language', 'en')
            
            if self.use_faster:
                segments, _ = self.model.transcribe(str(audio_file), language=language)
                text = " ".join([seg.text for seg in segments])
            else:
                result = self.model.transcribe(str(audio_file), language=language, fp16=False)
                text = result["text"]
            
            return text.strip()
        
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if local Whisper is available"""
        return FASTER_WHISPER or STANDARD_WHISPER
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend info"""
        return {
            "name": "Local Whisper",
            "type": "local",
            "model": self.model_size,
            "device": self.device,
            "faster_whisper": self.use_faster,
            "cost_per_minute": 0.0,
            "requires_internet": False,
            "privacy": "High - Audio never leaves device"
        }
