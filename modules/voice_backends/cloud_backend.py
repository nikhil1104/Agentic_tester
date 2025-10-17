# modules/voice_backends/cloud_backend.py
"""
OpenAI Whisper API backend (PAID)
Uses OpenAI's cloud API
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any

from .base import VoiceBackend

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class CloudWhisperBackend(VoiceBackend):
    """PAID OpenAI Whisper API backend"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.model = config.get('model', 'whisper-1')
        
        if self.is_available():
            self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client"""
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info("✅ OpenAI Whisper API configured")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def transcribe(self, audio_file: Path) -> str:
        """Transcribe audio using OpenAI API ($0.006 per minute)"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            with open(audio_file, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=f,
                    language=self.config.get('language', 'en')
                )
            
            return transcript.text.strip()
        
        except Exception as e:
            logger.error(f"Cloud transcription failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available"""
        return OPENAI_AVAILABLE and bool(self.api_key)
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend info"""
        return {
            "name": "OpenAI Whisper API",
            "type": "cloud",
            "model": self.model,
            "cost_per_minute": 0.006,
            "requires_internet": True,
            "privacy": "Medium - Audio sent to OpenAI"
        }
