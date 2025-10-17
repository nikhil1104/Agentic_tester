# modules/voice_backends/base.py
"""
Abstract base class for voice backends
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class VoiceBackend(ABC):
    """Abstract base class for voice transcription backends"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize backend with configuration"""
        pass
    
    @abstractmethod
    def transcribe(self, audio_file: Path) -> str:
        """
        Transcribe audio file to text
        
        Args:
            audio_file: Path to audio file
        
        Returns:
            Transcribed text
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available and configured"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get backend information (model, cost, etc.)"""
        pass
