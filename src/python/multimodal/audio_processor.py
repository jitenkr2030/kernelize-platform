"""
KERNELIZE Platform - Audio Processor
=====================================

This module implements audio processing capabilities for the KERNELIZE Platform.
It provides speech transcription using OpenAI Whisper, audio feature extraction,
and automatic summarization of audio content.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """
    Audio Segment with Timestamps
    
    Represents a portion of audio transcription with precise timing
    information for alignment and reference.
    """
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    text: str          # Transcribed text for this segment
    confidence: float  # Transcription confidence (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary for serialization"""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass
class AudioAnalysisResult:
    """
    Audio Analysis Result
    
    Complete analysis of audio content including full transcript,
    timestamped segments, detected language, and generated summary.
    """
    transcript: str
    segments: List[AudioSegment]
    language: str
    language_confidence: float
    summary: Optional[str]
    duration_seconds: float
    sample_rate: int
    audio_features: Dict[str, Any]
    processing_time_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "transcript": self.transcript,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "language_confidence": self.language_confidence,
            "summary": self.summary,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "audio_features": self.audio_features,
            "processing_time_ms": self.processing_time_ms,
        }


class AudioProcessor:
    """
    Audio Analysis Processor
    
    Handles audio transcription, feature extraction, and summarization
    for the KERNELIZE knowledge compression system. Supports multiple
    audio formats and provides integration with OpenAI Whisper model.
    """
    
    SUPPORTED_FORMATS = {"WAV", "MP3", "FLAC", "OGG", "M4A", "WEBM"}
    WHISPER_SAMPLE_RATE = 16000  # Whisper expects 16kHz mono audio
    MAX_AUDIO_LENGTH = 3600  # Maximum 1 hour per processing request
    
    def __init__(
        self,
        enable_transcription: bool = True,
        enable_summary: bool = True,
        enable_features: bool = True,
        whisper_model: str = "base",  # tiny, base, small, medium, large
        device: Optional[str] = None,  # "cuda", "cpu", or None for auto
    ):
        """
        Initialize the audio processor with specified configuration.
        
        Args:
            enable_transcription: Whether to perform speech-to-text
            enable_summary: Whether to generate audio summary
            enable_features: Whether to extract audio features
            whisper_model: Size of Whisper model to use
            device: Computation device ("cuda", "cpu", or auto-detect)
        """
        self.enable_transcription = enable_transcription
        self.enable_summary = enable_summary
        self.enable_features = enable_features
        self.whisper_model = whisper_model
        self.device = device
        
        # Model placeholder (lazy loaded)
        self._whisper_model = None
        
        # Set device
        self._setup_device()
        
        logger.info(f"AudioProcessor initialized with transcription={enable_transcription}, "
                   f"summary={enable_summary}, model={whisper_model}")
    
    def _setup_device(self) -> None:
        """
        Determine and configure the computation device.
        
        Sets up CUDA if available, otherwise defaults to CPU.
        """
        self._use_cuda = False
        
        if self.device:
            if self.device.lower() == "cuda":
                try:
                    import torch
                    self._use_cuda = torch.cuda.is_available()
                    if self._use_cuda:
                        logger.info("Using CUDA for audio processing")
                    else:
                        logger.warning("CUDA requested but not available")
                except ImportError:
                    logger.warning("PyTorch not installed, cannot use CUDA")
        else:
            try:
                import torch
                self._use_cuda = torch.cuda.is_available()
                if self._use_cuda:
                    logger.info("Auto-detected CUDA, using GPU acceleration")
            except ImportError:
                pass
    
    def process(
        self,
        audio_data: bytes,
        options: Optional[Dict[str, Any]] = None,
    ) -> AudioAnalysisResult:
        """
        Process audio data and return comprehensive analysis.
        
        Args:
            audio_data: Raw audio bytes
            options: Optional processing configuration
            
        Returns:
            AudioAnalysisResult containing all analysis outputs
        """
        import time
        start_time = time.time()
        
        options = options or {}
        
        # Parse options
        enable_transcription = options.get("enable_transcription", self.enable_transcription)
        enable_summary = options.get("enable_summary", self.enable_summary)
        enable_features = options.get("enable_features", self.enable_features)
        language = options.get("language", None)  # None for auto-detect
        
        try:
            # Load and validate audio
            audio_array, sample_rate = self._load_audio(audio_data)
            duration = len(audio_array) / sample_rate
            
            # Check duration limit
            if duration > self.MAX_AUDIO_LENGTH:
                logger.warning(f"Audio exceeds max length ({duration}s > {self.MAX_AUDIO_LENGTH}s)")
            
            # Extract audio features
            audio_features = {}
            if enable_features:
                audio_features = self._extract_features(audio_array, sample_rate)
            
            # Transcribe audio
            transcript = ""
            segments = []
            detected_language = "unknown"
            language_confidence = 0.0
            
            if enable_transcription:
                transcript, segments, detected_language, language_confidence = (
                    self._transcribe(audio_array, sample_rate, language)
                )
            
            # Generate summary
            summary = None
            if enable_summary and transcript:
                summary = self._summarize(transcript)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = AudioAnalysisResult(
                transcript=transcript,
                segments=segments,
                language=detected_language,
                language_confidence=language_confidence,
                summary=summary,
                duration_seconds=duration,
                sample_rate=sample_rate,
                audio_features=audio_features,
                processing_time_ms=processing_time_ms,
            )
            
            logger.info(f"Audio processed in {processing_time_ms}ms: "
                       f"duration={duration:.1f}s, language={detected_language}")
            
            return result
            
        except Exception as e:
            logger.exception(f"Audio processing failed: {e}")
            raise ValueError(f"Failed to process audio: {str(e)}")
    
    def _load_audio(self, audio_data: bytes) -> Tuple[np.ndarray, int]:
        """
        Load audio data and convert to required format.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            import soundfile as sf
            import librosa
            
            # Try soundfile first (supports most formats)
            audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample if necessary
            if sample_rate != self.WHISPER_SAMPLE_RATE:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sample_rate,
                    target_sr=self.WHISPER_SAMPLE_RATE,
                )
                sample_rate = self.WHISPER_SAMPLE_RATE
            
            return audio_array, sample_rate
            
        except Exception as e:
            logger.debug(f"Soundfile failed ({e}), trying alternative method")
            
            # Fallback using torchaudio or simple WAV parsing
            try:
                import torchaudio
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_data)
                    tmp_path = tmp.name
                
                try:
                    waveform, sample_rate = torchaudio.load(tmp_path)
                    
                    # Convert to mono
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0)
                    
                    audio_array = waveform.numpy()
                    
                    # Resample if necessary
                    if sample_rate != self.WHISPER_SAMPLE_RATE:
                        import torch
                        resampler = torchaudio.transforms.Resample(
                            sample_rate,
                            self.WHISPER_SAMPLE_RATE,
                        )
                        audio_array = resampler(
                            torch.tensor(audio_array).unsqueeze(0)
                        ).squeeze().numpy()
                        sample_rate = self.WHISPER_SAMPLE_RATE
                    
                    return audio_array, sample_rate
                    
                finally:
                    os.unlink(tmp_path)
                    
            except Exception as e2:
                logger.error(f"Audio loading failed: {e2}")
                raise ValueError(f"Unsupported audio format: {str(e2)}")
    
    def _transcribe(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> Tuple[str, List[AudioSegment], str, float]:
        """
        Transcribe audio using OpenAI Whisper model.
        
        Args:
            audio_array: Audio waveform as numpy array
            sample_rate: Audio sample rate
            language: Target language (None for auto-detect)
            
        Returns:
            Tuple of (transcript, segments, detected_language, language_confidence)
        """
        try:
            # Lazy load Whisper model
            if self._whisper_model is None:
                self._load_whisper_model()
            
            # Transcribe using Whisper
            with self._whisper_model.device:
                result = self._whisper_model.transcribe(
                    audio_array,
                    language=language,
                    verbose=False,
                    beam_size=5,
                    best_of=5,
                )
            
            # Extract segments
            segments = []
            for segment in result.get("segments", []):
                segments.append(AudioSegment(
                    start_time=segment.get("start", 0),
                    end_time=segment.get("end", 0),
                    text=segment.get("text", "").strip(),
                    confidence=segment.get("avg_logprob", -0.5),
                ))
            
            # Build full transcript
            transcript = " ".join(s.text for s in segments)
            
            # Get detected language info
            detected_language = result.get("language", "unknown")
            language_confidence = result.get("probability", 0.0)
            
            return transcript, segments, detected_language, language_confidence
            
        except Exception as e:
            logger.warning(f"Transcription failed: {e}")
            return "", [], "unknown", 0.0
    
    def _extract_features(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, Any]:
        """
        Extract audio features for analysis and indexing.
        
        Args:
            audio_array: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Dictionary containing extracted features
        """
        try:
            import librosa
            
            features = {}
            
            # Duration
            features["duration"] = len(audio_array) / sample_rate
            
            # Energy/Volume statistics
            rms_energy = librosa.feature.rms(y=audio_array)[0]
            features["mean_energy"] = float(np.mean(rms_energy))
            features["max_energy"] = float(np.max(rms_energy))
            features["dynamic_range"] = float(
                20 * np.log10(np.max(rms_energy) / (np.max(rms_energy) + 1e-10))
            )
            
            # Zero crossing rate (for speech/music detection)
            zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
            features["mean_zcr"] = float(np.mean(zcr))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_array, sr=sample_rate
            )[0]
            features["mean_spectral_centroid"] = float(np.mean(spectral_centroid))
            
            # Pitch estimation (fundamental frequency)
            try:
                pitches, magnitudes = librosa.piptrack(
                    y=audio_array, sr=sample_rate
                )
                pitch_values = pitches[magnitudes > np.median(magnitudes)]
                if len(pitch_values) > 0:
                    features["mean_pitch"] = float(np.mean(pitch_values))
                else:
                    features["mean_pitch"] = 0.0
            except Exception:
                features["mean_pitch"] = 0.0
            
            # Silence detection
            silence_threshold = 0.01
            non_silent_ratio = np.sum(rms_energy > silence_threshold) / len(rms_energy)
            features["speech_ratio"] = float(non_silent_ratio)
            
            # Speech rate estimation (words per minute approximation)
            # Based on energy peaks and silence intervals
            features["speech_rate_estimate"] = self._estimate_speech_rate(
                audio_array, sample_rate, rms_energy
            )
            
            return features
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return {"error": str(e)}
    
    def _estimate_speech_rate(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        rms_energy: np.ndarray,
    ) -> float:
        """
        Estimate approximate speech rate in words per minute.
        
        Args:
            audio_array: Audio waveform
            sample_rate: Sample rate
            rms_energy: Root mean square energy values
            
        Returns:
            Estimated speech rate in WPM
        """
        # Count energy peaks (rough approximation of word boundaries)
        from scipy.signal import find_peaks
        
        # Find peaks in energy signal
        peaks, properties = find_peaks(
            rms_energy,
            height=np.mean(rms_energy),
            distance=sample_rate // 2000,  # ~0.5 second minimum between words
        )
        
        # Estimate duration in minutes
        duration_minutes = len(audio_array) / sample_rate / 60
        
        if duration_minutes > 0 and len(peaks) > 0:
            return len(peaks) / duration_minutes
        
        return 0.0
    
    def _summarize(self, transcript: str) -> str:
        """
        Generate a summary of the transcript.
        
        Args:
            transcript: Full audio transcript
            
        Returns:
            Generated summary text
        """
        try:
            # Simple extractive summarization
            # In production, this would use an LLM for abstractive summarization
            
            if len(transcript) < 200:
                return transcript
            
            # Split into sentences
            import re
            sentences = re.split(r'[.!?]+', transcript)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 3:
                return transcript
            
            # Score sentences by length and position
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = len(sentence.split())  # Word count
                # Boost first and last sentences
                if i == 0:
                    score *= 1.5
                if i == len(sentences) - 1:
                    score *= 1.3
                sentence_scores.append((i, sentence, score))
            
            # Select top sentences
            sentence_scores.sort(key=lambda x: x[2], reverse=True)
            num_summary_sentences = min(3, len(sentences) // 3 + 1)
            selected_indices = [s[0] for s in sentence_scores[:num_summary_sentences]]
            selected_indices.sort()
            
            # Build summary
            summary = " ".join(sentences[i] for i in selected_indices)
            
            return summary if summary else None
            
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return None
    
    def _load_whisper_model(self) -> None:
        """
        Load OpenAI Whisper model for transcription.
        
        Downloads model if not cached locally.
        """
        try:
            import whisper
            
            logger.info(f"Loading Whisper model: {self.whisper_model}")
            
            self._whisper_model = whisper.load_model(
                self.whisper_model,
                device=self._whisper_model.device if self._use_cuda else "cpu",
                download_root=os.environ.get("WHISPER_CACHE_DIR", "./cache"),
            )
            
            self._whisper_model.eval()
            
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def clear_cache(self) -> None:
        """
        Clear loaded models from memory.
        
        Useful for memory-constrained environments.
        """
        import torch
        
        self._whisper_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model cache cleared")
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics and model information.
        
        Returns:
            Dictionary containing processor status and capabilities
        """
        return {
            "supported_formats": list(self.SUPPORTED_FORMATS),
            "whisper_sample_rate": self.WHISPER_SAMPLE_RATE,
            "max_audio_length_seconds": self.MAX_AUDIO_LENGTH,
            "transcription_enabled": self.enable_transcription,
            "summary_enabled": self.enable_summary,
            "features_enabled": self.enable_features,
            "whisper_model": self.whisper_model,
            "device": "cuda" if self._use_cuda else "cpu",
            "model_loaded": self._whisper_model is not None,
        }
