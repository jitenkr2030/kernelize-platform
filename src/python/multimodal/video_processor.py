"""
KERNELIZE Platform - Video Processor
=====================================

This module implements video processing capabilities for the KERNELIZE Platform.
It provides scene detection, keyframe extraction, and integration with audio
and image processors for comprehensive video analysis.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import base64
import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoScene:
    """
    Video Scene with Keyframes and Description
    
    Represents a detected scene in video with extracted keyframes
    and semantic descriptions.
    """
    scene_index: int
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    keyframe_id: str   # Base64-encoded keyframe image
    keyframe_timestamp: float  # Timestamp of keyframe
    description: str   # Generated scene description
    embedding: Optional[List[float]]  # CLIP embedding of keyframe
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scene to dictionary for serialization"""
        return {
            "scene_index": self.scene_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "keyframe_timestamp": self.keyframe_timestamp,
            "description": self.description,
            "embedding": self.embedding,
        }


@dataclass
class VideoAnalysisResult:
    """
    Video Analysis Result
    
    Complete analysis of video content including scene segmentation,
    extracted keyframes, audio transcription, and semantic indexing.
    """
    total_duration: float
    scene_count: int
    scenes: List[VideoScene]
    fps: float
    resolution: Tuple[int, int]
    audio_analysis: Optional["AudioAnalysisResult"]  # Forward reference
    keyframe_images: Dict[str, str]  # keyframe_id -> base64 image
    processing_time_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "total_duration": self.total_duration,
            "scene_count": self.scene_count,
            "scenes": [s.to_dict() for s in self.scenes],
            "fps": self.fps,
            "resolution": self.resolution,
            "audio_analysis": (
                self.audio_analysis.to_dict()
                if self.audio_analysis
                else None
            ),
            "processing_time_ms": self.processing_time_ms,
        }


class VideoProcessor:
    """
    Video Analysis Processor
    
    Handles video scene detection, keyframe extraction, and orchestrates
    audio and image processing for comprehensive video analysis.
    Supports common video formats and provides configurable processing options.
    """
    
    SUPPORTED_FORMATS = {"MP4", "AVI", "MOV", "MKV", "WEBM", "WMV"}
    MIN_SCENE_DURATION = 2.0  # Minimum seconds per scene
    KEYFRAME_INTERVAL = 5.0  # Default interval between keyframes
    SCENE_CHANGE_THRESHOLD = 0.4  # Threshold for scene change detection
    
    def __init__(
        self,
        enable_scene_detection: bool = True,
        enable_keyframes: bool = True,
        enable_audio: bool = True,
        scene_detection_method: str = "content",  # "content", "threshold", "interval"
        keyframe_interval: float = 5.0,
        max_scenes: int = 50,
    ):
        """
        Initialize the video processor with specified configuration.
        
        Args:
            enable_scene_detection: Whether to detect scene changes
            enable_keyframes: Whether to extract keyframes
            enable_audio: Whether to process audio track
            scene_detection_method: Method for scene detection
            keyframe_interval: Seconds between keyframe extractions
            max_scenes: Maximum number of scenes to detect
        """
        self.enable_scene_detection = enable_scene_detection
        self.enable_keyframes = enable_keyframes
        self.enable_audio = enable_audio
        self.scene_detection_method = scene_detection_method
        self.keyframe_interval = keyframe_interval
        self.max_scenes = max_scenes
        
        # Sub-processors
        self._image_processor = None
        self._audio_processor = None
        
        logger.info(f"VideoProcessor initialized with scene_detection={enable_scene_detection}, "
                   f"keyframes={enable_keyframes}, audio={enable_audio}")
    
    def process(
        self,
        video_data: bytes,
        options: Optional[Dict[str, Any]] = None,
    ) -> VideoAnalysisResult:
        """
        Process video data and return comprehensive analysis.
        
        Args:
            video_data: Raw video bytes
            options: Optional processing configuration
            
        Returns:
            VideoAnalysisResult containing all analysis outputs
        """
        import time
        start_time = time.time()
        
        options = options or {}
        
        # Parse options
        enable_scene_detection = options.get("enable_scene_detection", self.enable_scene_detection)
        enable_keyframes = options.get("enable_keyframes", self.enable_keyframes)
        enable_audio = options.get("enable_audio", self.enable_audio)
        keyframe_interval = options.get("keyframe_interval", self.keyframe_interval)
        
        try:
            # Create temporary file for video
            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False,
            ) as tmp_file:
                tmp_file.write(video_data)
                video_path = tmp_file.name
            
            try:
                # Get video properties
                fps, frame_count, resolution = self._get_video_properties(video_path)
                duration = frame_count / fps if fps > 0 else 0
                
                # Detect scenes
                scene_boundaries = []
                if enable_scene_detection:
                    scene_boundaries = self._detect_scenes(
                        video_path,
                        fps,
                        frame_count,
                        duration,
                    )
                
                # Extract keyframes
                keyframes = {}
                scene_keyframes = []
                if enable_keyframes:
                    keyframes, scene_keyframes = self._extract_keyframes(
                        video_path,
                        scene_boundaries,
                        keyframe_interval,
                    )
                
                # Analyze keyframes with image processor
                scene_descriptions = []
                if enable_keyframes and keyframes:
                    scene_descriptions = self._analyze_keyframes(keyframes)
                
                # Process audio track
                audio_analysis = None
                if enable_audio:
                    audio_analysis = self._process_audio(video_path)
                
                # Build scene objects
                scenes = self._build_scenes(
                    scene_boundaries,
                    scene_keyframes,
                    keyframes,
                    scene_descriptions,
                )
                
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                result = VideoAnalysisResult(
                    total_duration=duration,
                    scene_count=len(scenes),
                    scenes=scenes,
                    fps=fps,
                    resolution=resolution,
                    audio_analysis=audio_analysis,
                    keyframe_images=keyframes,
                    processing_time_ms=processing_time_ms,
                )
                
                logger.info(f"Video processed in {processing_time_ms}ms: "
                           f"duration={duration:.1f}s, scenes={len(scenes)}")
                
                return result
                
            finally:
                # Clean up temporary file
                if os.path.exists(video_path):
                    os.unlink(video_path)
                    
        except Exception as e:
            logger.exception(f"Video processing failed: {e}")
            raise ValueError(f"Failed to process video: {str(e)}")
    
    def _get_video_properties(
        self,
        video_path: str,
    ) -> Tuple[float, int, Tuple[int, int]]:
        """
        Extract video properties including FPS, frame count, and resolution.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (fps, frame_count, resolution)
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            return fps, frame_count, (width, height)
            
        except Exception as e:
            logger.warning(f"Could not read video properties: {e}")
            return 30.0, 0, (0, 0)
    
    def _detect_scenes(
        self,
        video_path: str,
        fps: float,
        frame_count: int,
        duration: float,
    ) -> List[Tuple[float, float]]:
        """
        Detect scene boundaries in video.
        
        Args:
            video_path: Path to video file
            fps: Frames per second
            frame_count: Total frame count
            duration: Video duration in seconds
            
        Returns:
            List of (start_time, end_time) tuples for each scene
        """
        try:
            import cv2
            
            scenes = []
            prev_frame = None
            frame_interval = max(1, int(fps * 0.5))  # Sample every 0.5 seconds
            
            for frame_idx in range(0, frame_count, frame_interval):
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply histogram equalization
                gray = cv2.equalizeHist(gray)
                
                if prev_frame is not None:
                    # Calculate difference between frames
                    diff = cv2.absdiff(prev_frame, gray)
                    
                    # Calculate percentage of different pixels
                    change_ratio = np.count_nonzero(diff) / diff.size
                    
                    # Detect scene change
                    if change_ratio > self.SCENE_CHANGE_THRESHOLD:
                        start_time = (frame_idx - frame_interval) / fps
                        end_time = frame_idx / fps
                        
                        # Ensure minimum scene duration
                        if not scenes or (start_time - scenes[-1][1]) >= self.MIN_SCENE_DURATION:
                            scenes.append((start_time, frame_idx / fps))
                
                prev_frame = gray
            
            # Add final scene
            if not scenes or duration - scenes[-1][1] >= self.MIN_SCENE_DURATION:
                scenes.append((scenes[-1][1] if scenes else 0, duration))
            
            # Limit to max scenes
            scenes = scenes[:self.max_scenes]
            
            logger.debug(f"Detected {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            logger.warning(f"Scene detection failed: {e}")
            # Return single scene covering entire video
            return [(0.0, duration)]
    
    def _extract_keyframes(
        self,
        video_path: str,
        scene_boundaries: List[Tuple[float, float]],
        keyframe_interval: float,
    ) -> Tuple[Dict[str, str], List[Tuple[float, float]]]:
        """
        Extract keyframes from video.
        
        Args:
            video_path: Path to video file
            scene_boundaries: Detected scene boundaries
            keyframe_interval: Interval between keyframes
            
        Returns:
            Tuple of (keyframes_dict, scene_keyframe_timestamps)
        """
        try:
            import cv2
            
            keyframes = {}
            scene_keyframes = []
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Determine keyframe extraction strategy
            if scene_boundaries:
                # Extract keyframes at scene boundaries and middle
                for i, (start, end) in enumerate(scene_boundaries):
                    # Keyframe at scene start
                    start_frame = int(start * fps)
                    timestamp = start
                    
                    keyframe_id = self._extract_keyframe_at_frame(
                        cap, start_frame, keyframes
                    )
                    if keyframe_id:
                        scene_keyframes.append(timestamp)
                    
                    # Additional keyframe for longer scenes
                    if (end - start) > keyframe_interval * 2:
                        middle_frame = int((start + end) / 2 * fps)
                        timestamp = (start + end) / 2
                        
                        keyframe_id = self._extract_keyframe_at_frame(
                            cap, middle_frame, keyframes
                        )
                        if keyframe_id:
                            scene_keyframes.append(timestamp)
            else:
                # Extract keyframes at regular intervals
                frame_interval = int(keyframe_interval * fps)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                for frame_idx in range(0, total_frames, frame_interval):
                    timestamp = frame_idx / fps
                    keyframe_id = self._extract_keyframe_at_frame(
                        cap, frame_idx, keyframes
                    )
                    if keyframe_id:
                        scene_keyframes.append(timestamp)
            
            cap.release()
            
            return keyframes, scene_keyframes
            
        except Exception as e:
            logger.warning(f"Keyframe extraction failed: {e}")
            return {}, []
    
    def _extract_keyframe_at_frame(
        self,
        cap,
        frame_idx: int,
        keyframes: Dict[str, str],
    ) -> Optional[str]:
        """
        Extract and encode a single keyframe.
        
        Args:
            cap: OpenCV video capture object
            frame_idx: Frame index to extract
            keyframes: Dictionary to store keyframes
            
        Returns:
            Keyframe ID or None if extraction failed
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            return None
        
        # Generate keyframe ID
        keyframe_id = f"kf_{frame_idx}_{len(keyframes)}"
        
        # Encode as JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Convert to base64
        keyframes[keyframe_id] = base64.b64encode(buffer).decode("utf-8")
        
        return keyframe_id
    
    def _analyze_keyframes(
        self,
        keyframes: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Analyze keyframes using image processor to generate descriptions.
        
        Args:
            keyframes: Dictionary of keyframe_id -> base64_image
            
        Returns:
            Dictionary of keyframe_id -> description
        """
        try:
            if self._image_processor is None:
                self._image_processor = ImageProcessor(
                    enable_ocr=False,
                    enable_caption=True,
                    enable_embedding=True,
                )
            
            descriptions = {}
            
            for keyframe_id, base64_image in keyframes.items():
                # Decode base64 to bytes
                image_bytes = base64.b64decode(base64_image)
                
                # Process with image processor
                result = self._image_processor.process(image_bytes)
                
                descriptions[keyframe_id] = result.caption
            
            return descriptions
            
        except Exception as e:
            logger.warning(f"Keyframe analysis failed: {e}")
            return {kf_id: "" for kf_id in keyframes.keys()}
    
    def _process_audio(self, video_path: str) -> "AudioAnalysisResult":
        """
        Extract and process audio track from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            AudioAnalysisResult
        """
        try:
            if self._audio_processor is None:
                self._audio_processor = AudioProcessor(
                    enable_transcription=True,
                    enable_summary=True,
                    enable_features=True,
                )
            
            # Extract audio using ffmpeg
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_path = tmp.name
            
            try:
                import subprocess
                
                # Extract audio track
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-vn",  # No video
                    "-acodec", "pcm_s16le",  # WAV format
                    "-ar", "16000",  # 16kHz for Whisper
                    "-ac", "1",  # Mono
                    audio_path,
                ], capture_output=True, check=True)
                
                # Read audio data
                with open(audio_path, "rb") as f:
                    audio_data = f.read()
                
                # Process with audio processor
                return self._audio_processor.process(audio_data)
                
            finally:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                    
        except Exception as e:
            logger.warning(f"Audio processing failed: {e}")
            return None
    
    def _build_scenes(
        self,
        scene_boundaries: List[Tuple[float, float]],
        scene_keyframe_timestamps: List[float],
        keyframes: Dict[str, str],
        descriptions: Dict[str, str],
    ) -> List[VideoScene]:
        """
        Build VideoScene objects from detection results.
        
        Args:
            scene_boundaries: Detected scene boundaries
            scene_keyframe_timestamps: Keyframe timestamps
            keyframes: Extracted keyframes
            descriptions: Generated descriptions
            
        Returns:
            List of VideoScene objects
        """
        scenes = []
        
        # Create mapping of timestamp to keyframe
        timestamp_to_keyframe = {}
        for kf_id, b64_image in keyframes.items():
            # Extract timestamp from keyframe_id (approximate)
            parts = kf_id.split("_")
            if len(parts) >= 2:
                try:
                    frame_idx = int(parts[1])
                    # Assume 30fps for timestamp calculation
                    timestamp = frame_idx / 30
                    timestamp_to_keyframe[timestamp] = kf_id
                except ValueError:
                    pass
        
        for i, (start, end) in enumerate(scene_boundaries):
            # Find closest keyframe for this scene
            scene_keyframe_ts = None
            for ts in scene_keyframe_timestamps:
                if start <= ts <= end:
                    scene_keyframe_ts = ts
                    break
            
            # Find closest keyframe if none in scene
            if scene_keyframe_ts is None:
                closest_ts = min(
                    timestamp_to_keyframe.keys(),
                    key=lambda ts: abs(ts - (start + end) / 2),
                    default=None
                )
                scene_keyframe_ts = closest_ts
            
            # Get keyframe ID and description
            keyframe_id = ""
            keyframe_embedding = None
            if scene_keyframe_ts and scene_keyframe_ts in timestamp_to_keyframe:
                kf_id = timestamp_to_keyframe[scene_keyframe_ts]
                keyframe_id = kf_id
                description = descriptions.get(kf_id, "")
                
                # Get embedding if available
                if self._image_processor:
                    try:
                        b64_image = keyframes.get(kf_id, "")
                        if b64_image:
                            image_bytes = base64.b64decode(b64_image)
                            result = self._image_processor.process(
                                image_bytes,
                                {"enable_embedding": True, "enable_caption": False}
                            )
                            if result.embedding:
                                keyframe_embedding = result.embedding
                    except Exception:
                        pass
            else:
                description = ""
            
            scenes.append(VideoScene(
                scene_index=i,
                start_time=start,
                end_time=end,
                keyframe_id=keyframe_id,
                keyframe_timestamp=scene_keyframe_ts or start,
                description=description or f"Scene {i + 1}",
                embedding=keyframe_embedding,
            ))
        
        return scenes
    
    def clear_cache(self) -> None:
        """
        Clear all sub-processor caches.
        """
        if self._image_processor:
            self._image_processor.clear_cache()
        if self._audio_processor:
            self._audio_processor.clear_cache()
        
        logger.info("Video processor cache cleared")
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics and configuration.
        
        Returns:
            Dictionary containing processor status and capabilities
        """
        return {
            "supported_formats": list(self.SUPPORTED_FORMATS),
            "min_scene_duration": self.MIN_SCENE_DURATION,
            "keyframe_interval": self.KEYFRAME_INTERVAL,
            "scene_change_threshold": self.SCENE_CHANGE_THRESHOLD,
            "max_scenes": self.max_scenes,
            "scene_detection_enabled": self.enable_scene_detection,
            "keyframes_enabled": self.enable_keyframes,
            "audio_processing_enabled": self.enable_audio,
            "scene_detection_method": self.scene_detection_method,
        }


# Import ImageProcessor for type hints
from .image_processor import ImageProcessor
from .audio_processor import AudioProcessor
