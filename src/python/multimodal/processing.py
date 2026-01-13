"""
KERNELIZE Platform - Multimodal Processing Engine
==================================================

This module implements the main multimodal processing orchestrator for the
KERNELIZE Platform. It provides a unified interface for processing images,
audio, and video content, with support for cross-modal semantic linking.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import base64
import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Supported content types for multimodal processing"""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class MultimodalRequest(BaseModel):
    """
    Request model for multimodal content processing.
    
    Supports single content type processing with configurable options
    for each modality.
    """
    content_type: ContentType = Field(..., description="Type of content to process")
    content: str = Field(..., description="Base64-encoded content or URL")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")


class MultimodalResponse(BaseModel):
    """
    Response model for multimodal content processing.
    
    Returns unified analysis results regardless of input type.
    """
    content_type: str
    kernel_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    processing_time_ms: int
    raw_result: Dict[str, Any]
    cross_modal_links: Dict[str, Any] = Field(default_factory=dict)


class CrossModalLink(BaseModel):
    """
    Cross-modal semantic linking result.
    
    Connects related information across different content modalities.
    """
    source_type: str
    target_type: str
    confidence: float
    linked_elements: List[Dict[str, Any]] = Field(default_factory=list)
    semantic_overlap_score: float = 0.0


class MultiModalEngine:
    """
    Multimodal Processing Engine
    
    Main orchestrator for processing images, audio, and video content.
    Provides lazy loading of sub-processors, unified API, and
    cross-modal semantic linking capabilities.
    """
    
    MAX_FILE_SIZE = {
        "image": 10 * 1024 * 1024,  # 10 MB
        "audio": 50 * 1024 * 1024,  # 50 MB
        "video": 500 * 1024 * 1024,  # 500 MB
    }
    
    def __init__(self):
        """
        Initialize the multimodal engine with lazy-loading processors.
        """
        self._image_processor = None
        self._audio_processor = None
        self._video_processor = None
        
        logger.info("MultiModalEngine initialized")
    
    def process(
        self,
        request: MultimodalRequest,
    ) -> MultimodalResponse:
        """
        Process multimodal content according to the specified type.
        
        Args:
            request: MultimodalRequest containing content and options
            
        Returns:
            MultimodalResponse with analysis results and embeddings
        """
        import time
        start_time = time.time()
        
        content_type = request.content_type
        content = request.content
        options = request.options
        
        # Validate content size
        try:
            content_bytes = self._decode_content(content)
            max_size = self.MAX_FILE_SIZE.get(content_type.value, 10 * 1024 * 1024)
            
            if len(content_bytes) > max_size:
                raise ValueError(
                    f"Content exceeds maximum size for {content_type.value}: "
                    f"{len(content_bytes)} > {max_size} bytes"
                )
        except Exception as e:
            raise ValueError(f"Invalid content: {str(e)}")
        
        # Route to appropriate processor
        raw_result = {}
        embedding = None
        kernel_id = None
        
        try:
            if content_type == ContentType.IMAGE:
                raw_result, embedding = self._process_image(content_bytes, options)
                
            elif content_type == ContentType.AUDIO:
                raw_result, embedding = self._process_audio(content_bytes, options)
                
            elif content_type == ContentType.VIDEO:
                raw_result, embedding = self._process_video(content_bytes, options)
            
            # Generate kernel ID
            import hashlib
            content_hash = hashlib.md5(content_bytes).hexdigest()[:12]
            kernel_id = f"kz_{content_type.value}_{content_hash}_{int(time.time())}"
            
        except Exception as e:
            logger.exception(f"Multimodal processing failed: {e}")
            raise ValueError(f"Failed to process {content_type.value}: {str(e)}")
        
        # Perform cross-modal linking if applicable
        cross_modal_links = {}
        
        # Generate cross-modal links
        cross_modal_links = self._generate_cross_modal_links(raw_result, content_type.value)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return MultimodalResponse(
            content_type=content_type.value,
            kernel_id=kernel_id,
            embedding=embedding,
            processing_time_ms=processing_time_ms,
            raw_result=raw_result,
            cross_modal_links=cross_modal_links,
        )
    
    def _decode_content(self, content: str) -> bytes:
        """
        Decode content from base64 or URL.
        
        Args:
            content: Base64-encoded string or URL
            
        Returns:
            Raw bytes content
        """
        # Check if URL
        if content.startswith(("http://", "https://")):
            import requests
            
            response = requests.get(content, timeout=30)
            response.raise_for_status()
            return response.content
        
        # Assume base64
        try:
            return base64.b64decode(content)
        except Exception as e:
            raise ValueError(f"Invalid content encoding: {str(e)}")
    
    def _process_image(
        self,
        content_bytes: bytes,
        options: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[List[float]]]:
        """
        Process image content.
        
        Args:
            content_bytes: Raw image bytes
            options: Processing options
            
        Returns:
            Tuple of (analysis_result, embedding)
        """
        if self._image_processor is None:
            self._image_processor = ImageProcessor(
                enable_ocr=options.get("enable_ocr", True),
                enable_caption=options.get("enable_caption", True),
                enable_embedding=options.get("enable_embedding", True),
            )
        
        result = self._image_processor.process(content_bytes, options)
        
        return result.to_dict(), result.embedding
    
    def _process_audio(
        self,
        content_bytes: bytes,
        options: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[List[float]]]:
        """
        Process audio content.
        
        Args:
            content_bytes: Raw audio bytes
            options: Processing options
            
        Returns:
            Tuple of (analysis_result, embedding)
        """
        if self._audio_processor is None:
            self._audio_processor = AudioProcessor(
                enable_transcription=options.get("enable_transcription", True),
                enable_summary=options.get("enable_summary", True),
                enable_features=options.get("enable_features", True),
                whisper_model=options.get("whisper_model", "base"),
            )
        
        result = self._audio_processor.process(content_bytes, options)
        
        return result.to_dict(), None  # Audio doesn't have standard embedding
    
    def _process_video(
        self,
        content_bytes: bytes,
        options: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[List[float]]]:
        """
        Process video content.
        
        Args:
            content_bytes: Raw video bytes
            options: Processing options
            
        Returns:
            Tuple of (analysis_result, embedding)
        """
        if self._video_processor is None:
            self._video_processor = VideoProcessor(
                enable_scene_detection=options.get("enable_scene_detection", True),
                enable_keyframes=options.get("enable_keyframes", True),
                enable_audio=options.get("enable_audio", True),
            )
        
        result = self._video_processor.process(content_bytes, options)
        
        # Extract first keyframe embedding if available
        embedding = None
        if result.scenes and result.scenes[0].embedding:
            embedding = result.scenes[0].embedding
        
        return result.to_dict(), embedding
    
    def _generate_cross_modal_links(
        self,
        raw_result: Dict[str, Any],
        content_type: str,
    ) -> Dict[str, Any]:
        """
        Generate semantic links across different modalities.
        
        Args:
            raw_result: Processing result from the processor
            content_type: Type of content processed
            
        Returns:
            Dictionary containing cross-modal links
        """
        links = {}
        
        if content_type == "video":
            # Link audio transcript with visual scene descriptions
            audio_analysis = raw_result.get("audio_analysis", {})
            transcript = audio_analysis.get("transcript", "")
            scenes = raw_result.get("scenes", [])
            
            if transcript and scenes:
                # Extract keywords from transcript
                transcript_keywords = self._extract_keywords(transcript)
                
                # Link with scene descriptions
                linked_scenes = []
                for scene in scenes:
                    scene_desc = scene.get("description", "")
                    scene_keywords = self._extract_keywords(scene_desc)
                    
                    # Calculate overlap
                    overlap = set(transcript_keywords) & set(scene_keywords)
                    
                    if overlap:
                        linked_scenes.append({
                            "scene_index": scene.get("scene_index"),
                            "start_time": scene.get("start_time"),
                            "linked_keywords": list(overlap),
                            "confidence": len(overlap) / max(len(transcript_keywords), 1),
                        })
                
                if linked_scenes:
                    links["audio_visual"] = {
                        "type": "transcript_scene_linking",
                        "linked_scenes": linked_scenes,
                        "transcript_keywords": transcript_keywords[:10],
                    }
        
        elif content_type == "image":
            # Link OCR text with image caption
            ocr_text = raw_result.get("ocr_text", "")
            caption = raw_result.get("caption", "")
            
            if ocr_text and caption:
                ocr_keywords = self._extract_keywords(ocr_text)
                caption_keywords = self._extract_keywords(caption)
                overlap = set(ocr_keywords) & set(caption_keywords)
                
                links["text_visual"] = {
                    "type": "ocr_caption_linking",
                    "overlap_keywords": list(overlap),
                    "overlap_score": len(overlap) / max(len(ocr_keywords), 1),
                }
        
        elif content_type == "audio":
            # Link transcript with audio features
            transcript = raw_result.get("transcript", "")
            features = raw_result.get("audio_features", {})
            
            if transcript:
                links["transcript_features"] = {
                    "type": "content_feature_linking",
                    "word_count": len(transcript.split()),
                    "speech_rate": features.get("speech_rate_estimate", 0),
                    "duration": features.get("duration", 0),
                }
        
        return links
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for semantic linking.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted keywords
        """
        import re
        from collections import Counter
        
        if not text:
            return []
        
        # Simple stopwords
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "to", "of", "in", "for", "on", "with", "at",
            "by", "from", "up", "about", "into", "through", "during",
            "and", "or", "but", "if", "then", "this", "that", "these",
            "those", "it", "its", "with", "without", "i", "you", "he",
            "she", "we", "they", "what", "which", "who", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "also",
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stopwords and count
        filtered = [w for w in words if w not in stopwords]
        word_counts = Counter(filtered)
        
        # Return most common words
        return [word for word, _ in word_counts.most_common(20)]
    
    def clear_all_caches(self) -> None:
        """
        Clear all processor caches to free memory.
        """
        if self._image_processor:
            self._image_processor.clear_cache()
        if self._audio_processor:
            self._audio_processor.clear_cache()
        if self._video_processor:
            self._video_processor.clear_cache()
        
        logger.info("All multimodal processor caches cleared")
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics and processor status.
        
        Returns:
            Dictionary containing engine status and capabilities
        """
        return {
            "max_file_sizes": {
                k: f"{v / (1024*1024):.1f} MB"
                for k, v in self.MAX_FILE_SIZE.items()
            },
            "supported_types": [ct.value for ct in ContentType],
            "processors": {
                "image": (
                    self._image_processor.get_processor_stats()
                    if self._image_processor
                    else {"loaded": False}
                ),
                "audio": (
                    self._audio_processor.get_processor_stats()
                    if self._audio_processor
                    else {"loaded": False}
                ),
                "video": (
                    self._video_processor.get_processor_stats()
                    if self._video_processor
                    else {"loaded": False}
                ),
            },
        }


# Import sub-processors
from .image_processor import ImageProcessor
from .audio_processor import AudioProcessor
from .video_processor import VideoProcessor


# Create global engine instance
multimodal_engine = MultiModalEngine()


# Convenience functions
def process_image(content: bytes, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process image content"""
    request = MultimodalRequest(
        content_type=ContentType.IMAGE,
        content=base64.b64encode(content).decode("utf-8"),
        options=options or {},
    )
    return multimodal_engine.process(request)


def process_audio(content: bytes, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process audio content"""
    request = MultimodalRequest(
        content_type=ContentType.AUDIO,
        content=base64.b64encode(content).decode("utf-8"),
        options=options or {},
    )
    return multimodal_engine.process(request)


def process_video(content: bytes, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process video content"""
    request = MultimodalRequest(
        content_type=ContentType.VIDEO,
        content=base64.b64encode(content).decode("utf-8"),
        options=options or {},
    )
    return multimodal_engine.process(request)


# Convenience function for main.py import
multimodal_processor = multimodal_engine


# Dataclasses for results (matching main.py expectations)
@dataclass
class MultimodalResult:
    """Result dataclass for unified multimodal processing"""
    kernel_id: str
    media_type: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    extracted_content: Optional[str]
    embedding: Optional[List[float]]
    processing_time_ms: int
