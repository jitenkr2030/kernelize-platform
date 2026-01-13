# KERNELIZE Multimodal Module
"""Multimodal content processing for images, audio, and video."""

from .processing import (
    MultiModalEngine,
    MultimodalRequest,
    MultimodalResponse,
    CrossModalLink,
    ContentType,
    multimodal_engine,
    process_image,
    process_audio,
    process_video,
)

from .image_processor import (
    ImageProcessor,
    ImageAnalysisResult,
    OCRResult,
)

from .audio_processor import (
    AudioProcessor,
    AudioAnalysisResult,
    AudioSegment,
)

from .video_processor import (
    VideoProcessor,
    VideoAnalysisResult,
    VideoScene,
)

__all__ = [
    # Main engine
    "MultiModalEngine",
    "MultimodalRequest",
    "MultimodalResponse",
    "CrossModalLink",
    "ContentType",
    "multimodal_engine",
    "process_image",
    "process_audio",
    "process_video",
    # Image
    "ImageProcessor",
    "ImageAnalysisResult",
    "OCRResult",
    # Audio
    "AudioProcessor",
    "AudioAnalysisResult",
    "AudioSegment",
    # Video
    "VideoProcessor",
    "VideoAnalysisResult",
    "VideoScene",
]
