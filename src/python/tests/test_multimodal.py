"""
KERNELIZE Platform - Multimodal Tests
======================================

This module contains unit tests for the multimodal processing system.
Tests cover image, audio, and video processing functionality.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImageProcessor:
    """Test cases for image processing"""
    
    def test_image_processor_initialization(self):
        """Test image processor can be initialized"""
        from multimodal.image_processor import ImageProcessor
        
        processor = ImageProcessor(
            enable_ocr=False,
            enable_caption=False,
            enable_embedding=False,
        )
        
        assert processor is not None
        assert processor.enable_ocr is False
        assert processor.enable_caption is False
        assert processor.enable_embedding is False
    
    def test_processor_stats(self):
        """Test processor statistics retrieval"""
        from multimodal.image_processor import ImageProcessor
        
        processor = ImageProcessor(
            enable_ocr=True,
            enable_caption=True,
            enable_embedding=True,
        )
        
        stats = processor.get_processor_stats()
        
        assert "supported_formats" in stats
        assert "max_image_size" in stats
        assert "ocr_enabled" in stats
        assert "caption_enabled" in stats
        assert "embedding_enabled" in stats
    
    def test_tag_extraction(self):
        """Test keyword tag extraction from caption and OCR"""
        from multimodal.image_processor import ImageProcessor
        
        processor = ImageProcessor(enable_ocr=False, enable_caption=True)
        
        # Simulate caption and OCR
        caption = "A red car parked on a street"
        ocr_result = None
        
        tags = processor._extract_tags(caption, ocr_result)
        
        assert isinstance(tags, list)
        # Should extract meaningful keywords
        assert len(tags) > 0
    
    def test_image_size_validation(self):
        """Test image size validation"""
        from multimodal.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        
        # Very small image data (simulated)
        small_data = b'\x89PNG\r\n\x1a\n'  # Partial PNG header
        
        try:
            result = processor.process(small_data)
            # Should either succeed or fail gracefully
            assert result is not None or isinstance(result, Exception)
        except Exception as e:
            # Expected for invalid data
            assert "Failed to process image" in str(e) or "Invalid image" in str(e)


class TestAudioProcessor:
    """Test cases for audio processing"""
    
    def test_audio_processor_initialization(self):
        """Test audio processor can be initialized"""
        from multimodal.audio_processor import AudioProcessor
        
        processor = AudioProcessor(
            enable_transcription=False,
            enable_summary=False,
            enable_features=True,
        )
        
        assert processor is not None
        assert processor.enable_features is True
    
    def test_processor_stats(self):
        """Test processor statistics retrieval"""
        from multimodal.audio_processor import AudioProcessor
        
        processor = AudioProcessor(
            enable_transcription=True,
            enable_summary=True,
            enable_features=True,
        )
        
        stats = processor.get_processor_stats()
        
        assert "supported_formats" in stats
        assert "whisper_sample_rate" in stats
        assert "max_audio_length_seconds" in stats
        assert "transcription_enabled" in stats
    
    def test_speech_rate_estimation(self):
        """Test speech rate estimation from audio features"""
        from multimodal.audio_processor import AudioProcessor
        
        processor = AudioProcessor()
        
        # Simulate audio data (silent)
        import numpy as np
        sample_rate = 16000
        duration = 5.0  # 5 seconds
        audio_array = np.zeros(int(sample_rate * duration))
        
        # RMS energy array
        rms_energy = np.zeros(100)
        
        rate = processor._estimate_speech_rate(audio_array, sample_rate, rms_energy)
        
        assert rate >= 0  # Should be non-negative
        assert isinstance(rate, float)
    
    def test_keyword_extraction(self):
        """Test keyword extraction from transcript"""
        from multimodal.processing import MultiModalEngine
        
        engine = MultiModalEngine()
        
        text = """
        The patient presents with chest pain and shortness of breath.
        Vital signs show elevated blood pressure and heart rate.
        Medical history includes diabetes and hypertension.
        """
        
        keywords = engine._extract_keywords(text)
        
        assert isinstance(keywords, list)
        assert "patient" in keywords or "pain" in keywords or "blood" in keywords


class TestVideoProcessor:
    """Test cases for video processing"""
    
    def test_video_processor_initialization(self):
        """Test video processor can be initialized"""
        from multimodal.video_processor import VideoProcessor
        
        processor = VideoProcessor(
            enable_scene_detection=True,
            enable_keyframes=True,
            enable_audio=False,
        )
        
        assert processor is not None
    
    def test_processor_stats(self):
        """Test processor statistics retrieval"""
        from multimodal.video_processor import VideoProcessor
        
        processor = VideoProcessor(
            enable_scene_detection=True,
            enable_keyframes=True,
            enable_audio=True,
        )
        
        stats = processor.get_processor_stats()
        
        assert "supported_formats" in stats
        assert "min_scene_duration" in stats
        assert "scene_detection_enabled" in stats
        assert "keyframes_enabled" in stats
    
    def test_scene_building(self):
        """Test scene object building"""
        from multimodal.video_processor import VideoProcessor, VideoScene
        
        processor = VideoProcessor()
        
        # Create test scenes
        scenes = [
            VideoScene(
                scene_index=0,
                start_time=0.0,
                end_time=10.0,
                keyframe_id="kf_1",
                keyframe_timestamp=5.0,
                description="Opening scene",
                embedding=None,
            ),
            VideoScene(
                scene_index=1,
                start_time=10.0,
                end_time=20.0,
                keyframe_id="kf_2",
                keyframe_timestamp=15.0,
                description="Action sequence",
                embedding=None,
            ),
        ]
        
        assert len(scenes) == 2
        assert scenes[0].scene_index == 0
        assert scenes[0].start_time == 0.0


class TestMultimodalEngine:
    """Test cases for multimodal orchestration engine"""
    
    def test_engine_initialization(self):
        """Test multimodal engine can be initialized"""
        from multimodal.processing import MultiModalEngine
        
        engine = MultiModalEngine()
        
        assert engine is not None
    
    def test_content_type_enum(self):
        """Test content type enumeration"""
        from multimodal.processing import ContentType
        
        assert ContentType.IMAGE.value == "image"
        assert ContentType.AUDIO.value == "audio"
        assert ContentType.VIDEO.value == "video"
    
    def test_max_file_sizes(self):
        """Test maximum file size configuration"""
        from multimodal.processing import MultiModalEngine
        
        engine = MultiModalEngine()
        
        max_sizes = engine.MAX_FILE_SIZE
        
        assert "image" in max_sizes
        assert "audio" in max_sizes
        assert "video" in max_sizes
        # Image should be smaller than video
        assert max_sizes["image"] < max_sizes["video"]
    
    def test_engine_stats(self):
        """Test engine statistics"""
        from multimodal.processing import MultiModalEngine
        
        engine = MultiModalEngine()
        
        stats = engine.get_engine_stats()
        
        assert "max_file_sizes" in stats
        assert "supported_types" in stats
        assert "processors" in stats
    
    def test_keyword_extraction_edge_cases(self):
        """Test keyword extraction with edge case inputs"""
        from multimodal.processing import MultiModalEngine
        
        engine = MultiModalEngine()
        
        # Empty string
        keywords = engine._extract_keywords("")
        assert keywords == []
        
        # Only stopwords
        keywords = engine._extract_keywords("the a and is are was were")
        assert keywords == []
    
    def test_cross_modal_link_generation(self):
        """Test cross-modal link generation"""
        from multimodal.processing import MultiModalEngine, ContentType
        
        engine = MultiModalEngine()
        
        # Video result with audio and visual analysis
        video_result = {
            "audio_analysis": {
                "transcript": "The patient shows improvement in symptoms after medication",
            },
            "scenes": [
                {
                    "scene_index": 0,
                    "start_time": 0,
                    "description": "Medical professional examining patient",
                },
            ],
        }
        
        links = engine._generate_cross_modal_links(video_result, "video")
        
        assert "audio_visual" in links
        assert "transcript_keywords" in links["audio_visual"]


class TestMultimodalRequests:
    """Test cases for multimodal request/response models"""
    
    def test_multimodal_request_creation(self):
        """Test multimodal request model"""
        from multimodal.processing import MultimodalRequest, ContentType
        
        request = MultimodalRequest(
            content_type=ContentType.IMAGE,
            content="base64_encoded_data_here",
            options={"enable_ocr": True},
        )
        
        assert request.content_type == ContentType.IMAGE
        assert "base64" in request.content
        assert request.options["enable_ocr"] is True
    
    def test_multimodal_response_creation(self):
        """Test multimodal response model"""
        from multimodal.processing import MultimodalResponse
        
        response = MultimodalResponse(
            content_type="image",
            kernel_id="kz_test_123",
            embedding=[0.1, 0.2, 0.3],
            processing_time_ms=150,
            raw_result={"caption": "Test image"},
            cross_modal_links={},
        )
        
        assert response.content_type == "image"
        assert response.kernel_id.startswith("kz_")
        assert response.processing_time_ms == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
