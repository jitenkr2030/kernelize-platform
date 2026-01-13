"""
KERNELIZE Platform - Image Processor
=====================================

This module implements image analysis capabilities for the KERNELIZE Platform.
It provides OCR text extraction, image captioning using BLIP, and semantic
embeddings using CLIP for downstream search and retrieval tasks.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import io
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """
    Optical Character Recognition Result
    
    Contains extracted text from images along with confidence scores
    and detected language information.
    """
    text: str
    confidence: float
    language: str
    words: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: int = 0


@dataclass
class ImageAnalysisResult:
    """
    Image Analysis Result
    
    Comprehensive analysis of an image including generated captions,
    extracted text, semantic embeddings, and detected tags.
    """
    caption: str
    embedding: Optional[List[float]]
    ocr_result: Optional[OCRResult]
    tags: List[str]
    image_size: Tuple[int, int]
    format: str
    processing_time_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "caption": self.caption,
            "embedding": self.embedding,
            "ocr_text": self.ocr_result.text if self.ocr_result else None,
            "ocr_confidence": self.ocr_result.confidence if self.ocr_result else None,
            "tags": self.tags,
            "image_size": self.image_size,
            "format": self.format,
            "processing_time_ms": self.processing_time_ms,
        }


class ImageProcessor:
    """
    Image Analysis Processor
    
    Handles image preprocessing, OCR extraction, caption generation,
    and embedding creation for the KERNELIZE knowledge compression system.
    Supports JPEG, PNG, and WEBP formats with configurable processing options.
    """
    
    SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
    MAX_IMAGE_SIZE = (2048, 2048)  # Maximum dimension for processing
    CLIP_DIMENSIONS = 512  # Standard CLIP embedding dimension
    
    def __init__(
        self,
        enable_ocr: bool = True,
        enable_caption: bool = True,
        enable_embedding: bool = True,
        ocr_language: str = "eng",
        blip_model_name: str = "Salesforce/blip-image-captioning-base",
        clip_model_name: str = "openai/clip-vit-base-patch32",
    ):
        """
        Initialize the image processor with specified model configurations.
        
        Args:
            enable_ocr: Whether to perform OCR text extraction
            enable_caption: Whether to generate image captions
            enable_embedding: Whether to create CLIP embeddings
            ocr_language: Language for OCR (default: English)
            blip_model_name: HuggingFace model for caption generation
            clip_model_name: HuggingFace model for embeddings
        """
        self.enable_ocr = enable_ocr
        self.enable_caption = enable_caption
        self.enable_embedding = enable_embedding
        self.ocr_language = ocr_language
        
        # Model placeholders (lazy loaded)
        self._blip_model = None
        self._clip_model = None
        self._blip_processor = None
        self._clip_processor = None
        self._blip_model_name = blip_model_name
        self._clip_model_name = clip_model_name
        
        logger.info(f"ImageProcessor initialized with OCR={enable_ocr}, "
                   f"Caption={enable_caption}, Embedding={enable_embedding}")
    
    def process(
        self,
        image_data: bytes,
        options: Optional[Dict[str, Any]] = None,
    ) -> ImageAnalysisResult:
        """
        Process an image and return comprehensive analysis results.
        
        Args:
            image_data: Raw image bytes
            options: Optional processing configuration
            
        Returns:
            ImageAnalysisResult containing all analysis outputs
        """
        import time
        start_time = time.time()
        
        options = options or {}
        
        # Parse options
        enable_ocr = options.get("enable_ocr", self.enable_ocr)
        enable_caption = options.get("enable_caption", self.enable_caption)
        enable_embedding = options.get("enable_embedding", self.enable_embedding)
        ocr_language = options.get("ocr_language", self.ocr_language)
        
        try:
            # Load and validate image
            image = self._load_image(image_data)
            image_format = image.format or "UNKNOWN"
            
            # Preprocess image
            image = self._preprocess_image(image)
            
            # Generate caption
            caption = ""
            if enable_caption:
                caption = self._generate_caption(image)
            
            # Extract OCR text
            ocr_result = None
            if enable_ocr:
                ocr_result = self._extract_text(image, language=ocr_language)
            
            # Generate CLIP embedding
            embedding = None
            if enable_embedding:
                embedding = self._get_embedding(image)
            
            # Extract tags from caption and OCR
            tags = self._extract_tags(caption, ocr_result)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = ImageAnalysisResult(
                caption=caption or "Unable to generate caption",
                embedding=embedding,
                ocr_result=ocr_result,
                tags=tags,
                image_size=image.size,
                format=image_format,
                processing_time_ms=processing_time_ms,
            )
            
            logger.info(f"Image processed in {processing_time_ms}ms: "
                       f"caption={bool(caption)}, ocr={ocr_result is not None}")
            
            return result
            
        except Exception as e:
            logger.exception(f"Image processing failed: {e}")
            raise ValueError(f"Failed to process image: {str(e)}")
    
    def _load_image(self, image_data: bytes) -> Image.Image:
        """
        Load image from raw bytes with format validation.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            PIL Image object
            
        Raises:
            ValueError: If image format is not supported
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB for consistent processing
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            elif image.mode != "RGB":
                image = image.convert("RGB")
            
            # Validate format
            if image.format and image.format.upper() not in self.SUPPORTED_FORMATS:
                logger.warning(f"Unsupported image format: {image.format}")
            
            return image
            
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for model input.
        
        Applies resizing, normalization, and format standardization.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Resize if too large while maintaining aspect ratio
        if image.size[0] > self.MAX_IMAGE_SIZE[0] or image.size[1] > self.MAX_IMAGE_SIZE[1]:
            image = ImageOps.contain(image, self.MAX_IMAGE_SIZE)
        
        # Apply slight sharpening for better OCR results
        if self.enable_ocr:
            image = ImageOps.autocontrast(image)
        
        return image
    
    def _generate_caption(self, image: Image.Image) -> str:
        """
        Generate a natural language description of the image using BLIP.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Text caption describing the image
        """
        try:
            # Lazy load BLIP model
            if self._blip_model is None:
                self._load_blip_model()
            
            # Prepare input
            inputs = self._blip_processor(
                images=image,
                return_tensors="pt",
                do_resize=False,  # Already preprocessed
            )
            
            # Generate caption
            with self._blip_model.device:
                outputs = self._blip_model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=5,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            caption = self._blip_processor.decode(outputs[0], skip_special_tokens=True)
            
            logger.debug(f"Generated caption: {caption[:100]}...")
            return caption
            
        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")
            return ""
    
    def _extract_text(
        self,
        image: Image.Image,
        language: str = "eng",
    ) -> OCRResult:
        """
        Extract text from image using OCR.
        
        Args:
            image: Input PIL Image
            language: Language code for OCR
            
        Returns:
            OCRResult containing extracted text and metadata
        """
        import time
        import pytesseract
        
        start_time = time.time()
        
        try:
            # Convert to grayscale for better OCR
            gray_image = image.convert("L")
            
            # Perform OCR
            text = pytesseract.image_to_string(gray_image, lang=language)
            
            # Get confidence scores
            data = pytesseract.image_to_data(
                gray_image,
                lang=language,
                output_type=pytesseract.Output.DICT,
            )
            
            # Calculate average confidence
            confidences = [
                int(conf) for conf in data.get("conf", [])
                if conf != "-1" and conf != ""
            ]
            avg_confidence = (
                sum(confidences) / len(confidences)
                if confidences
                else 0.0
            )
            
            # Extract words with positions
            words = []
            for i, word in enumerate(data.get("text", [])):
                if word.strip():
                    words.append({
                        "text": word,
                        "confidence": data.get("conf", [0])[i],
                        "left": data.get("left", [0])[i],
                        "top": data.get("top", [0])[i],
                        "width": data.get("width", [0])[i],
                        "height": data.get("height", [0])[i],
                    })
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence / 100.0,
                language=language,
                words=words,
                processing_time_ms=processing_time_ms,
            )
            
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract OCR not installed")
            return OCRResult(
                text="",
                confidence=0.0,
                language=language,
                words=[],
                processing_time_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                language=language,
                words=[],
                processing_time_ms=int((time.time() - start_time) * 1000),
            )
    
    def _get_embedding(self, image: Image.Image) -> List[float]:
        """
        Generate semantic embedding vector using CLIP.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Normalized embedding vector
        """
        try:
            # Lazy load CLIP model
            if self._clip_model is None:
                self._load_clip_model()
            
            # Prepare input
            inputs = self._clip_processor(
                images=image,
                return_tensors="pt",
                do_resize=True,
                size={"shortest_edge": 224, "longest_edge": 224},
            )
            
            # Get embedding
            with self._clip_model.device:
                with torch.no_grad():
                    image_features = self._clip_model.get_image_features(**inputs)
            
            # Normalize embedding
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return embedding.squeeze().tolist()
            
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None
    
    def _extract_tags(
        self,
        caption: str,
        ocr_result: Optional[OCRResult],
    ) -> List[str]:
        """
        Extract relevant tags from caption and OCR text.
        
        Args:
            caption: Generated image caption
            ocr_result: OCR extraction result
            
        Returns:
            List of extracted tags
        """
        import re
        from collections import Counter
        
        # Combine text sources
        texts = [caption] if caption else []
        if ocr_result and ocr_result.text:
            texts.append(ocr_result.text)
        
        # Simple keyword extraction
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "to", "of", "in", "for", "on", "with", "at",
            "by", "from", "up", "about", "into", "through", "during",
            "and", "or", "but", "in", "on", "at", "this", "that", "these",
            "those", "it", "its", "with", "without", "photo", "image",
        }
        
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            all_words.extend([w for w in words if w not in stopwords])
        
        # Get most common words as tags
        word_counts = Counter(all_words)
        tags = [word for word, count in word_counts.most_common(10)]
        
        return tags
    
    def _load_blip_model(self) -> None:
        """
        Load BLIP image captioning model from HuggingFace.
        
        Downloads model if not cached locally.
        """
        try:
            from transformers import BlipProcessor, BlipForImageCaptioning
            
            logger.info(f"Loading BLIP model: {self._blip_model_name}")
            
            self._blip_processor = BlipProcessor.from_pretrained(
                self._blip_model_name,
                cache_dir=os.environ.get("HF_CACHE_DIR", "./cache")
            )
            self._blip_model = BlipForImageCaptioning.from_pretrained(
                self._blip_model_name,
                cache_dir=os.environ.get("HF_CACHE_DIR", "./cache")
            )
            
            self._blip_model.eval()
            
            logger.info("BLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            raise
    
    def _load_clip_model(self) -> None:
        """
        Load CLIP embedding model from HuggingFace.
        
        Downloads model if not cached locally.
        """
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            logger.info(f"Loading CLIP model: {self._clip_model_name}")
            
            self._clip_processor = CLIPProcessor.from_pretrained(
                self._clip_model_name,
                cache_dir=os.environ.get("HF_CACHE_DIR", "./cache")
            )
            self._clip_model = CLIPModel.from_pretrained(
                self._clip_model_name,
                cache_dir=os.environ.get("HF_CACHE_DIR", "./cache")
            )
            
            self._clip_model.eval()
            
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def clear_cache(self) -> None:
        """
        Clear loaded models from memory.
        
        Useful for memory-constrained environments.
        """
        import torch
        
        self._blip_model = None
        self._clip_model = None
        self._blip_processor = None
        self._clip_processor = None
        
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
            "max_image_size": self.MAX_IMAGE_SIZE,
            "clip_embedding_dimensions": self.CLIP_DIMENSIONS,
            "ocr_enabled": self.enable_ocr,
            "caption_enabled": self.enable_caption,
            "embedding_enabled": self.enable_embedding,
            "blip_model_loaded": self._blip_model is not None,
            "clip_model_loaded": self._clip_model is not None,
            "blip_model_name": self._blip_model_name,
            "clip_model_name": self._clip_model_name,
        }


# Lazy import for torch to handle optional dependency
try:
    import torch
except ImportError:
    torch = None
