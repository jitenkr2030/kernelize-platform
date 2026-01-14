"""
KERNELIZE Platform - Large Document Processing Optimization
============================================================

Efficient handling of enterprise-scale documents.
Implements streaming processing, context-preserving chunking,
progress tracking, and memory optimization.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple
from threading import RLock
from pathlib import Path
import asyncio
import tempfile
import mmap
import re

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Document chunking strategies"""
    FIXED_SIZE = "fixed_size"  # Fixed number of tokens/characters
    SENTENCE = "sentence"  # Split by sentences
    PARAGRAPH = "paragraph"  # Split by paragraphs
    SEMANTIC = "semantic"  # Split by semantic boundaries
    OVERLAP = "overlap"  # Fixed size with overlap
    RECURSIVE = "recursive"  # Hierarchical splitting


class ProcessingStatus(Enum):
    """Processing job status"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ChunkConfig:
    """Configuration for document chunking"""
    strategy: str = ChunkingStrategy.SEMANTIC.value
    
    # Size limits
    max_chunk_size: int = 2000  # tokens
    min_chunk_size: int = 100
    chunk_overlap: int = 200
    
    # Token estimation (approximate)
    tokens_per_character: float = 0.25
    
    # Semantic boundaries
    paragraph_separator: str = "\n\n"
    sentence_separators: List[str] = field(default_factory=lambda: [
        ".", "!", "?", "\n"
    ])
    
    # Context preservation
    preserve_headers: bool = True
    preserve_formatting: bool = True
    include_metadata: bool = True


@dataclass
class DocumentChunk:
    """A chunk of a document"""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    chunk_index: int = 0
    
    # Content
    content: str = ""
    start_offset: int = 0
    end_offset: int = 0
    
    # Context
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    
    # Metadata
    token_count: int = 0
    char_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing info
    embedding_generated: bool = False
    embedding_vector: Optional[List[float]] = None


@dataclass
class ProcessingJob:
    """Document processing job"""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    document_path: str = ""
    
    status: str = ProcessingStatus.PENDING.value
    priority: int = 100  # Lower = higher priority
    
    # Progress
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    progress_percent: float = 0.0
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_completion: Optional[str] = None
    
    # Configuration
    chunk_config: Dict[str, Any] = field(default_factory=dict)
    compression_quality: str = "balanced"
    
    # Results
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    
    # Webhook
    webhook_url: Optional[str] = None
    webhook_retries: int = 3
    
    # User
    user_id: Optional[str] = None
    organization_id: Optional[str] = None


@dataclass
class ProcessingProgress:
    """Progress update for a processing job"""
    job_id: str = ""
    status: str = ""
    progress_percent: float = 0.0
    processed_items: int = 0
    total_items: int = 0
    
    current_operation: str = ""
    items_per_second: float = 0.0
    estimated_remaining_seconds: float = 0.0
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingDocumentReader:
    """Streams large documents without loading into memory"""
    
    def __init__(self, chunk_size: int = 8192):
        """
        Initialize streaming reader
        
        Args:
            chunk_size: Size of chunks to read
        """
        self.chunk_size = chunk_size
    
    async def stream_file(
        self,
        file_path: str,
        encoding: str = "utf-8"
    ) -> AsyncIterator[str]:
        """
        Stream file contents asynchronously
        
        Args:
            file_path: Path to file
            encoding: File encoding
            
        Yields:
            Chunks of file content
        """
        file_size = os.path.getsize(file_path)
        bytes_read = 0
        
        with open(file_path, 'r', encoding=encoding) as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                
                bytes_read += len(chunk)
                progress = (bytes_read / file_size) * 100 if file_size > 0 else 0
                
                yield chunk, progress
        
        yield "", 100.0
    
    async def stream_with_overlap(
        self,
        content: str,
        chunk_size: int,
        overlap: int
    ) -> AsyncIterator[Tuple[str, int, int]]:
        """
        Stream content with overlap between chunks
        
        Args:
            content: Full content
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Yields:
            (chunk_content, chunk_start, chunk_end)
        """
        start = 0
        content_length = len(content)
        
        while start < content_length:
            end = min(start + chunk_size, content_length)
            
            # Try to break at sentence boundary if possible
            if end < content_length:
                # Look for sentence endings in the remaining content
                for sep in ['. ', '! ', '? ', '\n']:
                    boundary = content[end:end+50].find(sep)
                    if boundary > 0 and boundary < 30:
                        end += boundary + len(sep)
                        break
            
            chunk = content[start:end]
            yield chunk, start, end
            
            # Move forward with overlap
            start = end - overlap
            
            if start >= content_length:
                break


class DocumentChunker:
    """
    Document chunking with context preservation
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize document chunker
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkConfig()
    
    def chunk_document(
        self,
        content: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Split document into chunks
        
        Args:
            content: Full document content
            document_id: Document identifier
            metadata: Document metadata
            
        Returns:
            List of document chunks
        """
        strategy = self.config.strategy
        
        if strategy == ChunkingStrategy.FIXED_SIZE.value:
            chunks = self._chunk_fixed_size(content)
        elif strategy == ChunkingStrategy.SENTENCE.value:
            chunks = self._chunk_by_sentence(content)
        elif strategy == ChunkingStrategy.PARAGRAPH.value:
            chunks = self._chunk_by_paragraph(content)
        elif strategy == ChunkingStrategy.SEMANTIC.value:
            chunks = self._chunk_semantic(content)
        elif strategy == ChunkingStrategy.OVERLAP.value:
            chunks = self._chunk_with_overlap(content)
        else:
            chunks = self._chunk_recursive(content)
        
        # Add metadata and calculate token counts
        for i, chunk in enumerate(chunks):
            chunk.document_id = document_id
            chunk.chunk_index = i
            chunk.token_count = self._estimate_tokens(chunk.content)
            chunk.char_count = len(chunk.content)
            chunk.metadata = metadata or {}
            
            # Link chunks
            if i > 0:
                chunk.previous_chunk_id = chunks[i-1].chunk_id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i+1].chunk_id
        
        return chunks
    
    def _chunk_fixed_size(self, content: str) -> List[DocumentChunk]:
        """Split by fixed size"""
        max_size = int(self.config.max_chunk_size / self.config.tokens_per_character)
        chunks = []
        
        for i in range(0, len(content), max_size):
            chunk = DocumentChunk(
                content=content[i:i + max_size],
                start_offset=i,
                end_offset=min(i + max_size, len(content))
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sentence(self, content: str) -> List[DocumentChunk]:
        """Split by sentences"""
        # Split by common sentence endings
        separators = self.config.sentence_separators
        pattern = '|'.join(re.escape(s) for s in separators)
        
        sentences = re.split(f'({pattern})', content)
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.config.max_chunk_size / self.config.tokens_per_character:
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        start_offset=content.find(current_chunk.strip()),
                        end_offset=content.find(current_chunk.strip()) + len(current_chunk.strip())
                    ))
                current_chunk = sentence
                current_size = sentence_size
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                start_offset=content.find(current_chunk.strip()),
                end_offset=content.find(current_chunk.strip()) + len(current_chunk.strip())
            ))
        
        return chunks
    
    def _chunk_by_paragraph(self, content: str) -> List[DocumentChunk]:
        """Split by paragraphs"""
        paragraphs = content.split(self.config.paragraph_separator)
        chunks = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Further split if too large
            max_size = int(self.config.max_chunk_size / self.config.tokens_per_character)
            
            if len(paragraph) <= max_size:
                chunks.append(DocumentChunk(
                    content=paragraph,
                    start_offset=content.find(paragraph),
                    end_offset=content.find(paragraph) + len(paragraph)
                ))
            else:
                # Split large paragraph
                sub_chunks = self._chunk_fixed_size(paragraph)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _chunk_semantic(self, content: str) -> List[DocumentChunk]:
        """
        Split by semantic boundaries
        
        Attempts to split at logical boundaries like:
        - Section headers
        - Paragraphs
        - Sentences
        """
        chunks = []
        
        # Try to identify section headers (lines that are short and end with colon or are all caps)
        lines = content.split('\n')
        current_section = []
        section_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if this is a header
            is_header = (
                len(line) < 100 and
                (line.endswith(':') or line.isupper()) and
                len(line) > 3
            )
            
            if is_header and current_section:
                # Save previous section
                section_content = '\n'.join(current_section)
                chunks.append(DocumentChunk(
                    content=section_content,
                    start_offset=section_start,
                    end_offset=section_start + len(section_content)
                ))
                current_section = []
                section_start = content.find(line)
            
            current_section.append(line)
        
        # Don't forget last section
        if current_section:
            section_content = '\n'.join(current_section)
            chunks.append(DocumentChunk(
                content=section_content,
                start_offset=section_start,
                end_offset=section_start + len(section_content)
            ))
        
        # If no headers found, fall back to paragraph splitting
        if not chunks:
            chunks = self._chunk_by_paragraph(content)
        
        return chunks
    
    def _chunk_with_overlap(self, content: str) -> List[DocumentChunk]:
        """Split with overlap between chunks"""
        max_size = int(self.config.max_chunk_size / self.config.tokens_per_character)
        overlap = self.config.chunk_overlap
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + max_size, len(content))
            
            chunk = DocumentChunk(
                content=content[start:end],
                start_offset=start,
                end_offset=end,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
            
            start = end - overlap
            chunk_index += 1
        
        return chunks
    
    def _chunk_recursive(self, content: str) -> List[DocumentChunk]:
        """
        Hierarchical splitting
        
        Try progressively smaller separators until chunks are small enough
        """
        # First try paragraphs
        chunks = self._chunk_by_paragraph(content)
        
        # Check if chunks are too large
        max_size = int(self.config.max_chunk_size / self.config.tokens_per_character)
        
        # If any chunk is too large, split further
        result_chunks = []
        for chunk in chunks:
            if len(chunk.content) <= max_size:
                result_chunks.append(chunk)
            else:
                # Split by sentences
                sentence_chunks = self._chunk_by_sentence(chunk.content)
                result_chunks.extend(sentence_chunks)
        
        return result_chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimate: 4 characters per token on average
        return max(1, int(len(text) * self.config.tokens_per_character))


class ProcessingJobManager:
    """
    Manages document processing jobs
    
    Handles job queuing, progress tracking, and webhook notifications.
    """
    
    def __init__(self, storage_path: str = "data/processing"):
        """
        Initialize job manager
        
        Args:
            storage_path: Path for job storage
        """
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        self._jobs: Dict[str, ProcessingJob] = {}
        self._job_queue: List[ProcessingJob] = []
        self._lock = RLock()
        
        # Worker configuration
        self._max_concurrent_jobs = 4
        self._running_jobs: Dict[str, asyncio.Task] = {}
        
        # Callbacks
        self._progress_callbacks: List[Callable[[ProcessingProgress], None]] = []
        self._completion_callbacks: List[Callable[[ProcessingJob], None]] = []
    
    def create_job(
        self,
        document_id: str,
        document_path: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        chunk_config: Optional[Dict[str, Any]] = None,
        compression_quality: str = "balanced",
        webhook_url: Optional[str] = None,
        priority: int = 100
    ) -> ProcessingJob:
        """
        Create a new processing job
        
        Args:
            document_id: Document identifier
            document_path: Path to document
            user_id: User who initiated
            organization_id: User's organization
            chunk_config: Chunking configuration
            compression_quality: Quality setting
            webhook_url: URL to notify on completion
            priority: Job priority
            
        Returns:
            Created job
        """
        job = ProcessingJob(
            document_id=document_id,
            document_path=document_path,
            priority=priority,
            chunk_config=chunk_config or {},
            compression_quality=compression_quality,
            webhook_url=webhook_url,
            user_id=user_id,
            organization_id=organization_id
        )
        
        with self._lock:
            self._jobs[job.job_id] = job
            self._job_queue.append(job)
            self._job_queue.sort(key=lambda j: j.priority)
        
        # Try to start job
        self._try_start_jobs()
        
        return job
    
    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job by ID"""
        return self._jobs.get(job_id)
    
    def update_progress(
        self,
        job_id: str,
        processed_items: int,
        total_items: int,
        current_operation: str = ""
    ):
        """
        Update job progress
        
        Args:
            job_id: Job to update
            processed_items: Items processed so far
            total_items: Total items to process
            current_operation: What's being done now
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            
            job.processed_items = processed_items
            job.total_items = total_items
            job.progress_percent = (
                (processed_items / total_items * 100) if total_items > 0 else 0
            )
            
            # Estimate completion time
            elapsed = (
                datetime.now(timezone.utc) - 
                datetime.fromisoformat(job.started_at)
            ).total_seconds() if job.started_at else 0
            
            if processed_items > 0 and elapsed > 0:
                rate = processed_items / elapsed
                remaining = (total_items - processed_items) / rate if rate > 0 else 0
                job.estimated_completion = (
                    datetime.now(timezone.utc) + 
                    timezone.timedelta(seconds=remaining)
                ).isoformat()
        
        # Create progress update
        progress = ProcessingProgress(
            job_id=job_id,
            status=job.status,
            progress_percent=job.progress_percent,
            processed_items=processed_items,
            total_items=total_items,
            current_operation=current_operation
        )
        
        # Notify callbacks
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
    
    def complete_job(
        self,
        job_id: str,
        output_path: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Mark job as completed
        
        Args:
            job_id: Job to complete
            output_path: Path to output
            error: Error message if failed
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.status = ProcessingStatus.FAILED.value if error else ProcessingStatus.COMPLETED.value
            job.error_message = error
            job.output_path = output_path
            job.progress_percent = 100.0 if not error else job.progress_percent
            
            # Remove from running jobs
            if job_id in self._running_jobs:
                del self._running_jobs[job_id]
        
        # Notify completion callbacks
        for callback in self._completion_callbacks:
            try:
                callback(job)
            except Exception as e:
                logger.error(f"Completion callback failed: {e}")
        
        # Send webhook notification
        if job.webhook_url:
            self._send_webhook(job)
        
        # Try to start next job
        self._try_start_jobs()
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job
        
        Args:
            job_id: Job to cancel
            
        Returns:
            True if cancelled
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            
            job.status = ProcessingStatus.CANCELLED.value
            job.completed_at = datetime.now(timezone.utc).isoformat()
            
            # Cancel running task
            if job_id in self._running_jobs:
                self._running_jobs[job_id].cancel()
                del self._running_jobs[job_id]
            
            # Remove from queue
            self._job_queue = [j for j in self._job_queue if j.job_id != job_id]
        
        return True
    
    def _try_start_jobs(self):
        """Start jobs if there's capacity"""
        with self._lock:
            while (
                len(self._running_jobs) < self._max_concurrent_jobs and
                self._job_queue
            ):
                job = self._job_queue.pop(0)
                
                if job.status == ProcessingStatus.CANCELLED.value:
                    continue
                
                job.status = ProcessingStatus.PROCESSING.value
                job.started_at = datetime.now(timezone.utc).isoformat()
                
                # Start async task
                task = asyncio.create_task(self._run_job(job))
                self._running_jobs[job.job_id] = task
    
    async def _run_job(self, job: ProcessingJob):
        """Run a processing job"""
        try:
            # Process document (simplified)
            self.update_progress(
                job.job_id, 0, 100, "Starting processing"
            )
            
            # Simulate processing steps
            for i in range(1, 101, 10):
                await asyncio.sleep(0.1)
                self.update_progress(
                    job.job_id, i, 100, f"Processing chunk {i//10 + 1}"
                )
            
            job.output_path = f"output/{job.document_id}.compressed"
            
            self.complete_job(job.job_id, output_path=job.output_path)
            
        except asyncio.CancelledError:
            self.complete_job(job.job_id, error="Job was cancelled")
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            self.complete_job(job.job_id, error=str(e))
    
    def _send_webhook(self, job: ProcessingJob):
        """Send webhook notification"""
        # Would implement actual HTTP request here
        logger.info(f"Would send webhook to {job.webhook_url} for job {job.job_id}")
    
    def add_progress_callback(self, callback: Callable[[ProcessingProgress], None]):
        """Add progress callback"""
        self._progress_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable[[ProcessingJob], None]):
        """Add completion callback"""
        self._completion_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get job manager statistics"""
        with self._lock:
            return {
                'total_jobs': len(self._jobs),
                'queued_jobs': len(self._job_queue),
                'running_jobs': len(self._running_jobs),
                'completed_jobs': len([
                    j for j in self._jobs.values()
                    if j.status == ProcessingStatus.COMPLETED.value
                ]),
                'failed_jobs': len([
                    j for j in self._jobs.values()
                    if j.status == ProcessingStatus.FAILED.value
                ]),
                'max_concurrent': self._max_concurrent_jobs
            }


class MemoryOptimizedProcessor:
    """
    Memory-optimized document processor
    
    Uses streaming and chunking to process large documents
    without loading them entirely into memory.
    """
    
    def __init__(self, max_memory_mb: int = 1024):
        """
        Initialize processor
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._current_usage = 0
        self._lock = RLock()
    
    def estimate_content_size(self, content: str) -> int:
        """Estimate memory size of content"""
        # UTF-8: 1-4 bytes per character, assume 2 on average
        return len(content.encode('utf-8'))
    
    def check_memory_pressure(self) -> bool:
        """Check if memory pressure is high"""
        import psutil
        process = psutil.Process()
        current_memory = process.memory_info().rss
        
        return current_memory > (self.max_memory_bytes * 0.8)
    
    def get_processing_chunk_size(self, content_size: int) -> int:
        """Get appropriate chunk size based on available memory"""
        available = self.max_memory_bytes - self._current_usage
        
        # Use at most 50% of available memory for a single chunk
        max_chunk_size = int(available * 0.5)
        
        return min(content_size, max_chunk_size)
    
    async def process_large_document(
        self,
        file_path: str,
        processor: Callable[[str], Any],
        chunk_size: int = 1024 * 1024  # 1MB chunks
    ) -> AsyncIterator[Tuple[Any, float]]:
        """
        Process large document in streaming fashion
        
        Args:
            file_path: Path to document
            processor: Function to process each chunk
            chunk_size: Size of chunks to read
            
        Yields:
            (result, progress_percent)
        """
        file_size = os.path.getsize(file_path)
        bytes_processed = 0
        
        with open(file_path, 'rb') as f:
            while True:
                # Check memory pressure
                if self.check_memory_pressure():
                    # Wait for memory to be freed
                    await asyncio.sleep(1)
                
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # Process chunk
                result = processor(chunk)
                
                bytes_processed += len(chunk)
                progress = (bytes_processed / file_size * 100) if file_size > 0 else 0
                
                yield result, progress


# Singleton instances
_job_manager: Optional[ProcessingJobManager] = None
_chunker: Optional[DocumentChunker] = None


def get_job_manager() -> ProcessingJobManager:
    """Get job manager singleton"""
    global _job_manager
    
    if _job_manager is None:
        _job_manager = ProcessingJobManager()
    
    return _job_manager


def get_chunker(config: Optional[ChunkConfig] = None) -> DocumentChunker:
    """Get document chunker"""
    global _chunker
    
    if _chunker is None:
        _chunker = DocumentChunker(config)
    
    return _chunker


def init_document_processing(
    storage_path: str = "data/processing",
    max_memory_mb: int = 1024,
    chunk_config: Optional[ChunkConfig] = None
) -> Tuple[ProcessingJobManager, DocumentChunker]:
    """Initialize document processing system"""
    global _job_manager, _chunker
    
    _job_manager = ProcessingJobManager(storage_path)
    _chunker = DocumentChunker(chunk_config)
    
    return _job_manager, _chunker
