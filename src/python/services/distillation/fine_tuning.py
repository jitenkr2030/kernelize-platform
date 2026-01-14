#!/usr/bin/env python3
"""
LLM Fine-Tuning Pipeline
========================

Enables kernels to be used for model training. Implements dataset generation,
data quality filtering, fine-tuning job management, and model versioning.

Author: MiniMax Agent
"""

import json
import uuid
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from abc import ABC, abstractmethod
import logging
import tempfile
import shutil
import subprocess
import os

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a distillation job"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelArchitecture(Enum):
    """Supported model architectures for fine-tuning"""
    LLAMA = "llama"
    MISTRAL = "mistral"
    GPT_NEOX = "gpt_neox"
    FALCON = "falcon"
    CUSTOM = "custom"


class TrainingStage(Enum):
    """Stages of the fine-tuning pipeline"""
    DATASET_GENERATION = "dataset_generation"
    QUALITY_FILTERING = "quality_filtering"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATION = "evaluation"
    EXPORT = "export"


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning jobs"""
    # Model settings
    base_model: str = "meta-llama/Llama-2-7b-hf"
    architecture: ModelArchitecture = ModelArchitecture.LLAMA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Optimization
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # Validation
    validation_split: float = 0.1
    eval_steps: int = 500
    save_steps: int = 500
    
    # Output
    output_dir: str = "./fine_tuned_models"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityGateResult:
    """Result of quality gate validation"""
    passed: bool
    score: float
    checks: Dict[str, Dict[str, Any]]
    rejected_samples: int = 0
    total_samples: int = 0
    message: str = ""


@dataclass
class TrainingMetrics:
    """Metrics from training process"""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    eval_loss: float = 0.0
    learning_rate: float = 0.0
    train_samples_per_second: float = 0.0
    grad_norm: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DistillationJob:
    """Represents a complete fine-tuning/distillation job"""
    job_id: str
    kernel_id: str
    user_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    
    # Configuration
    training_config: TrainingConfig
    dataset_path: Optional[str] = None
    
    # Results
    output_model_path: Optional[str] = None
    metrics: List[TrainingMetrics] = field(default_factory=list)
    quality_gate_result: Optional[QualityGateResult] = None
    
    # Progress tracking
    current_stage: TrainingStage = TrainingStage.DATASET_GENERATION
    progress_percent: float = 0.0
    stage_progress: Dict[str, float] = field(default_factory=dict)
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kernel_id": self.kernel_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "training_config": self.training_config.to_dict(),
            "current_stage": self.current_stage.value,
            "progress_percent": self.progress_percent,
            "output_model_path": self.output_model_path,
            "error_message": self.error_message,
        }


class DatasetGenerator:
    """
    Generates training datasets from knowledge kernels.
    
    Supports multiple output formats for different fine-tuning frameworks.
    """
    
    SUPPORTED_FORMATS = ["jsonl", "csv", "parquet", "huggingface"]
    
    def __init__(self, kernel_storage):
        """
        Initialize dataset generator.
        
        Args:
            kernel_storage: Storage backend for accessing kernel content
        """
        self.kernel_storage = kernel_storage
    
    def generate_dataset(
        self,
        kernel_id: str,
        output_path: str,
        format: str = "jsonl",
        instruction_template: str = None,
        max_samples: int = None,
        shuffle: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a fine-tuning dataset from a kernel.
        
        Args:
            kernel_id: ID of the source kernel
            output_path: Path for output file(s)
            format: Output format
            instruction_template: Template for instruction format
            max_samples: Maximum number of samples to generate
            shuffle: Whether to shuffle samples
            
        Returns:
            (output_path, statistics_dict)
        """
        # Get kernel content
        chunks = self.kernel_storage.get_chunks_by_kernel(kernel_id)
        
        if not chunks:
            raise ValueError(f"Kernel has no content: {kernel_id}")
        
        # Apply limit
        if max_samples:
            import random
            if shuffle:
                random.shuffle(chunks)
            chunks = chunks[:max_samples]
        
        # Generate samples
        samples = []
        for i, chunk in enumerate(chunks):
            sample = self._create_sample(
                chunk=chunk,
                index=i,
                kernel_id=kernel_id,
                instruction_template=instruction_template,
            )
            samples.append(sample)
        
        # Shuffle if requested
        if shuffle:
            import random
            random.shuffle(samples)
        
        # Write output
        stats = self._write_output(samples, output_path, format, kernel_id)
        
        logger.info(f"Generated dataset from kernel {kernel_id}: {stats['sample_count']} samples")
        return output_path, stats
    
    def _create_sample(
        self,
        chunk: Dict[str, Any],
        index: int,
        kernel_id: str,
        instruction_template: str = None,
    ) -> Dict[str, Any]:
        """Create a single training sample from a chunk"""
        content = chunk.get("text_content", "")
        metadata = chunk.get("metadata", {})
        
        # Default instruction template
        if not instruction_template:
            instruction_template = "Answer questions about the following topic:"
        
        # Create instruction-input-output format
        sample = {
            "id": f"{kernel_id}_{index}",
            "instruction": instruction_template,
            "input": content[:1000],  # Truncate for input field
            "output": content,
            "metadata": {
                "chunk_id": chunk.get("id"),
                "kernel_id": kernel_id,
                "source": metadata.get("source", "kernel"),
                "domain": metadata.get("domain", "general"),
            }
        }
        
        return sample
    
    def _write_output(
        self,
        samples: List[Dict[str, Any]],
        output_path: str,
        format: str,
        kernel_id: str,
    ) -> Dict[str, Any]:
        """Write samples to output file(s)"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(output_path, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
        
        elif format == "csv":
            import csv
            fieldnames = ["id", "instruction", "input", "output"]
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(samples)
        
        elif format == "parquet":
            try:
                import pandas as pd
                df = pd.DataFrame(samples)
                df.to_parquet(output_path, index=False)
            except ImportError:
                raise ImportError("pandas and pyarrow required for parquet format")
        
        elif format == "huggingface":
            # Create HuggingFace dataset structure
            dataset_path = output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Write train split
            train_path = output_path / "train.jsonl"
            with open(train_path, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
            
            # Write dataset metadata
            metadata = {
                "dataset_name": f"kernel_{kernel_id}",
                "num_samples": len(samples),
                "columns": list(samples[0].keys()) if samples else [],
            }
            with open(output_path / "dataset_info.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            dataset_path = str(dataset_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return {
            "sample_count": len(samples),
            "format": format,
            "output_path": str(output_path),
            "total_chars": sum(len(json.dumps(s)) for s in samples),
        }


class QualityGate:
    """
    Validates datasets against quality thresholds before fine-tuning.
    
    Implements multiple quality checks to ensure training data quality.
    """
    
    DEFAULT_THRESHOLDS = {
        "min_sample_length": 50,
        "max_sample_length": 10000,
        "min_avg_word_length": 2.0,
        "max_duplicate_ratio": 0.1,
        "min_unique_samples": 0.8,
        "max_toxicity_score": 0.1,
        "min_readability_score": 30.0,
    }
    
    def __init__(self, thresholds: Dict[str, float] = None):
        """
        Initialize quality gate with configurable thresholds.
        
        Args:
            thresholds: Custom threshold values (uses defaults if not provided)
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
    
    def validate_dataset(self, dataset_path: str) -> QualityGateResult:
        """
        Run all quality checks on a dataset.
        
        Args:
            dataset_path: Path to dataset file
            
        Returns:
            QualityGateResult with validation outcome
        """
        # Load samples
        samples = self._load_samples(dataset_path)
        
        if not samples:
            return QualityGateResult(
                passed=False,
                score=0.0,
                checks={},
                message="Dataset is empty"
            )
        
        checks = {}
        rejected = set()
        total = len(samples)
        
        # Run all checks
        checks["sample_length"] = self._check_sample_length(samples)
        checks["duplicate_detection"] = self._check_duplicates(samples)
        checks["content_quality"] = self._check_content_quality(samples)
        checks["metadata_completeness"] = self._check_metadata(samples)
        
        # Calculate overall score
        check_scores = [c["score"] for c in checks.values()]
        overall_score = sum(check_scores) / len(check_scores) if check_scores else 0.0
        
        # Determine pass/fail
        passed = overall_score >= 0.7  # 70% threshold
        
        return QualityGateResult(
            passed=passed,
            score=overall_score,
            checks=checks,
            rejected_samples=len(rejected),
            total_samples=total,
            message=f"Score: {overall_score:.2%}, Checks: {len(checks)}"
        )
    
    def _load_samples(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load samples from dataset file"""
        samples = []
        path = Path(dataset_path)
        
        if path.suffix == ".jsonl":
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        
        elif path.suffix == ".csv":
            import csv
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(row)
        
        elif path.is_dir():
            # Assume HuggingFace dataset structure
            train_file = path / "train.jsonl"
            if train_file.exists():
                with open(train_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
        
        return samples
    
    def _check_sample_length(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check that samples are within length bounds"""
        valid = 0
        lengths = []
        
        for sample in samples:
            content = sample.get("output", "") or sample.get("content", "")
            length = len(content)
            lengths.append(length)
            
            if (length >= self.thresholds["min_sample_length"] and
                length <= self.thresholds["max_sample_length"]):
                valid += 1
        
        score = valid / len(samples) if samples else 0
        
        return {
            "score": score,
            "valid_count": valid,
            "total_count": len(samples),
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
        }
    
    def _check_duplicates(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for duplicate samples"""
        content_hashes = set()
        duplicates = 0
        
        for sample in samples:
            content = sample.get("output", "") or sample.get("content", "")
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash in content_hashes:
                duplicates += 1
            else:
                content_hashes.add(content_hash)
        
        duplicate_ratio = duplicates / len(samples) if samples else 0
        passed = duplicate_ratio <= self.thresholds["max_duplicate_ratio"]
        
        return {
            "score": 1.0 - duplicate_ratio,
            "duplicate_count": duplicates,
            "duplicate_ratio": duplicate_ratio,
            "passed": passed,
        }
    
    def _check_content_quality(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Basic content quality checks"""
        valid = 0
        
        for sample in samples:
            content = sample.get("output", "") or sample.get("content", "")
            
            # Check for meaningful content
            if (len(content) > 0 and
                " " in content and  # Has words
                not content.isupper()):  # Not all caps
                valid += 1
        
        score = valid / len(samples) if samples else 0
        
        return {
            "score": score,
            "valid_count": valid,
            "total_count": len(samples),
        }
    
    def _check_metadata(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check that samples have complete metadata"""
        valid = 0
        
        for sample in samples:
            if "metadata" in sample and isinstance(sample["metadata"], dict):
                if sample["metadata"].get("kernel_id"):
                    valid += 1
        
        score = valid / len(samples) if samples else 0
        
        return {
            "score": score,
            "valid_count": valid,
            "total_count": len(samples),
        }


class FineTuningJobManager:
    """
    Manages the lifecycle of fine-tuning jobs.
    
    Handles job queuing, execution, monitoring, and result storage.
    """
    
    def __init__(
        self,
        storage_path: str = "./fine_tuning_jobs",
        max_workers: int = 4,
        checkpoint_interval: int = 300,  # 5 minutes
    ):
        """
        Initialize job manager.
        
        Args:
            storage_path: Path for job state persistence
            max_workers: Maximum concurrent jobs
            checkpoint_interval: Seconds between checkpoint saves
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.checkpoint_interval = checkpoint_interval
        
        self._jobs: Dict[str, DistillationJob] = {}
        self._executors: Dict[str, ThreadPoolExecutor] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # Load existing jobs
        self._load_job_states()
    
    def create_job(
        self,
        kernel_id: str,
        user_id: str,
        training_config: TrainingConfig = None,
        metadata: Dict[str, Any] = None,
    ) -> DistillationJob:
        """Create a new fine-tuning job"""
        job_id = str(uuid.uuid4())
        
        job = DistillationJob(
            job_id=job_id,
            kernel_id=kernel_id,
            user_id=user_id,
            status=JobStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            training_config=training_config or TrainingConfig(),
            metadata=metadata or {},
        )
        
        self._jobs[job_id] = job
        self._save_job_state(job)
        
        logger.info(f"Created fine-tuning job: {job_id}")
        return job
    
    def submit_job(self, job_id: str) -> bool:
        """Submit a job for execution"""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        job.status = JobStatus.QUEUED
        job.updated_at = datetime.utcnow()
        
        # Start execution in background
        executor = ThreadPoolExecutor(max_workers=1)
        self._executors[job_id] = executor
        
        executor.submit(self._execute_job, job_id)
        
        logger.info(f"Submitted job for execution: {job_id}")
        return True
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            return False
        
        job.status = JobStatus.CANCELLED
        job.updated_at = datetime.utcnow()
        
        # Shutdown executor
        if job_id in self._executors:
            self._executors[job_id].shutdown(wait=False)
            del self._executors[job_id]
        
        self._save_job_state(job)
        
        logger.info(f"Cancelled job: {job_id}")
        return True
    
    def get_job(self, job_id: str) -> Optional[DistillationJob]:
        """Get job by ID"""
        return self._jobs.get(job_id)
    
    def list_jobs(
        self,
        user_id: str = None,
        status: JobStatus = None,
        limit: int = 50,
    ) -> List[DistillationJob]:
        """List jobs with optional filtering"""
        jobs = list(self._jobs.values())
        
        if user_id:
            jobs = [j for j in jobs if j.user_id == user_id]
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Sort by creation date
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return jobs[:limit]
    
    def add_callback(self, job_id: str, callback: Callable):
        """Add a callback for job completion"""
        if job_id not in self._callbacks:
            self._callbacks[job_id] = []
        self._callbacks[job_id].append(callback)
    
    def _execute_job(self, job_id: str):
        """Execute a fine-tuning job (runs in background)"""
        job = self._jobs.get(job_id)
        if not job:
            return
        
        try:
            job.status = JobStatus.RUNNING
            job.updated_at = datetime.utcnow()
            self._save_job_state(job)
            
            # Stage 1: Dataset Generation
            job.current_stage = TrainingStage.DATASET_GENERATION
            self._run_stage(job, self._generate_dataset)
            
            # Stage 2: Quality Filtering
            job.current_stage = TrainingStage.QUALITY_FILTERING
            self._run_stage(job, self._run_quality_gate)
            
            # Stage 3: Preprocessing
            job.current_stage = TrainingStage.PREPROCESSING
            self._run_stage(job, self._preprocess_data)
            
            # Stage 4: Training (placeholder - would integrate with actual training)
            job.current_stage = TrainingStage.TRAINING
            self._run_stage(job, self._run_training)
            
            # Stage 5: Evaluation
            job.current_stage = TrainingStage.EVALUATION
            self._run_stage(job, self._run_evaluation)
            
            # Stage 6: Export
            job.current_stage = TrainingStage.EXPORT
            self._run_stage(job, self._export_model)
            
            job.status = JobStatus.COMPLETED
            job.progress_percent = 100.0
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            logger.error(f"Job {job_id} failed: {e}")
        
        finally:
            job.updated_at = datetime.utcnow()
            self._save_job_state(job)
            
            # Run callbacks
            if job_id in self._callbacks:
                for callback in self._callbacks[job_id]:
                    try:
                        callback(job)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            
            # Cleanup executor
            if job_id in self._executors:
                del self._executors[job_id]
    
    def _run_stage(self, job: DistillationJob, stage_func: Callable):
        """Run a single stage with progress tracking"""
        stage_name = job.current_stage.value
        logger.info(f"Running stage: {stage_name} for job {job.job_id}")
        
        try:
            stage_func(job)
            job.stage_progress[stage_name] = 100.0
        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            job.stage_progress[stage_name] = 0.0
            raise
    
    def _generate_dataset(self, job: DistillationJob):
        """Generate training dataset from kernel"""
        from .knowledge_pack import KnowledgePackFormat
        
        # Placeholder implementation
        output_dir = Path(job.training_config.output_dir) / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_path = output_dir / "dataset.jsonl"
        
        # In real implementation, would use DatasetGenerator
        job.dataset_path = str(dataset_path)
        job.progress_percent = 20.0
    
    def _run_quality_gate(self, job: DistillationJob):
        """Run quality gate validation"""
        gate = QualityGate()
        result = gate.validate_dataset(job.dataset_path)
        job.quality_gate_result = result
        
        if not result.passed:
            raise ValueError(f"Quality gate failed: {result.message}")
        
        job.progress_percent = 30.0
    
    def _preprocess_data(self, job: DistillationJob):
        """Preprocess data for training"""
        job.progress_percent = 50.0
    
    def _run_training(self, job: DistillationJob):
        """Run the actual training process"""
        # Placeholder for actual training integration
        # Would integrate with HuggingFace Transformers, PEFT, etc.
        
        # Simulate training progress
        for epoch in range(job.training_config.num_epochs):
            for step in range(100):
                metric = TrainingMetrics(
                    epoch=epoch,
                    step=step,
                    loss=1.0 - (epoch * 0.2 + step * 0.01),
                    learning_rate=job.training_config.learning_rate,
                )
                job.metrics.append(metric)
                job.progress_percent = 50.0 + (step / 100) * 30.0
        
        job.progress_percent = 80.0
    
    def _run_evaluation(self, job: DistillationJob):
        """Run evaluation on validation set"""
        job.progress_percent = 90.0
    
    def _export_model(self, job: DistillationJob):
        """Export the fine-tuned model"""
        output_dir = Path(job.training_config.output_dir) / job.job_id
        job.output_model_path = str(output_dir)
        job.progress_percent = 100.0
    
    def _save_job_state(self, job: DistillationJob):
        """Persist job state to storage"""
        state_path = self.storage_path / f"{job.job_id}.json"
        
        with open(state_path, 'w') as f:
            json.dump(job.to_dict(), f, indent=2, default=str)
    
    def _load_job_states(self):
        """Load persisted job states"""
        for state_file in self.storage_path.glob("*.json"):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    job = DistillationJob(
                        job_id=state["job_id"],
                        kernel_id=state["kernel_id"],
                        user_id=state["user_id"],
                        status=JobStatus(state["status"]),
                        created_at=datetime.fromisoformat(state["created_at"]),
                        updated_at=datetime.fromisoformat(state["updated_at"]),
                        training_config=TrainingConfig(**state.get("training_config", {})),
                    )
                    job.current_stage = TrainingStage(state.get("current_stage", "dataset_generation"))
                    job.progress_percent = state.get("progress_percent", 0.0)
                    job.output_model_path = state.get("output_model_path")
                    job.error_message = state.get("error_message")
                    self._jobs[job.job_id] = job
            except Exception as e:
                logger.error(f"Failed to load job state {state_file}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about job manager"""
        by_status = {}
        for job in self._jobs.values():
            status = job.status.value
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            "total_jobs": len(self._jobs),
            "by_status": by_status,
            "active_executors": len(self._executors),
        }


class ModelVersionManager:
    """
    Manages versioning and deployment of fine-tuned models.
    """
    
    def __init__(self, storage_path: str = "./model_versions"):
        """
        Initialize model version manager.
        
        Args:
            storage_path: Base path for model storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._versions: Dict[str, Dict[str, Any]] = {}
        self._load_versions()
    
    def register_model(
        self,
        job_id: str,
        model_path: str,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Register a fine-tuned model.
        
        Args:
            job_id: Source job ID
            model_path: Path to model files
            metadata: Additional metadata
            
        Returns:
            Version ID
        """
        version_id = str(uuid.uuid4())
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(model_path)
        
        version_info = {
            "version_id": version_id,
            "job_id": job_id,
            "model_path": model_path,
            "model_hash": model_hash,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "deployments": [],
        }
        
        self._versions[version_id] = version_info
        self._save_version(version_id, version_info)
        
        logger.info(f"Registered model version: {version_id}")
        return version_id
    
    def deploy_version(
        self,
        version_id: str,
        endpoint_name: str = None,
        replica_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Deploy a model version to an endpoint.
        
        Args:
            version_id: Version to deploy
            endpoint_name: Optional custom endpoint name
            replica_count: Number of replicas
            
        Returns:
            Deployment info
        """
        if version_id not in self._versions:
            raise ValueError(f"Version not found: {version_id}")
        
        version = self._versions[version_id]
        
        deployment = {
            "deployment_id": str(uuid.uuid4()),
            "version_id": version_id,
            "endpoint": endpoint_name or f"model-{version_id[:8]}",
            "replica_count": replica_count,
            "status": "deploying",
            "deployed_at": datetime.utcnow().isoformat(),
        }
        
        version["deployments"].append(deployment)
        self._save_version(version_id, version)
        
        return deployment
    
    def list_versions(self, job_id: str = None) -> List[Dict[str, Any]]:
        """List all versions, optionally filtered by job"""
        versions = list(self._versions.values())
        
        if job_id:
            versions = [v for v in versions if v["job_id"] == job_id]
        
        return versions
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate hash of model files"""
        sha256 = hashlib.sha256()
        
        for file_path in Path(model_path).rglob("*"):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _save_version(self, version_id: str, version_info: Dict[str, Any]):
        """Persist version info"""
        version_path = self.storage_path / f"{version_id}.json"
        with open(version_path, 'w') as f:
            json.dump(version_info, f, indent=2, default=str)
    
    def _load_versions(self):
        """Load persisted versions"""
        for version_file in self.storage_path.glob("*.json"):
            try:
                with open(version_file, 'r') as f:
                    version_info = json.load(f)
                    self._versions[version_info["version_id"]] = version_info
            except Exception as e:
                logger.error(f"Failed to load version {version_file}: {e}")
