#!/usr/bin/env python3
"""
Agent Knowledge Pack (AKP) Format and Loader
==============================================

Defines the standard format for portable knowledge packs that can be
distributed to agent systems. Includes manifest schema, loader implementation,
and compatibility verification.

Author: MiniMax Agent
"""

import json
import hashlib
import zipfile
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Tiered access levels for kernel licensing"""
    VIEW_ONLY = "view_only"
    QUERY = "query"
    MODIFY = "modify"
    REDISTRIBUTE = "redistribute"


class CompatibilityLevel(Enum):
    """Model compatibility levels"""
    RAW_TEXT = "raw_text"
    EMBEDDING_OPTIMIZED = "embedding_optimized"
    FINE_TUNE_READY = "fine_tune_ready"
    DEPLOYMENT_READY = "deployment_ready"


@dataclass
class AKPSystemPrompt:
    """System prompt configuration for the knowledge pack"""
    template: str
    variables: List[str] = field(default_factory=list)
    chain_of_thought: bool = False
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class AKPConstraint:
    """Behavioral constraints for agent behavior"""
    type: str  # "output_format", "safety", "domain_specific", "tone"
    description: str
    rule: str
    severity: str = "warning"  # "error", "warning", "info"


@dataclass
class AKPToolBinding:
    """Tool binding configuration"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str] = field(default_factory=list)


@dataclass
class AKPDependency:
    """Dependency on other knowledge packs"""
    pack_id: str
    minimum_version: str
    import_names: List[str] = field(default_factory=list)


@dataclass
class AKPPermissions:
    """Permissions required for the knowledge pack"""
    capabilities: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None  # requests per minute
    context_window: Optional[int] = None  # preferred context window size
    requires_api_key: bool = True


@dataclass
class AKPUtilizationMetrics:
    """Metrics for pack utilization tracking"""
    total_queries: int = 0
    unique_users: int = 0
    average_response_time_ms: float = 0.0
    token_usage: int = 0
    last_accessed: Optional[datetime] = None


class AgentKnowledgePack:
    """
    Represents a complete Agent Knowledge Pack with all metadata and assets.
    
    This is the in-memory representation. For storage/transmission, use
    the KnowledgePackFormat class to serialize to/from AKP files.
    """
    
    def __init__(
        self,
        pack_id: str,
        name: str,
        version: str,
        author_id: str,
        description: str = "",
        domain: str = "general",
        tags: List[str] = None,
    ):
        self.pack_id = pack_id
        self.name = name
        self.version = version
        self.author_id = author_id
        self.description = description
        self.domain = domain
        self.tags = tags or []
        
        # Core content
        self.chunks: List[Dict[str, Any]] = []
        self.vector_index_config: Optional[Dict[str, Any]] = None
        self.embeddings: Optional[Dict[str, Any]] = None  # Store or reference to embeddings
        
        # System configuration
        self.system_prompt: Optional[AKPSystemPrompt] = None
        self.constraints: List[AKPConstraint] = []
        self.tool_bindings: List[AKPToolBinding] = []
        
        # Dependencies and compatibility
        self.dependencies: List[AKPDependency] = []
        self.compatibility_level: CompatibilityLevel = CompatibilityLevel.RAW_TEXT
        self.supported_frameworks: List[str] = []  # ["langchain", "autoGPT", "crewai"]
        
        # Licensing and distribution
        self.license_type: str = "proprietary"
        self.pricing_tier: str = "free"
        self.access_level: AccessLevel = AccessLevel.VIEW_ONLY
        self.permissions: AKPPermissions = AKPPermissions()
        
        # Quality and certification
        self.quality_score: float = 0.0
        self.certification_status: str = "uncertified"
        self.review_count: int = 0
        self.average_rating: float = 0.0
        
        # Metadata
        self.created_at: datetime = datetime.utcnow()
        self.updated_at: datetime = datetime.utcnow()
        self.file_size_bytes: int = 0
        self.chunk_count: int = 0
        self.token_count: int = 0
        self.metrics: AKPUtilizationMetrics = AKPUtilizationMetrics()
        
        # Signature for integrity verification
        self.signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "pack_id": self.pack_id,
            "name": self.name,
            "version": self.version,
            "author_id": self.author_id,
            "description": self.description,
            "domain": self.domain,
            "tags": self.tags,
            "chunk_count": self.chunk_count,
            "token_count": self.token_count,
            "compatibility_level": self.compatibility_level.value,
            "supported_frameworks": self.supported_frameworks,
            "license_type": self.license_type,
            "pricing_tier": self.pricing_tier,
            "access_level": self.access_level.value,
            "quality_score": self.quality_score,
            "certification_status": self.certification_status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "file_size_bytes": self.file_size_bytes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentKnowledgePack":
        """Create from dictionary"""
        pack = cls(
            pack_id=data["pack_id"],
            name=data["name"],
            version=data["version"],
            author_id=data["author_id"],
            description=data.get("description", ""),
            domain=data.get("domain", "general"),
            tags=data.get("tags", []),
        )
        pack.chunk_count = data.get("chunk_count", 0)
        pack.token_count = data.get("token_count", 0)
        pack.compatibility_level = CompatibilityLevel(data.get("compatibility_level", "raw_text"))
        pack.supported_frameworks = data.get("supported_frameworks", [])
        pack.license_type = data.get("license_type", "proprietary")
        pack.pricing_tier = data.get("pricing_tier", "free")
        pack.access_level = AccessLevel(data.get("access_level", "view_only"))
        pack.quality_score = data.get("quality_score", 0.0)
        pack.certification_status = data.get("certification_status", "uncertified")
        pack.created_at = datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat()))
        pack.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat()))
        pack.file_size_bytes = data.get("file_size_bytes", 0)
        return pack


class KnowledgePackFormat:
    """
    Handles serialization and deserialization of knowledge packs
    to/from the AKP file format.
    
    AKP File Structure:
    /pack.akp
      ├── manifest.json          # Core metadata
      ├── chunks.jsonl          # Content chunks
      ├── vectors.bin           # Embeddings (optional)
      ├── system_prompt.yaml    # Prompt configuration
      ├── constraints.yaml      # Behavioral constraints
      ├── tool_bindings.yaml    # Tool definitions
      ├── dependencies.yaml     # External dependencies
      ├── vector_config.yaml    # Vector index settings
      └── checksum.sha256       # Integrity verification
    """
    
    MANIFEST_FILENAME = "manifest.json"
    CHUNKS_FILENAME = "chunks.jsonl"
    EMBEDDINGS_FILENAME = "embeddings.bin"
    VECTOR_CONFIG_FILENAME = "vector_config.yaml"
    SYSTEM_PROMPT_FILENAME = "system_prompt.yaml"
    CONSTRAINTS_FILENAME = "constraints.yaml"
    TOOL_BINDINGS_FILENAME = "tool_bindings.yaml"
    DEPENDENCIES_FILENAME = "dependencies.yaml"
    CHECKSUM_FILENAME = "checksum.sha256"
    
    def __init__(self, storage_path: str = "./knowledge_packs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def pack_to_file(self, pack: AgentKnowledgePack, filepath: str) -> str:
        """
        Serialize a knowledge pack to AKP file format.
        
        Args:
            pack: The knowledge pack to serialize
            filepath: Output file path (should end with .akp)
            
        Returns:
            SHA256 checksum of the packed file
        """
        filepath = Path(filepath)
        
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 1. Write manifest
            manifest = pack.to_dict()
            zf.writestr(self.MANIFEST_FILENAME, json.dumps(manifest, indent=2))
            
            # 2. Write chunks as JSONL
            if pack.chunks:
                chunk_lines = [json.dumps(chunk) for chunk in pack.chunks]
                zf.writestr(self.CHUNKS_FILENAME, '\n'.join(chunk_lines))
            
            # 3. Write vector config if present
            if pack.vector_index_config:
                zf.writestr(
                    self.VECTOR_CONFIG_FILENAME,
                    yaml.dump(pack.vector_index_config)
                )
            
            # 4. Write system prompt if present
            if pack.system_prompt:
                sp_data = {
                    "template": pack.system_prompt.template,
                    "variables": pack.system_prompt.variables,
                    "chain_of_thought": pack.system_prompt.chain_of_thought,
                    "few_shot_examples": pack.system_prompt.few_shot_examples,
                }
                zf.writestr(self.SYSTEM_PROMPT_FILENAME, yaml.dump(sp_data))
            
            # 5. Write constraints if present
            if pack.constraints:
                constraints_data = [
                    {
                        "type": c.type,
                        "description": c.description,
                        "rule": c.rule,
                        "severity": c.severity,
                    }
                    for c in pack.constraints
                ]
                zf.writestr(self.CONSTRAINTS_FILENAME, yaml.dump(constraints_data))
            
            # 6. Write tool bindings if present
            if pack.tool_bindings:
                tb_data = [
                    {
                        "name": tb.name,
                        "description": tb.description,
                        "parameters": tb.parameters,
                        "required_params": tb.required_params,
                    }
                    for tb in pack.tool_bindings
                ]
                zf.writestr(self.TOOL_BINDINGS_FILENAME, yaml.dump(tb_data))
            
            # 7. Write dependencies if present
            if pack.dependencies:
                dep_data = [
                    {
                        "pack_id": d.pack_id,
                        "minimum_version": d.minimum_version,
                        "import_names": d.import_names,
                    }
                    for d in pack.dependencies
                ]
                zf.writestr(self.DEPENDENCIES_FILENAME, yaml.dump(dep_data))
        
        # Calculate checksum
        checksum = self._calculate_checksum(filepath)
        
        # Append checksum file
        with zipfile.ZipFile(filepath, 'a', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(self.CHECKSUM_FILENAME, checksum)
        
        # Update pack metadata
        pack.file_size_bytes = filepath.stat().st_size
        pack.updated_at = datetime.utcnow()
        
        logger.info(f"Packed knowledge pack: {pack.pack_id} -> {filepath}")
        return checksum
    
    def unpack_from_file(self, filepath: str) -> AgentKnowledgePack:
        """
        Deserialize a knowledge pack from AKP file format.
        
        Args:
            filepath: Path to the .akp file
            
        Returns:
            Deserialized knowledge pack
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Knowledge pack not found: {filepath}")
        
        with zipfile.ZipFile(filepath, 'r') as zf:
            # 1. Read manifest
            manifest_data = json.loads(zf.read(self.MANIFEST_FILENAME))
            pack = AgentKnowledgePack.from_dict(manifest_data)
            
            # 2. Read chunks if present
            if self.CHUNKS_FILENAME in zf.namelist():
                chunk_data = zf.read(self.CHUNKS_FILENAME).decode('utf-8')
                pack.chunks = [json.loads(line) for line in chunk_data.strip().split('\n') if line]
            
            # 3. Read vector config if present
            if self.VECTOR_CONFIG_FILENAME in zf.namelist():
                config_data = yaml.safe_load(zf.read(self.VECTOR_CONFIG_FILENAME))
                pack.vector_index_config = config_data
            
            # 4. Read system prompt if present
            if self.SYSTEM_PROMPT_FILENAME in zf.namelist():
                sp_data = yaml.safe_load(zf.read(self.SYSTEM_PROMPT_FILENAME))
                pack.system_prompt = AKPSystemPrompt(
                    template=sp_data["template"],
                    variables=sp_data.get("variables", []),
                    chain_of_thought=sp_data.get("chain_of_thought", False),
                    few_shot_examples=sp_data.get("few_shot_examples", []),
                )
            
            # 5. Read constraints if present
            if self.CONSTRAINTS_FILENAME in zf.namelist():
                constraints_data = yaml.safe_load(zf.read(self.CONSTRAINTS_FILENAME))
                pack.constraints = [
                    AKPConstraint(
                        type=c["type"],
                        description=c["description"],
                        rule=c["rule"],
                        severity=c.get("severity", "warning"),
                    )
                    for c in constraints_data
                ]
            
            # 6. Read tool bindings if present
            if self.TOOL_BINDINGS_FILENAME in zf.namelist():
                tb_data = yaml.safe_load(zf.read(self.TOOL_BINDINGS_FILENAME))
                pack.tool_bindings = [
                    AKPToolBinding(
                        name=tb["name"],
                        description=tb["description"],
                        parameters=tb["parameters"],
                        required_params=tb.get("required_params", []),
                    )
                    for tb in tb_data
                ]
            
            # 7. Read dependencies if present
            if self.DEPENDENCIES_FILENAME in zf.namelist():
                dep_data = yaml.safe_load(zf.read(self.DEPENDENCIES_FILENAME))
                pack.dependencies = [
                    AKPDependency(
                        pack_id=d["pack_id"],
                        minimum_version=d["minimum_version"],
                        import_names=d.get("import_names", []),
                    )
                    for d in dep_data
                ]
            
            # 8. Verify checksum
            if self.CHECKSUM_FILENAME in zf.namelist():
                stored_checksum = zf.read(self.CHECKSUM_FILENAME).decode('utf-8').strip()
                calculated_checksum = self._calculate_checksum(filepath)
                if stored_checksum != calculated_checksum:
                    raise ValueError("Knowledge pack checksum mismatch - file may be corrupted")
        
        logger.info(f"Unpacked knowledge pack: {pack.pack_id} from {filepath}")
        return pack
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def verify_integrity(self, filepath: str) -> bool:
        """Verify the integrity of a knowledge pack file"""
        try:
            pack = self.unpack_from_file(filepath)
            return True
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False


class KnowledgePackLoader:
    """
    Loads and initializes knowledge packs for use with agent frameworks.
    
    Provides framework-specific integration methods for:
    - LangChain
    - AutoGPT
    - CrewAI
    - Custom implementations
    """
    
    def __init__(self, format_handler: KnowledgePackFormat = None):
        self.format = format_handler or KnowledgePackFormat()
        self._loaded_packs: Dict[str, AgentKnowledgePack] = {}
    
    def load_pack(self, filepath: str) -> AgentKnowledgePack:
        """Load a knowledge pack into memory"""
        pack = self.format.unpack_from_file(filepath)
        self._loaded_packs[pack.pack_id] = pack
        return pack
    
    def get_pack(self, pack_id: str) -> Optional[AgentKnowledgePack]:
        """Get a loaded pack by ID"""
        return self._loaded_packs.get(pack_id)
    
    def unload_pack(self, pack_id: str) -> bool:
        """Unload a knowledge pack from memory"""
        if pack_id in self._loaded_packs:
            del self._loaded_packs[pack_id]
            return True
        return False
    
    def list_loaded(self) -> List[str]:
        """List IDs of all loaded packs"""
        return list(self._loaded_packs.keys())
    
    def to_langchain_retriever(self, pack_id: str, search_kwargs: Dict[str, Any] = None):
        """
        Convert a loaded pack to a LangChain retriever.
        
        Requires: langchain-core, langchain-qdrant
        
        Args:
            pack_id: ID of the loaded pack
            search_kwargs: Additional search parameters (k, filter, etc.)
            
        Returns:
            LangChain Retriever instance
        """
        pack = self.get_pack(pack_id)
        if not pack:
            raise ValueError(f"Pack not loaded: {pack_id}")
        
        try:
            from langchain_core.retrievers import BaseRetriever
            from langchain_core.documents import Document
            
            class KernelPackRetriever(BaseRetriever):
                """Custom retriever for knowledge pack chunks"""
                
                def __init__(self, chunks, vector_config=None):
                    self.chunks = chunks
                    self.vector_config = vector_config
                    super().__init__()
                
                def _get_relevant_documents(self, query: str) -> List[Document]:
                    # Simple keyword-based retrieval
                    # In production, would use embeddings/vector search
                    relevant = [
                        chunk for chunk in self.chunks
                        if query.lower() in chunk.get("content", "").lower()
                    ][:search_kwargs.get("k", 5)]
                    
                    return [
                        Document(
                            page_content=chunk.get("content", ""),
                            metadata=chunk.get("metadata", {})
                        )
                        for chunk in relevant
                    ]
            
            return KernelPackRetriever(
                chunks=pack.chunks,
                vector_config=pack.vector_index_config
            )
            
        except ImportError:
            logger.warning("LangChain not installed. Returning mock retriever.")
            return None
    
    def to_langchain_prompt(self, pack_id: str) -> str:
        """
        Get the system prompt from a pack for LangChain use.
        
        Returns:
            Formatted system prompt string
        """
        pack = self.get_pack(pack_id)
        if not pack or not pack.system_prompt:
            return ""
        
        prompt = pack.system_prompt.template
        
        # Replace variables
        for var in pack.system_prompt.variables:
            prompt = prompt.replace(f"{{{{{var}}}}}", f"{{{{{var}}}}}")
        
        return prompt
    
    def to_crewai_agent(self, pack_id: str, agent_name: str = None):
        """
        Create a CrewAI agent configuration from a knowledge pack.
        
        Requires: crewai
        
        Args:
            pack_id: ID of the loaded pack
            agent_name: Optional name for the agent
            
        Returns:
            CrewAI Agent configuration dict
        """
        pack = self.get_pack(pack_id)
        if not pack:
            raise ValueError(f"Pack not loaded: {pack_id}")
        
        agent_config = {
            "name": agent_name or pack.name,
            "role": f"{pack.domain} Expert",
            "goal": pack.description,
            "backstory": f"You are a specialized agent trained on {pack.name}. "
                        f"You have deep expertise in the {pack.domain} domain.",
            "allow_delegation": False,
            "tools": [tb.name for tb in pack.tool_bindings],
        }
        
        # Add memory/context if available
        if pack.system_prompt:
            agent_config["system_prompt"] = pack.system_prompt.template
        
        return agent_config
    
    def get_tool_definitions(self, pack_id: str) -> List[Dict[str, Any]]:
        """
        Get tool binding definitions for API/tool integration.
        
        Args:
            pack_id: ID of the loaded pack
            
        Returns:
            List of tool definitions for framework integration
        """
        pack = self.get_pack(pack_id)
        if not pack:
            return []
        
        return [
            {
                "name": tb.name,
                "description": tb.description,
                "input_schema": tb.parameters,
            }
            for tb in pack.tool_bindings
        ]
    
    def check_dependencies(self, pack_id: str) -> Dict[str, Any]:
        """
        Check if a pack's dependencies are satisfied.
        
        Args:
            pack_id: ID of the loaded pack
            
        Returns:
            Dict with 'satisfied' bool and 'missing' list
        """
        pack = self.get_pack(pack_id)
        if not pack:
            return {"satisfied": False, "missing": [pack_id]}
        
        missing = []
        for dep in pack.dependencies:
            if dep.pack_id not in self._loaded_packs:
                missing.append(dep.pack_id)
        
        return {
            "satisfied": len(missing) == 0,
            "missing": missing,
        }
    
    def export_for_fine_tuning(self, pack_id: str, output_path: str, format: str = "jsonl") -> str:
        """
        Export a knowledge pack in fine-tuning ready format.
        
        Args:
            pack_id: ID of the loaded pack
            output_path: Path for output file
            format: Export format ("jsonl", "csv", "parquet")
            
        Returns:
            Path to exported file
        """
        pack = self.get_pack(pack_id)
        if not pack:
            raise ValueError(f"Pack not loaded: {pack_id}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(output_path, 'w') as f:
                for chunk in pack.chunks:
                    # Format for instruction fine-tuning
                    record = {
                        "instruction": f"Answer questions about {pack.domain}",
                        "input": chunk.get("content", "")[:500],  # Truncate for input
                        "output": chunk.get("content", ""),  # Full content as reference
                    }
                    f.write(json.dumps(record) + '\n')
        
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["instruction", "input", "output"])
                writer.writeheader()
                for chunk in pack.chunks:
                    writer.writerow({
                        "instruction": f"Answer questions about {pack.domain}",
                        "input": chunk.get("content", "")[:500],
                        "output": chunk.get("content", ""),
                    })
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported pack {pack_id} for fine-tuning: {output_path}")
        return str(output_path)


# Convenience function
def create_knowledge_pack(
    name: str,
    version: str,
    author_id: str,
    chunks: List[Dict[str, Any]],
    description: str = "",
    domain: str = "general",
) -> AgentKnowledgePack:
    """Create a new knowledge pack from chunks"""
    import uuid
    
    pack_id = str(uuid.uuid4())
    pack = AgentKnowledgePack(
        pack_id=pack_id,
        name=name,
        version=version,
        author_id=author_id,
        description=description,
        domain=domain,
    )
    pack.chunks = chunks
    pack.chunk_count = len(chunks)
    pack.token_count = sum(len(c.get("content", "")) for c in chunks) // 4  # Rough estimate
    
    return pack
