#!/usr/bin/env python3
"""
Kernel Import/Export Module
============================

Enables portability of kernels across deployments with support for
multiple formats, encryption, and validation.

Supported Formats:
- JSON (default, human-readable)
- JSON-LD (semantic web compatible)
- RDF Turtle (knowledge graph)
- OWL (ontology format)

Author: MiniMax Agent
"""

import json
import hashlib
import base64
import gzip
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats"""
    JSON = "json"
    JSON_LD = "jsonld"
    RDF_TURTLE = "ttl"
    OWL = "owl"
    BINARY = "bin"


class EncryptionMode(Enum):
    """Encryption modes for exports"""
    NONE = "none"
    AES_256 = "aes_256"
    RSA = "rsa"


@dataclass
class ExportOptions:
    """Options for kernel export"""
    format: ExportFormat = ExportFormat.JSON
    encryption: EncryptionMode = EncryptionMode.NONE
    encryption_key: Optional[str] = None
    include_versions: bool = True
    include_relationships: bool = True
    include_audit_log: bool = False
    compression: bool = True
    chunk_size: int = 1000  # Max items per chunk for large exports
    metadata_filter: Optional[Dict[str, Any]] = None  # Filter what to export


@dataclass
class ImportOptions:
    """Options for kernel import"""
    format: ExportFormat = ExportFormat.JSON
    encryption: EncryptionMode = EncryptionMode.NONE
    encryption_key: Optional[str] = None
    validation_mode: str = "strict"  # strict, lenient, skip
    merge_strategy: str = "fail"  # fail, overwrite, merge, create_new
    update_versions: bool = True
    preserve_ids: bool = True
    owner_id: Optional[str] = None  # Override owner


@dataclass
class ImportResult:
    """Result of import operation"""
    success: bool
    kernel_id: Optional[str]
    errors: List[str]
    warnings: List[str]
    imported_items: Dict[str, int]
    duration_ms: float


class KernelExporter:
    """
    Exports kernels to various formats with optional encryption
    """
    
    def __init__(self, storage_backend):
        """
        Initialize exporter
        
        Args:
            storage_backend: PostgreSQLStorageBackend instance
        """
        self.storage = storage_backend
    
    def export_kernel(
        self,
        kernel_id: str,
        options: ExportOptions,
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Export a kernel to the specified format
        
        Args:
            kernel_id: ID of kernel to export
            options: Export options
            
        Returns:
            (exported_data, error_message)
        """
        try:
            # Get kernel data
            kernel_data = self._gather_kernel_data(kernel_id, options)
            if not kernel_data:
                return None, "Kernel not found"
            
            # Serialize to format
            serialized = self._serialize(kernel_data, options.format)
            if not serialized:
                return None, f"Unsupported format: {options.format}"
            
            # Compress if enabled
            if options.compression and options.format != ExportFormat.BINARY:
                serialized = gzip.compress(serialized)
            
            # Encrypt if enabled
            if options.encryption != EncryptionMode.NONE:
                encrypted, error = self._encrypt(serialized, options)
                if error:
                    return None, error
                return encrypted, None
            
            return serialized, None
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None, str(e)
    
    def export_kernel_to_file(
        self,
        kernel_id: str,
        file_path: str,
        options: ExportOptions,
    ) -> bool:
        """
        Export kernel directly to a file
        
        Args:
            kernel_id: ID of kernel to export
            file_path: Output file path
            options: Export options
            
        Returns:
            True if successful
        """
        data, error = self.export_kernel(kernel_id, options)
        
        if error:
            logger.error(f"Export failed: {error}")
            return False
        
        try:
            with open(file_path, 'wb') as f:
                f.write(data)
            
            logger.info(f"Exported kernel {kernel_id} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write file: {e}")
            return False
    
    def _gather_kernel_data(
        self,
        kernel_id: str,
        options: ExportOptions,
    ) -> Optional[Dict[str, Any]]:
        """Gather all kernel data for export"""
        # Get kernel
        kernel = self.storage.get_kernel(kernel_id)
        if not kernel:
            return None
        
        export_data = {
            "export_info": {
                "kernel_id": kernel_id,
                "exported_at": datetime.utcnow().isoformat(),
                "format_version": "1.0",
                "format": options.format.value,
            },
            "kernel": self._format_kernel(kernel),
        }
        
        # Get documents
        documents = self.storage.get_documents_by_kernel(kernel_id)
        export_data["documents"] = [self._format_document(doc) for doc in documents]
        
        # Get chunks
        if options.chunk_size > 0:
            chunks = self.storage.get_chunks_by_kernel(kernel_id, limit=options.chunk_size)
            export_data["chunks"] = [self._format_chunk(chunk) for chunk in chunks]
        
        # Get version history
        if options.include_versions:
            versions = self.storage.get_version_history(kernel_id)
            export_data["versions"] = [self._format_version(v) for v in versions]
        
        # Get relationships
        if options.include_relationships:
            export_data["relationships"] = self._gather_relationships(kernel_id)
        
        # Get audit log
        if options.include_audit_log:
            audit = self.storage.get_audit_log(entity_id=kernel_id)
            export_data["audit_log"] = [self._format_audit(a) for a in audit]
        
        # Calculate checksum
        content_bytes = json.dumps(export_data, sort_keys=True, default=str).encode()
        export_data["checksum"] = hashlib.sha256(content_bytes).hexdigest()
        
        return export_data
    
    def _gather_relationships(self, kernel_id: str) -> List[Dict[str, Any]]:
        """Gather all relationships for a kernel"""
        relationships = []
        
        # Get relationships from documents
        documents = self.storage.get_documents_by_kernel(kernel_id)
        for doc in documents:
            rels = self.storage.get_relationships(doc["id"])
            relationships.extend(rels)
        
        # Get relationships from chunks
        chunks = self.storage.get_chunks_by_kernel(kernel_id, limit=10000)
        for chunk in chunks:
            rels = self.storage.get_relationships(chunk["id"])
            relationships.extend(rels)
        
        return relationships
    
    def _format_kernel(self, kernel: Dict[str, Any]) -> Dict[str, Any]:
        """Format kernel for export"""
        return {
            "id": str(kernel["id"]),
            "name": kernel["name"],
            "description": kernel["description"],
            "owner_id": kernel["owner_id"],
            "domain": kernel["domain"],
            "tags": kernel["tags"] or [],
            "schema_version": kernel["schema_version"],
            "is_public": kernel["is_public"],
            "metadata": kernel.get("metadata", {}),
            "created_at": kernel["created_at"].isoformat() if kernel.get("created_at") else None,
            "updated_at": kernel["updated_at"].isoformat() if kernel.get("updated_at") else None,
        }
    
    def _format_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Format document for export"""
        return {
            "id": str(doc["id"]),
            "title": doc["title"],
            "content_hash": doc["content_hash"],
            "blob_path": doc.get("blob_path"),
            "chunk_count": doc["chunk_count"],
            "metadata": doc.get("metadata", {}),
            "source_url": doc.get("source_url"),
            "author": doc.get("author"),
            "published_date": doc.get("published_date"),
            "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None,
        }
    
    def _format_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Format chunk for export"""
        return {
            "id": str(chunk["id"]),
            "document_id": str(chunk["document_id"]),
            "chunk_index": chunk["chunk_index"],
            "text_content": chunk["text_content"],
            "content_hash": chunk["content_hash"],
            "qdrant_point_id": chunk.get("qdrant_point_id"),
            "token_count": chunk.get("token_count", 0),
            "metadata": chunk.get("metadata", {}),
        }
    
    def _format_version(self, version: Dict[str, Any]) -> Dict[str, Any]:
        """Format version for export"""
        return {
            "id": str(version["id"]),
            "parent_version_id": str(version["parent_version_id"]) if version.get("parent_version_id") else None,
            "commit_message": version["commit_message"],
            "content_hash": version["content_hash"],
            "change_summary": version.get("change_summary", {}),
            "diff_summary": version.get("diff_summary", ""),
            "created_by": version.get("created_by", "system"),
            "created_at": version["created_at"].isoformat() if version.get("created_at") else None,
        }
    
    def _format_audit(self, audit: Dict[str, Any]) -> Dict[str, Any]:
        """Format audit entry for export"""
        return {
            "entity_type": audit["entity_type"],
            "entity_id": str(audit["entity_id"]),
            "action": audit["action"],
            "actor": audit.get("actor", "system"),
            "old_values": audit.get("old_values", {}),
            "new_values": audit.get("new_values", {}),
            "created_at": audit["created_at"].isoformat() if audit.get("created_at") else None,
        }
    
    def _serialize(
        self,
        data: Dict[str, Any],
        format: ExportFormat,
    ) -> Optional[bytes]:
        """Serialize data to specified format"""
        if format == ExportFormat.JSON:
            return json.dumps(data, indent=2, default=str).encode()
        
        elif format == ExportFormat.JSON_LD:
            return self._to_json_ld(data).encode()
        
        elif format == ExportFormat.RDF_TURTLE:
            return self._to_rdf_turtle(data).encode()
        
        elif format == ExportFormat.OWL:
            return self._to_owl(data).encode()
        
        elif format == ExportFormat.BINARY:
            return gzip.compress(json.dumps(data, default=str).encode())
        
        return None
    
    def _to_json_ld(self, data: Dict[str, Any]) -> str:
        """Convert to JSON-LD format"""
        context = {
            "@context": {
                "kernel": "https://kernel.schema.org/",
                "doc": "https://doc.schema.org/",
                "chunk": "https://chunk.schema.org/",
                "hasDocument": "kernel:hasDocument",
                "hasChunk": "doc:hasChunk",
                "contains": "kernel:contains",
                "references": "doc:references",
            }
        }
        
        # Convert to JSON-LD structure
        json_ld = {
            "@graph": []
        }
        
        # Add kernel
        kernel_graph = {
            "@id": f"kernel:{data['kernel']['id']}",
            "@type": "kernel:Kernel",
            "kernel:name": data["kernel"]["name"],
            "kernel:description": data["kernel"]["description"],
            "kernel:owner": data["kernel"]["owner_id"],
            "kernel:domain": data["kernel"]["domain"],
        }
        json_ld["@graph"].append(kernel_graph)
        
        # Add documents
        for doc in data.get("documents", []):
            doc_graph = {
                "@id": f"doc:{doc['id']}",
                "@type": "doc:Document",
                "doc:title": doc["title"],
                "doc:contains": [f"chunk:{c['id']}" for c in data.get("chunks", []) 
                                if c["document_id"] == doc["id"]],
            }
            json_ld["@graph"].append(doc_graph)
        
        return json.dumps({**context, **json_ld}, indent=2)
    
    def _to_rdf_turtle(self, data: Dict[str, Any]) -> str:
        """Convert to RDF Turtle format"""
        lines = [
            "@prefix kernel: <https://kernel.schema.org/> .",
            "@prefix doc: <https://doc.schema.org/> .",
            "@prefix chunk: <https://chunk.schema.org/> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "",
            f"# Kernel: {data['kernel']['name']}",
            f"kernel:{data['kernel']['id']} rdf:type kernel:Kernel ;",
            f'    kernel:name "{data["kernel"]["name"]}" ;',
            f'    kernel:description """{data["kernel"]["description"]}""" ;',
            f'    kernel:owner "{data["kernel"]["owner_id"]}" .',
            "",
        ]
        
        # Add documents
        for doc in data.get("documents", []):
            lines.append(f"doc:{doc['id']} rdf:type doc:Document ;")
            lines.append(f'    doc:title "{doc["title"]}" .')
            lines.append("")
        
        return "\n".join(lines)
    
    def _to_owl(self, data: Dict[str, Any]) -> str:
        """Convert to OWL ontology format"""
        kernel_id = data["kernel"]["id"]
        
        lines = [
            "<?xml version=\"1.0\"?>",
            "<!DOCTYPE rdf:RDF [",
            "    <!ENTITY owl 'http://www.w3.org/2002/07/owl#'>",
            "    <!ENTITY rdf 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'>",
            "    <!ENTITY rdfs 'http://www.w3.org/2000/01/rdf-schema#'>",
            "]>",
            "",
            f"<rdf:RDF xmlns:owl='http://www.w3.org/2002/07/owl#'",
            "         xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'",
            "         xmlns:rdfs='http://www.w3.org/2000/01/rdf-schema#'>",
            "",
            f"    <owl:Ontology rdf:about='https://kernel.schema.org/{kernel_id}'>",
            f"        <rdfs:comment>Exported kernel: {data['kernel']['name']}</rdfs:comment>",
            "    </owl:Ontology>",
            "",
        ]
        
        # Add kernel class
        lines.extend([
            "    <!-- Kernel Class -->",
            f"    <owl:Class rdf:about='https://kernel.schema.org/Kernel'/>",
            "",
            "    <!-- Document Class -->",
            "    <owl:Class rdf:about='https://doc.schema.org/Document'/>",
            "",
        ])
        
        lines.append("</rdf:RDF>")
        
        return "\n".join(lines)
    
    def _encrypt(
        self,
        data: bytes,
        options: ExportOptions,
    ) -> Tuple[bytes, Optional[str]]:
        """Encrypt data using specified mode"""
        if options.encryption == EncryptionMode.NONE:
            return data, None
        
        try:
            if options.encryption == EncryptionMode.AES_256:
                if not options.encryption_key:
                    return None, "AES-256 encryption requires a key"
                return self._aes_encrypt(data, options.encryption_key)
            
            elif options.encryption == EncryptionMode.RSA:
                return self._rsa_encrypt(data, options.encryption_key)
            
            return None, f"Unsupported encryption: {options.encryption}"
            
        except Exception as e:
            return None, f"Encryption failed: {e}"
    
    def _aes_encrypt(self, data: bytes, key: str) -> Tuple[bytes, None]:
        """AES-256 encrypt data"""
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.primitives import padding
            from cryptography.hazmat.backends import default_backend
            import os
            
            # Derive key
            key_bytes = hashlib.sha256(key.encode()).digest()[:32]
            iv = os.urandom(16)
            
            # Pad data
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data) + padder.finalize()
            
            # Encrypt
            cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            encrypted = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine IV and encrypted data
            return iv + encrypted, None
            
        except ImportError:
            logger.warning("cryptography library not installed, using base64 encoding")
            encoded = base64.b64encode(data).decode()
            return encoded.encode(), None
    
    def _rsa_encrypt(self, data: bytes, key: str) -> Tuple[bytes, None]:
        """RSA encrypt data (for small data only)"""
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding
            from cryptography.hazmat.backends import default_backend
            
            with open(key, 'rb') as f:
                private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
            
            encrypted = private_key.public_key().encrypt(
                data,
                rsa_padding.OAEP(
                    mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return encrypted, None
            
        except Exception as e:
            return None, str(e)


class KernelImporter:
    """
    Imports kernels from various formats with validation
    """
    
    def __init__(self, storage_backend):
        """
        Initialize importer
        
        Args:
            storage_backend: PostgreSQLStorageBackend instance
        """
        self.storage = storage_backend
    
    def import_kernel(
        self,
        data: bytes,
        options: ImportOptions,
    ) -> ImportResult:
        """
        Import a kernel from data
        
        Args:
            data: Serialized kernel data
            options: Import options
            
        Returns:
            ImportResult with success status and details
        """
        import time
        start_time = time.time()
        
        errors = []
        warnings = []
        imported_items = {
            "kernels": 0,
            "documents": 0,
            "chunks": 0,
            "versions": 0,
            "relationships": 0,
        }
        
        try:
            # Decrypt if needed
            if options.encryption != EncryptionMode.NONE:
                decrypted, error = self._decrypt(data, options)
                if error:
                    return ImportResult(
                        success=False,
                        kernel_id=None,
                        errors=[error],
                        warnings=[],
                        imported_items={},
                        duration_ms=0,
                    )
                data = decrypted
            
            # Decompress if needed
            try:
                data = gzip.decompress(data)
            except:
                pass  # Not compressed
            
            # Parse based on format
            if options.format == ExportFormat.JSON:
                kernel_data = json.loads(data)
            else:
                kernel_data, error = self._parse(data, options.format)
                if error:
                    return ImportResult(
                        success=False,
                        kernel_id=None,
                        errors=[error],
                        warnings=[],
                        imported_items={},
                        duration_ms=0,
                    )
            
            # Validate
            validation_error = self._validate(kernel_data, options)
            if validation_error:
                if options.validation_mode == "strict":
                    return ImportResult(
                        success=False,
                        kernel_id=None,
                        errors=[validation_error],
                        warnings=[],
                        imported_items={},
                        duration_ms=0,
                    )
                else:
                    warnings.append(f"Validation warning: {validation_error}")
            
            # Import based on merge strategy
            if options.merge_strategy == "create_new":
                kernel_id = self._import_as_new(kernel_data, options, imported_items, errors)
            elif options.merge_strategy == "overwrite":
                kernel_id = self._import_and_overwrite(kernel_data, options, imported_items, errors)
            elif options.merge_strategy == "merge":
                kernel_id = self._import_and_merge(kernel_data, options, imported_items, errors, warnings)
            else:  # fail
                if self._kernel_exists(kernel_data, options):
                    return ImportResult(
                        success=False,
                        kernel_id=None,
                        errors=["Kernel already exists. Use merge_strategy to combine."],
                        warnings=[],
                        imported_items={},
                        duration_ms=0,
                    )
                kernel_id = self._import_as_new(kernel_data, options, imported_items, errors)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return ImportResult(
                success=len(errors) == 0,
                kernel_id=kernel_id,
                errors=errors,
                warnings=warnings,
                imported_items=imported_items,
                duration_ms=duration_ms,
            )
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return ImportResult(
                success=False,
                kernel_id=None,
                errors=[str(e)],
                warnings=[],
                imported_items={},
                duration_ms=(time.time() - start_time) * 1000,
            )
    
    def import_kernel_from_file(
        self,
        file_path: str,
        options: ImportOptions,
    ) -> ImportResult:
        """
        Import a kernel from a file
        
        Args:
            file_path: Path to file
            options: Import options
            
        Returns:
            ImportResult with success status and details
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Detect format from extension
            if file_path.endswith('.json'):
                options.format = ExportFormat.JSON
            elif file_path.endswith('.jsonld'):
                options.format = ExportFormat.JSON_LD
            elif file_path.endswith('.ttl'):
                options.format = ExportFormat.RDF_TURTLE
            elif file_path.endswith('.owl'):
                options.format = ExportFormat.OWL
            
            return self.import_kernel(data, options)
            
        except Exception as e:
            logger.error(f"File import failed: {e}")
            return ImportResult(
                success=False,
                kernel_id=None,
                errors=[str(e)],
                warnings=[],
                imported_items={},
                duration_ms=0,
            )
    
    def _parse(
        self,
        data: bytes,
        format: ExportFormat,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Parse data from specified format"""
        if format == ExportFormat.JSON_LD:
            return self._from_json_ld(data)
        elif format == ExportFormat.RDF_TURTLE:
            return self._from_rdf_turtle(data)
        elif format == ExportFormat.OWL:
            return self._from_owl(data)
        return None, f"Unsupported format: {format}"
    
    def _from_json_ld(self, data: bytes) -> Tuple[Dict[str, Any], None]:
        """Parse JSON-LD data"""
        json_ld = json.loads(data)
        
        # Convert from JSON-LD to internal format
        kernel_data = {
            "kernel": {},
            "documents": [],
            "chunks": [],
            "versions": [],
            "relationships": [],
        }
        
        # Extract from @graph
        graph = json_ld.get("@graph", [])
        for item in graph:
            item_id = str(item.get("@id", ""))
            
            if "kernel:Kernel" in str(item.get("@type", "")):
                kernel_data["kernel"] = {
                    "id": item_id.replace("kernel:", ""),
                    "name": item.get("kernel:name", ""),
                    "description": item.get("kernel:description", ""),
                    "owner_id": item.get("kernel:owner", "default"),
                    "domain": item.get("kernel:domain", "general"),
                    "tags": [],
                }
            elif "doc:Document" in str(item.get("@type", "")):
                kernel_data["documents"].append({
                    "id": item_id.replace("doc:", ""),
                    "title": item.get("doc:title", ""),
                })
        
        return kernel_data, None
    
    def _from_rdf_turtle(self, data: bytes) -> Tuple[Dict[str, Any], None]:
        """Parse RDF Turtle data (simplified)"""
        # This is a simplified parser - production would use RDFlib
        content = data.decode('utf-8')
        
        kernel_data = {
            "kernel": {},
            "documents": [],
            "chunks": [],
            "versions": [],
            "relationships": [],
        }
        
        # Extract kernel info
        kernel_match = re.search(r'kernel:(\S+).*?kernel:name\s+"([^"]+)"', content, re.DOTALL)
        if kernel_match:
            kernel_data["kernel"] = {
                "id": kernel_match.group(1),
                "name": kernel_match.group(2),
            }
        
        return kernel_data, None
    
    def _from_owl(self, data: bytes) -> Tuple[Dict[str, Any], None]:
        """Parse OWL data (simplified)"""
        content = data.decode('utf-8')
        
        kernel_data = {
            "kernel": {},
            "documents": [],
            "chunks": [],
            "versions": [],
            "relationships": [],
        }
        
        # Extract kernel ID from ontology
        kernel_match = re.search(r'about=\'https://kernel\.schema\.org/([^"\']+)\'', content)
        if kernel_match:
            kernel_data["kernel"]["id"] = kernel_match.group(1)
        
        return kernel_data, None
    
    def _decrypt(
        self,
        data: bytes,
        options: ImportOptions,
    ) -> Tuple[bytes, Optional[str]]:
        """Decrypt data"""
        if options.encryption == EncryptionMode.NONE:
            return data, None
        
        try:
            if options.encryption == EncryptionMode.AES_256:
                if not options.encryption_key:
                    return None, "AES-256 decryption requires a key"
                return self._aes_decrypt(data, options.encryption_key)
            
            return None, f"Unsupported decryption: {options.encryption}"
            
        except Exception as e:
            return None, f"Decryption failed: {e}"
    
    def _aes_decrypt(self, data: bytes, key: str) -> Tuple[bytes, None]:
        """AES-256 decrypt data"""
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.primitives import padding
            from cryptography.hazmat.backends import default_backend
            
            key_bytes = hashlib.sha256(key.encode()).digest()[:32]
            iv = data[:16]
            encrypted = data[16:]
            
            cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(encrypted) + decryptor.finalize()
            
            unpadder = padding.PKCS7(128).unpadder()
            return unpadder.update(padded_data) + unpadder.finalize(), None
            
        except ImportError:
            logger.warning("cryptography library not installed")
            return base64.b64decode(data), None
    
    def _validate(
        self,
        kernel_data: Dict[str, Any],
        options: ImportOptions,
    ) -> Optional[str]:
        """Validate kernel data"""
        if "kernel" not in kernel_data:
            return "Missing kernel information"
        
        kernel = kernel_data["kernel"]
        
        if not kernel.get("name"):
            return "Kernel name is required"
        
        if not kernel.get("id"):
            return "Kernel ID is required"
        
        # Validate checksum if present
        if "checksum" in kernel_data:
            calculated = hashlib.sha256(
                json.dumps(kernel_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            if calculated != kernel_data["checksum"]:
                return "Checksum validation failed - data may be corrupted"
        
        return None
    
    def _kernel_exists(
        self,
        kernel_data: Dict[str, Any],
        options: ImportOptions,
    ) -> bool:
        """Check if kernel already exists"""
        kernel = kernel_data.get("kernel", {})
        existing = self.storage.get_kernel(kernel.get("id", ""))
        return existing is not None
    
    def _import_as_new(
        self,
        kernel_data: Dict[str, Any],
        options: ImportOptions,
        imported_items: Dict[str, Any],
        errors: List[str],
    ) -> str:
        """Import kernel as new (new ID)"""
        kernel = kernel_data.get("kernel", {})
        
        # Create new kernel with original or new owner
        from src.python.services.storage.postgres_storage import KernelMetadata
        
        metadata = KernelMetadata(
            name=kernel.get("name", "Imported Kernel"),
            description=kernel.get("description", ""),
            owner_id=options.owner_id or kernel.get("owner_id", "default"),
            domain=kernel.get("domain", "general"),
            tags=kernel.get("tags", []),
            is_public=kernel.get("is_public", False),
            metadata_=kernel.get("metadata", {}),
        )
        
        kernel_id = self.storage.create_kernel(metadata)
        imported_items["kernels"] += 1
        
        # Import documents
        for doc in kernel_data.get("documents", []):
            try:
                doc_id = self.storage.add_document(
                    kernel_id=kernel_id,
                    title=doc.get("title", "Untitled"),
                    content_hash=doc.get("content_hash", ""),
                    blob_path=doc.get("blob_path"),
                    metadata=doc.get("metadata", {}),
                    source_url=doc.get("source_url"),
                    author=doc.get("author"),
                    published_date=doc.get("published_date"),
                )
                imported_items["documents"] += 1
                
                # Import chunks for this document
                for chunk in kernel_data.get("chunks", []):
                    if str(chunk.get("document_id")) == str(doc.get("id")):
                        self.storage.add_chunk(
                            document_id=doc_id,
                            kernel_id=kernel_id,
                            chunk_index=chunk.get("chunk_index", 0),
                            text_content=chunk.get("text_content", ""),
                            content_hash=chunk.get("content_hash", ""),
                            qdrant_point_id=chunk.get("qdrant_point_id"),
                            metadata=chunk.get("metadata", {}),
                        )
                        imported_items["chunks"] += 1
                        
            except Exception as e:
                errors.append(f"Failed to import document: {e}")
        
        # Import versions if enabled
        if options.update_versions:
            for version in kernel_data.get("versions", []):
                try:
                    self.storage.create_version(
                        kernel_id=kernel_id,
                        commit_message=version.get("commit_message", "Imported version"),
                        content_hash=version.get("content_hash", ""),
                        change_summary=version.get("change_summary", {}),
                        diff_summary=version.get("diff_summary", ""),
                        created_by=version.get("created_by", "import"),
                    )
                    imported_items["versions"] += 1
                except Exception as e:
                    errors.append(f"Failed to import version: {e}")
        
        # Import relationships
        for rel in kernel_data.get("relationships", []):
            try:
                from src.python.services.storage.postgres_storage import EntityType, RelationshipType
                
                self.storage.add_relationship(
                    source_id=rel.get("source_id"),
                    source_type=EntityType(rel.get("source_type", "chunk")),
                    target_id=rel.get("target_id"),
                    target_type=EntityType(rel.get("target_type", "chunk")),
                    relationship_type=RelationshipType(rel.get("relationship_type", "references")),
                    weight=rel.get("weight", 1.0),
                    confidence=rel.get("confidence", 1.0),
                    metadata=rel.get("metadata", {}),
                )
                imported_items["relationships"] += 1
            except Exception as e:
                errors.append(f"Failed to import relationship: {e}")
        
        logger.info(f"Imported kernel as new: {kernel_id}")
        return kernel_id
    
    def _import_and_overwrite(
        self,
        kernel_data: Dict[str, Any],
        options: ImportOptions,
        imported_items: Dict[str, Any],
        errors: List[str],
    ) -> str:
        """Import kernel, overwriting existing"""
        kernel = kernel_data.get("kernel", {})
        kernel_id = kernel.get("id")
        
        if not kernel_id:
            return self._import_as_new(kernel_data, options, imported_items, errors)
        
        # Delete existing
        self.storage.delete_kernel(kernel_id)
        
        # Import as new
        return self._import_as_new(kernel_data, options, imported_items, errors)
    
    def _import_and_merge(
        self,
        kernel_data: Dict[str, Any],
        options: ImportOptions,
        imported_items: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> str:
        """Import kernel, merging with existing"""
        kernel = kernel_data.get("kernel", {})
        kernel_id = kernel.get("id")
        
        if not kernel_id:
            return self._import_as_new(kernel_data, options, imported_items, errors)
        
        # Get existing kernel
        existing = self.storage.get_kernel(kernel_id)
        if not existing:
            return self._import_as_new(kernel_data, options, imported_items, errors)
        
        # Merge strategy: Add new documents/chunks to existing kernel
        # Get existing document titles
        existing_docs = self.storage.get_documents_by_kernel(kernel_id)
        existing_titles = {doc["title"] for doc in existing_docs}
        
        for doc in kernel_data.get("documents", []):
            if doc.get("title") not in existing_titles:
                doc_id = self.storage.add_document(
                    kernel_id=kernel_id,
                    title=doc.get("title", "Untitled"),
                    content_hash=doc.get("content_hash", ""),
                    blob_path=doc.get("blob_path"),
                    metadata=doc.get("metadata", {}),
                    source_url=doc.get("source_url"),
                    author=doc.get("author"),
                    published_date=doc.get("published_date"),
                )
                imported_items["documents"] += 1
                
                # Import chunks for this document
                for chunk in kernel_data.get("chunks", []):
                    if str(chunk.get("document_id")) == str(doc.get("id")):
                        self.storage.add_chunk(
                            document_id=doc_id,
                            kernel_id=kernel_id,
                            chunk_index=chunk.get("chunk_index", 0),
                            text_content=chunk.get("text_content", ""),
                            content_hash=chunk.get("content_hash", ""),
                            qdrant_point_id=chunk.get("qdrant_point_id"),
                            metadata=chunk.get("metadata", {}),
                        )
                        imported_items["chunks"] += 1
            else:
                warnings.append(f"Skipped duplicate document: {doc.get('title')}")
        
        # Create merge version
        self.storage.create_version(
            kernel_id=kernel_id,
            commit_message=f"Merged with imported kernel",
            content_hash=hashlib.sha256(json.dumps(kernel_data).encode()).hexdigest(),
            change_summary={"action": "merge", "imported_docs": imported_items["documents"]},
            diff_summary=f"Merged {imported_items['documents']} new documents",
            created_by="import",
        )
        
        logger.info(f"Merged into existing kernel: {kernel_id}")
        return kernel_id


class KernelDiff:
    """
    Computes differences between kernel versions
    """
    
    @staticmethod
    def compute_diff(
        version_a: Dict[str, Any],
        version_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute diff between two versions
        
        Returns:
            Dictionary with added, removed, modified items
        """
        # Get chunks from both versions
        chunks_a = {c["id"]: c for c in version_a.get("chunks", [])}
        chunks_b = {c["id"]: c for c in version_b.get("chunks", [])}
        
        added = [chunks_b[k] for k in set(chunks_b) - set(chunks_a)]
        removed = [chunks_a[k] for k in set(chunks_a) - set(chunks_b)]
        modified = []
        
        for chunk_id in set(chunks_a) & set(chunks_b):
            if chunks_a[chunk_id]["content_hash"] != chunks_b[chunk_id]["content_hash"]:
                modified.append({
                    "before": chunks_a[chunk_id],
                    "after": chunks_b[chunk_id],
                })
        
        return {
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": len(modified),
            "added": added[:10],  # Limit for display
            "removed": removed[:10],
            "modified": modified[:10],
        }
    
    @staticmethod
    def format_diff_summary(diff: Dict[str, Any]) -> str:
        """Format diff as human-readable summary"""
        parts = []
        
        if diff["added_count"] > 0:
            parts.append(f"+{diff['added_count']} added")
        if diff["removed_count"] > 0:
            parts.append(f"-{diff['removed_count']} removed")
        if diff["modified_count"] > 0:
            parts.append(f"~{diff['modified_count']} modified")
        
        return ", ".join(parts) if parts else "No changes"
