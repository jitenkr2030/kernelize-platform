#!/usr/bin/env python3
"""
Persistent Kernel Storage Layer
================================

PostgreSQL-based storage for kernel metadata, relationships, and versioning.
Provides persistent storage with full audit trails and version control.

Author: MiniMax Agent
"""

import json
import hashlib
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager

import logging
logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in the knowledge graph"""
    KERNEL = "kernel"
    DOCUMENT = "document"
    CHUNK = "chunk"
    CONCEPT = "concept"
    RELATION = "relation"


class RelationshipType(Enum):
    """Types of relationships between entities"""
    CONTAINS = "contains"
    REFERENCES = "references"
    DEPENDS_ON = "depends_on"
    SIMILAR_TO = "similar_to"
    CAUSES = "causes"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    DERIVED_FROM = "derived_from"


@dataclass
class KernelMetadata:
    """Metadata for a knowledge kernel"""
    name: str
    description: str = ""
    owner_id: str = "default"
    domain: str = "general"
    tags: List[str] = field(default_factory=list)
    schema_version: str = "1.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_public: bool = False
    metadata_: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelVersion:
    """Version information for a kernel"""
    version_id: str
    kernel_id: str
    parent_version_id: Optional[str]
    commit_message: str
    created_at: datetime
    created_by: str
    content_hash: str
    change_summary: Dict[str, Any] = field(default_factory=dict)
    diff_summary: str = ""


@dataclass
class DocumentRecord:
    """Document record in storage"""
    document_id: str
    kernel_id: str
    title: str
    content_hash: str
    blob_path: Optional[str]
    chunk_count: int
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class ChunkRecord:
    """Chunk record for granular storage"""
    chunk_id: str
    document_id: str
    kernel_id: str
    chunk_index: int
    text_content: str
    content_hash: str
    qdrant_point_id: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class RelationshipRecord:
    """Relationship between entities"""
    relationship_id: str
    source_id: str
    source_type: EntityType
    target_id: str
    target_type: EntityType
    relationship_type: RelationshipType
    weight: float
    confidence: float
    metadata: Dict[str, Any]
    created_at: datetime


class KernelStorageSchema:
    """
    SQL schema definition for kernel storage
    
    Schema includes:
    - Kernels: Main knowledge kernel containers
    - Documents: Documents within kernels
    - Versions: Version history for kernels
    - Chunks: Granular content chunks
    - Relationships: Entity relationships for reasoning
    - Vector Index: Mapping to Qdrant vectors
    """
    
    @staticmethod
    def get_create_statements() -> List[str]:
        """Generate SQL statements to create the schema"""
        return [
            # Enable UUID extension
            "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";",
            
            # Kernels table
            """
            CREATE TABLE IF NOT EXISTS kernels (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name VARCHAR(255) NOT NULL,
                description TEXT,
                owner_id VARCHAR(255) NOT NULL DEFAULT 'default',
                domain VARCHAR(100) DEFAULT 'general',
                tags TEXT[] DEFAULT ARRAY[]::TEXT[],
                schema_version VARCHAR(20) DEFAULT '1.0',
                current_version_id UUID,
                is_public BOOLEAN DEFAULT FALSE,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            
            # Create indexes for kernels
            """
            CREATE INDEX IF NOT EXISTS idx_kernels_owner ON kernels(owner_id);
            CREATE INDEX IF NOT EXISTS idx_kernels_domain ON kernels(domain);
            CREATE INDEX IF NOT EXISTS idx_kernels_created ON kernels(created_at);
            CREATE INDEX IF NOT EXISTS idx_kernels_name ON kernels(name);
            """,
            
            # Versions table
            """
            CREATE TABLE IF NOT EXISTS kernel_versions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                kernel_id UUID NOT NULL REFERENCES kernels(id) ON DELETE CASCADE,
                parent_version_id UUID REFERENCES kernel_versions(id),
                commit_message TEXT NOT NULL,
                content_hash VARCHAR(64) NOT NULL,
                change_summary JSONB DEFAULT '{}',
                diff_summary TEXT,
                created_by VARCHAR(255) DEFAULT 'system',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            
            # Create indexes for versions
            """
            CREATE INDEX IF NOT EXISTS idx_versions_kernel ON kernel_versions(kernel_id);
            CREATE INDEX IF NOT EXISTS idx_versions_parent ON kernel_versions(parent_version_id);
            CREATE INDEX IF NOT EXISTS idx_versions_created ON kernel_versions(created_at);
            """,
            
            # Update kernels with foreign key to current version
            """
            ALTER TABLE kernels DROP CONSTRAINT IF EXISTS fk_current_version;
            ALTER TABLE kernels ADD CONSTRAINT fk_current_version 
                FOREIGN KEY (current_version_id) REFERENCES kernel_versions(id);
            """,
            
            # Documents table
            """
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                kernel_id UUID NOT NULL REFERENCES kernels(id) ON DELETE CASCADE,
                title VARCHAR(500) NOT NULL,
                content_hash VARCHAR(64) NOT NULL,
                blob_path VARCHAR(1000),
                chunk_count INTEGER DEFAULT 0,
                metadata JSONB DEFAULT '{}',
                source_url TEXT,
                author VARCHAR(255),
                published_date TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            
            # Create indexes for documents
            """
            CREATE INDEX IF NOT EXISTS idx_documents_kernel ON documents(kernel_id);
            CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
            CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);
            """,
            
            # Chunks table (granular content)
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                kernel_id UUID NOT NULL REFERENCES kernels(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text_content TEXT NOT NULL,
                content_hash VARCHAR(64) NOT NULL,
                qdrant_point_id VARCHAR(100),
                token_count INTEGER DEFAULT 0,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            
            # Create indexes for chunks
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_kernel ON chunks(kernel_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
            CREATE INDEX IF NOT EXISTS idx_chunks_text ON chunks USING gin(to_tsvector('english', text_content));
            """,
            
            # Entity relationships table
            """
            CREATE TABLE IF NOT EXISTS relationships (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                source_id UUID NOT NULL,
                source_type VARCHAR(50) NOT NULL,
                target_id UUID NOT NULL,
                target_type VARCHAR(50) NOT NULL,
                relationship_type VARCHAR(50) NOT NULL,
                weight FLOAT DEFAULT 1.0,
                confidence FLOAT DEFAULT 1.0,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                valid_from TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                valid_to TIMESTAMP WITH TIME ZONE
            );
            """,
            
            # Create indexes for relationships
            """
            CREATE INDEX IF NOT EXISTS idx_rels_source ON relationships(source_id, source_type);
            CREATE INDEX IF NOT EXISTS idx_rels_target ON relationships(target_id, target_type);
            CREATE INDEX IF NOT EXISTS idx_rels_type ON relationships(relationship_type);
            CREATE INDEX IF NOT EXISTS idx_rels_created ON relationships(created_at);
            """,
            
            # Vector index mapping (Qdrant point IDs)
            """
            CREATE TABLE IF NOT EXISTS vector_index (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                kernel_id UUID NOT NULL REFERENCES kernels(id) ON DELETE CASCADE,
                qdrant_point_id VARCHAR(100) NOT NULL,
                embedding_model VARCHAR(100) DEFAULT 'all-MiniLM-L6-v2',
                vector_dimensions INTEGER DEFAULT 384,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            
            # Create indexes for vector index
            """
            CREATE INDEX IF NOT EXISTS idx_vector_chunk ON vector_index(chunk_id);
            CREATE INDEX IF NOT EXISTS idx_vector_kernel ON vector_index(kernel_id);
            CREATE INDEX IF NOT EXISTS idx_vector_qdrant ON vector_index(qdrant_point_id);
            """,
            
            # Audit log table
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                entity_type VARCHAR(50) NOT NULL,
                entity_id UUID NOT NULL,
                action VARCHAR(50) NOT NULL,
                actor VARCHAR(255) DEFAULT 'system',
                old_values JSONB,
                new_values JSONB,
                ip_address INET,
                user_agent TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            
            # Create index for audit log
            """
            CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id);
            CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
            CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);
            """,
        ]
    
    @staticmethod
    def get_drop_statements() -> List[str]:
        """Generate SQL statements to drop the schema"""
        return [
            "DROP TABLE IF EXISTS audit_log CASCADE;",
            "DROP TABLE IF EXISTS vector_index CASCADE;",
            "DROP TABLE IF EXISTS relationships CASCADE;",
            "DROP TABLE IF EXISTS chunks CASCADE;",
            "DROP TABLE IF EXISTS documents CASCADE;",
            "DROP TABLE IF EXISTS kernel_versions CASCADE;",
            "DROP TABLE IF EXISTS kernels CASCADE;",
        ]


class PostgreSQLStorageBackend:
    """
    PostgreSQL backend for kernel storage
    
    Provides CRUD operations for kernels, documents, chunks,
    and version management with full audit trail support.
    """
    
    def __init__(
        self,
        connection_string: str = "postgresql://localhost:5432/kernel_db",
        pool_size: int = 10,
        max_overflow: int = 20,
    ):
        """
        Initialize PostgreSQL storage backend
        
        Args:
            connection_string: PostgreSQL connection URI
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
        """
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self._engine = None
        self._session_factory = None
    
    def _get_engine(self):
        """Get or create SQLAlchemy engine"""
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                from sqlalchemy.orm import sessionmaker
                
                self._engine = create_engine(
                    self.connection_string,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    pool_pre_ping=True,
                    echo=False,
                )
                
                self._session_factory = sessionmaker(
                    bind=self._engine,
                    autocommit=False,
                    autoflush=False,
                )
                
                logger.info(f"PostgreSQL engine created: {self.connection_string}")
            except ImportError:
                logger.error("SQLAlchemy not installed. Install with: pip install sqlalchemy psycopg2-binary")
                raise ImportError("SQLAlchemy required for PostgreSQL backend")
        
        return self._engine
    
    def _get_session(self):
        """Get a database session"""
        if self._session_factory is None:
            self._get_engine()
        
        return self._session_factory()
    
    @contextmanager
    def session(self):
        """Context manager for database sessions"""
        session = self._get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def initialize_schema(self) -> bool:
        """Initialize the database schema"""
        try:
            engine = self._get_engine()
            
            with engine.connect() as conn:
                for statement in KernelStorageSchema.get_create_statements():
                    conn.execute(statement)
                conn.commit()
            
            logger.info("Database schema initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            return False
    
    def drop_schema(self) -> bool:
        """Drop all tables (DANGER: data loss)"""
        try:
            engine = self._get_engine()
            
            with engine.connect() as conn:
                for statement in KernelStorageSchema.get_drop_statements():
                    conn.execute(statement)
                conn.commit()
            
            logger.warning("Database schema dropped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop schema: {e}")
            return False
    
    # Kernel Operations
    def create_kernel(self, metadata: KernelMetadata) -> str:
        """Create a new kernel and return its ID"""
        with self.session() as session:
            from sqlalchemy import text
            
            kernel_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            query = text("""
                INSERT INTO kernels (
                    id, name, description, owner_id, domain, tags,
                    schema_version, is_public, metadata, created_at, updated_at
                ) VALUES (
                    :id, :name, :description, :owner_id, :domain, :tags,
                    :schema_version, :is_public, :metadata, :created_at, :updated_at
                )
            """)
            
            session.execute(query, {
                "id": kernel_id,
                "name": metadata.name,
                "description": metadata.description,
                "owner_id": metadata.owner_id,
                "domain": metadata.domain,
                "tags": metadata.tags,
                "schema_version": metadata.schema_version,
                "is_public": metadata.is_public,
                "metadata": json.dumps(metadata.metadata_),
                "created_at": now,
                "updated_at": now,
            })
            
            logger.info(f"Created kernel: {kernel_id}")
            return kernel_id
    
    def get_kernel(self, kernel_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve kernel by ID"""
        with self.session() as session:
            from sqlalchemy import text
            
            query = text("""
                SELECT k.*, kv.id as version_id, kv.commit_message, kv.created_at as version_created
                FROM kernels k
                LEFT JOIN kernel_versions kv ON k.current_version_id = kv.id
                WHERE k.id = :id
            """)
            
            result = session.execute(query, {"id": kernel_id}).fetchone()
            
            if result:
                return dict(result._mapping)
            return None
    
    def update_kernel(self, kernel_id: str, updates: Dict[str, Any]) -> bool:
        """Update kernel metadata"""
        with self.session() as session:
            from sqlalchemy import text
            
            updates["updated_at"] = datetime.utcnow()
            updates_str = ", ".join([f"{k} = :{k}" for k in updates.keys()])
            
            query = text(f"UPDATE kernels SET {updates_str} WHERE id = :kernel_id")
            updates["kernel_id"] = kernel_id
            
            result = session.execute(query, updates)
            
            return result.rowcount > 0
    
    def delete_kernel(self, kernel_id: str) -> bool:
        """Delete a kernel and all its contents"""
        with self.session() as session:
            from sqlalchemy import text
            
            # Cascading delete will remove versions, documents, chunks, relationships
            query = text("DELETE FROM kernels WHERE id = :id")
            result = session.execute(query, {"id": kernel_id})
            
            logger.info(f"Deleted kernel: {kernel_id}")
            return result.rowcount > 0
    
    def list_kernels(
        self,
        owner_id: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List kernels with optional filtering"""
        with self.session() as session:
            from sqlalchemy import text
            
            conditions = []
            params = {"limit": limit, "offset": offset}
            
            if owner_id:
                conditions.append("k.owner_id = :owner_id")
                params["owner_id"] = owner_id
            
            if domain:
                conditions.append("k.domain = :domain")
                params["domain"] = domain
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = text(f"""
                SELECT k.*, 
                       (SELECT COUNT(*) FROM documents d WHERE d.kernel_id = k.id) as doc_count,
                       (SELECT COUNT(*) FROM chunks c WHERE c.kernel_id = k.id) as chunk_count
                FROM kernels k
                {where_clause}
                ORDER BY k.updated_at DESC
                LIMIT :limit OFFSET :offset
            """)
            
            results = session.execute(query, params).fetchall()
            return [dict(row._mapping) for row in results]
    
    # Version Operations
    def create_version(
        self,
        kernel_id: str,
        commit_message: str,
        content_hash: str,
        change_summary: Dict[str, Any],
        diff_summary: str = "",
        created_by: str = "system",
    ) -> str:
        """Create a new version of a kernel"""
        with self.session() as session:
            from sqlalchemy import text
            
            # Get current version to set as parent
            current_query = text("SELECT current_version_id FROM kernels WHERE id = :id")
            current = session.execute(current_query, {"id": kernel_id}).fetchone()
            parent_version_id = current.current_version_id if current else None
            
            version_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            # Create version
            version_query = text("""
                INSERT INTO kernel_versions (
                    id, kernel_id, parent_version_id, commit_message,
                    content_hash, change_summary, diff_summary, created_by, created_at
                ) VALUES (
                    :id, :kernel_id, :parent_version_id, :commit_message,
                    :content_hash, :change_summary, :diff_summary, :created_by, :created_at
                )
            """)
            
            session.execute(version_query, {
                "id": version_id,
                "kernel_id": kernel_id,
                "parent_version_id": parent_version_id,
                "commit_message": commit_message,
                "content_hash": content_hash,
                "change_summary": json.dumps(change_summary),
                "diff_summary": diff_summary,
                "created_by": created_by,
                "created_at": now,
            })
            
            # Update kernel's current version
            update_query = text("""
                UPDATE kernels SET current_version_id = :version_id, updated_at = :updated_at
                WHERE id = :kernel_id
            """)
            
            session.execute(update_query, {
                "version_id": version_id,
                "kernel_id": kernel_id,
                "updated_at": now,
            })
            
            logger.info(f"Created version {version_id} for kernel {kernel_id}")
            return version_id
    
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific version"""
        with self.session() as session:
            from sqlalchemy import text
            
            query = text("SELECT * FROM kernel_versions WHERE id = :id")
            result = session.execute(query, {"id": version_id}).fetchone()
            
            if result:
                return dict(result._mapping)
            return None
    
    def get_version_history(self, kernel_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get version history for a kernel"""
        with self.session() as session:
            from sqlalchemy import text
            
            query = text("""
                SELECT * FROM kernel_versions
                WHERE kernel_id = :kernel_id
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            
            results = session.execute(query, {"kernel_id": kernel_id, "limit": limit}).fetchall()
            return [dict(row._mapping) for row in results]
    
    def rollback_kernel(self, kernel_id: str, target_version_id: str) -> bool:
        """Rollback kernel to a previous version"""
        with self.session() as session:
            from sqlalchemy import text
            
            # Verify target version exists
            version_query = text("SELECT * FROM kernel_versions WHERE id = :id AND kernel_id = :kernel_id")
            version = session.execute(version_query, {
                "id": target_version_id,
                "kernel_id": kernel_id,
            }).fetchone()
            
            if not version:
                logger.error(f"Version {target_version_id} not found for kernel {kernel_id}")
                return False
            
            # Update kernel to point to target version
            now = datetime.utcnow()
            update_query = text("""
                UPDATE kernels SET current_version_id = :version_id, updated_at = :updated_at
                WHERE id = :kernel_id
            """)
            
            session.execute(update_query, {
                "version_id": target_version_id,
                "kernel_id": kernel_id,
                "updated_at": now,
            })
            
            logger.info(f"Rolled back kernel {kernel_id} to version {target_version_id}")
            return True
    
    # Document Operations
    def add_document(
        self,
        kernel_id: str,
        title: str,
        content_hash: str,
        blob_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_url: Optional[str] = None,
        author: Optional[str] = None,
        published_date: Optional[datetime] = None,
    ) -> str:
        """Add a document to a kernel"""
        with self.session() as session:
            from sqlalchemy import text
            
            document_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            query = text("""
                INSERT INTO documents (
                    id, kernel_id, title, content_hash, blob_path,
                    chunk_count, metadata, source_url, author, published_date,
                    created_at, updated_at
                ) VALUES (
                    :id, :kernel_id, :title, :content_hash, :blob_path,
                    0, :metadata, :source_url, :author, :published_date,
                    :created_at, :updated_at
                )
            """)
            
            session.execute(query, {
                "id": document_id,
                "kernel_id": kernel_id,
                "title": title,
                "content_hash": content_hash,
                "blob_path": blob_path,
                "metadata": json.dumps(metadata or {}),
                "source_url": source_url,
                "author": author,
                "published_date": published_date,
                "created_at": now,
                "updated_at": now,
            })
            
            logger.info(f"Added document {document_id} to kernel {kernel_id}")
            return document_id
    
    def update_document_chunks(self, document_id: str, chunk_count: int) -> bool:
        """Update chunk count after adding chunks"""
        with self.session() as session:
            from sqlalchemy import text
            
            query = text("""
                UPDATE documents SET chunk_count = :count, updated_at = :now
                WHERE id = :id
            """)
            
            result = session.execute(query, {"id": document_id, "count": chunk_count, "now": datetime.utcnow()})
            return result.rowcount > 0
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document"""
        with self.session() as session:
            from sqlalchemy import text
            
            query = text("SELECT * FROM documents WHERE id = :id")
            result = session.execute(query, {"id": document_id}).fetchone()
            
            if result:
                return dict(result._mapping)
            return None
    
    def get_documents_by_kernel(self, kernel_id: str) -> List[Dict[str, Any]]:
        """Get all documents in a kernel"""
        with self.session() as session:
            from sqlalchemy import text
            
            query = text("SELECT * FROM documents WHERE kernel_id = :kernel_id ORDER BY created_at")
            results = session.execute(query, {"kernel_id": kernel_id}).fetchall()
            return [dict(row._mapping) for row in results]
    
    # Chunk Operations
    def add_chunk(
        self,
        document_id: str,
        kernel_id: str,
        chunk_index: int,
        text_content: str,
        content_hash: str,
        qdrant_point_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: int = 0,
    ) -> str:
        """Add a chunk to a document"""
        with self.session() as session:
            from sqlalchemy import text
            
            chunk_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            query = text("""
                INSERT INTO chunks (
                    id, document_id, kernel_id, chunk_index, text_content,
                    content_hash, qdrant_point_id, token_count, metadata, created_at
                ) VALUES (
                    :id, :document_id, :kernel_id, :chunk_index, :text_content,
                    :content_hash, :qdrant_point_id, :token_count, :metadata, :created_at
                )
            """)
            
            session.execute(query, {
                "id": chunk_id,
                "document_id": document_id,
                "kernel_id": kernel_id,
                "chunk_index": chunk_index,
                "text_content": text_content,
                "content_hash": content_hash,
                "qdrant_point_id": qdrant_point_id,
                "token_count": token_count,
                "metadata": json.dumps(metadata or {}),
                "created_at": now,
            })
            
            return chunk_id
    
    def add_chunks_batch(self, chunks: List[Dict[str, Any]]) -> int:
        """Add multiple chunks in a batch"""
        with self.session() as session:
            from sqlalchemy import text
            
            if not chunks:
                return 0
            
            now = datetime.utcnow()
            
            query = text("""
                INSERT INTO chunks (
                    id, document_id, kernel_id, chunk_index, text_content,
                    content_hash, qdrant_point_id, token_count, metadata, created_at
                ) VALUES (
                    :id, :document_id, :kernel_id, :chunk_index, :text_content,
                    :content_hash, :qdrant_point_id, :token_count, :metadata, :created_at
                )
            """)
            
            for chunk in chunks:
                chunk["id"] = chunk.get("id", str(uuid.uuid4()))
                chunk["created_at"] = now
                if chunk.get("metadata"):
                    chunk["metadata"] = json.dumps(chunk["metadata"])
                
                session.execute(query, chunk)
            
            return len(chunks)
    
    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        with self.session() as session:
            from sqlalchemy import text
            
            query = text("""
                SELECT * FROM chunks
                WHERE document_id = :document_id
                ORDER BY chunk_index
            """)
            
            results = session.execute(query, {"document_id": document_id}).fetchall()
            return [dict(row._mapping) for row in results]
    
    def get_chunks_by_kernel(self, kernel_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get chunks for a kernel"""
        with self.session() as session:
            from sqlalchemy import text
            
            query = text("""
                SELECT c.*, d.title as document_title
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.kernel_id = :kernel_id
                ORDER BY c.created_at
                LIMIT :limit
            """)
            
            results = session.execute(query, {"kernel_id": kernel_id, "limit": limit}).fetchall()
            return [dict(row._mapping) for row in results]
    
    # Relationship Operations
    def add_relationship(
        self,
        source_id: str,
        source_type: EntityType,
        target_id: str,
        target_type: EntityType,
        relationship_type: RelationshipType,
        weight: float = 1.0,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a relationship between entities"""
        with self.session() as session:
            from sqlalchemy import text
            
            relationship_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            query = text("""
                INSERT INTO relationships (
                    id, source_id, source_type, target_id, target_type,
                    relationship_type, weight, confidence, metadata, created_at
                ) VALUES (
                    :id, :source_id, :source_type, :target_id, :target_type,
                    :relationship_type, :weight, :confidence, :metadata, :created_at
                )
            """)
            
            session.execute(query, {
                "id": relationship_id,
                "source_id": source_id,
                "source_type": source_type.value,
                "target_id": target_id,
                "target_type": target_type.value,
                "relationship_type": relationship_type.value,
                "weight": weight,
                "confidence": confidence,
                "metadata": json.dumps(metadata or {}),
                "created_at": now,
            })
            
            return relationship_id
    
    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",  # "outgoing", "incoming", "both"
        relationship_type: Optional[RelationshipType] = None,
    ) -> List[Dict[str, Any]]:
        """Get relationships for an entity"""
        with self.session() as session:
            from sqlalchemy import text
            
            if direction == "outgoing":
                query = text("SELECT * FROM relationships WHERE source_id = :id")
            elif direction == "incoming":
                query = text("SELECT * FROM relationships WHERE target_id = :id")
            else:
                query = text("""
                    SELECT * FROM relationships
                    WHERE source_id = :id OR target_id = :id
                """)
            
            results = session.execute(query, {"id": entity_id}).fetchall()
            return [dict(row._mapping) for row in results]
    
    def find_related_entities(
        self,
        entity_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_confidence: float = 0.5,
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find related entities using graph traversal"""
        # This is a simplified version - production would use proper graph traversal
        with self.session() as session:
            from sqlalchemy import text
            
            rel_types = relationship_types or [r.value for r in RelationshipType]
            
            query = text("""
                SELECT r.*,
                       s.name as source_name, t.name as target_name
                FROM relationships r
                LEFT JOIN chunks s ON r.source_id = s.id
                LEFT JOIN chunks t ON r.target_id = t.id
                WHERE (r.source_id = :id OR r.target_id = :id)
                AND r.confidence >= :min_confidence
                AND r.relationship_type = ANY(:rel_types)
                ORDER BY r.confidence DESC
                LIMIT 100
            """)
            
            results = session.execute(query, {
                "id": entity_id,
                "min_confidence": min_confidence,
                "rel_types": rel_types,
            }).fetchall()
            
            return [dict(row._mapping) for row in results]
    
    # Vector Index Operations
    def link_vector_to_chunk(
        self,
        chunk_id: str,
        kernel_id: str,
        qdrant_point_id: str,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> str:
        """Link a Qdrant vector to a chunk"""
        with self.session() as session:
            from sqlalchemy import text
            
            index_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            query = text("""
                INSERT INTO vector_index (
                    id, chunk_id, kernel_id, qdrant_point_id,
                    embedding_model, created_at
                ) VALUES (
                    :id, :chunk_id, :kernel_id, :qdrant_point_id,
                    :embedding_model, :created_at
                )
            """)
            
            session.execute(query, {
                "id": index_id,
                "chunk_id": chunk_id,
                "kernel_id": kernel_id,
                "qdrant_point_id": qdrant_point_id,
                "embedding_model": embedding_model,
                "created_at": now,
            })
            
            return index_id
    
    def get_chunk_by_vector(self, qdrant_point_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by Qdrant point ID"""
        with self.session() as session:
            from sqlalchemy import text
            
            query = text("""
                SELECT c.*, v.embedding_model, v.created_at as vector_created
                FROM chunks c
                JOIN vector_index v ON c.id = v.chunk_id
                WHERE v.qdrant_point_id = :point_id
            """)
            
            result = session.execute(query, {"point_id": qdrant_point_id}).fetchone()
            
            if result:
                return dict(result._mapping)
            return None
    
    # Audit Operations
    def log_action(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        actor: str = "system",
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        """Log an action to the audit trail"""
        with self.session() as session:
            from sqlalchemy import text
            
            audit_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            query = text("""
                INSERT INTO audit_log (
                    id, entity_type, entity_id, action, actor,
                    old_values, new_values, ip_address, user_agent, created_at
                ) VALUES (
                    :id, :entity_type, :entity_id, :action, :actor,
                    :old_values, :new_values, :ip_address, :user_agent, :created_at
                )
            """)
            
            session.execute(query, {
                "id": audit_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": action,
                "actor": actor,
                "old_values": json.dumps(old_values or {}),
                "new_values": json.dumps(new_values or {}),
                "ip_address": ip_address,
                "user_agent": user_agent,
                "created_at": now,
            })
            
            return audit_id
    
    def get_audit_log(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query the audit log"""
        with self.session() as session:
            from sqlalchemy import text
            
            conditions = []
            params = {"limit": limit}
            
            if entity_type:
                conditions.append("entity_type = :entity_type")
                params["entity_type"] = entity_type
            
            if entity_id:
                conditions.append("entity_id = :entity_id")
                params["entity_id"] = entity_id
            
            if action:
                conditions.append("action = :action")
                params["action"] = action
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = text(f"""
                SELECT * FROM audit_log
                {where_clause}
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            
            results = session.execute(query, params).fetchall()
            return [dict(row._mapping) for row in results]
    
    # Statistics
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with self.session() as session:
            from sqlalchemy import text
            
            stats = {}
            
            # Count tables
            tables = ["kernels", "documents", "chunks", "kernel_versions", "relationships", "vector_index"]
            for table in tables:
                query = text(f"SELECT COUNT(*) FROM {table}")
                result = session.execute(query).fetchone()
                stats[f"{table}_count"] = result[0]
            
            return stats


class PostgresKernelStorage:
    """
    Simplified kernel storage interface for the verification script.
    
    This class provides a convenient wrapper around PostgreSQLStorageBackend
    with a simplified API for basic kernel operations.
    """
    
    def __init__(self, db_url: str = "postgresql://localhost:5432/kernel_db"):
        """
        Initialize the kernel storage.
        
        Args:
            db_url: Database connection URL. For testing, use "sqlite:///:memory:"
        """
        from sqlalchemy import create_engine
        
        self.backend = PostgreSQLStorageBackend(connection_string=db_url)
        self.db_url = db_url
        
        # For SQLite, we need to use raw SQL operations since PostgreSQL-specific
        # features like uuid-ossp extension are not available
        self._is_sqlite = "sqlite" in db_url
        
        # Create a shared engine to avoid SQLite in-memory database issues
        self._engine = create_engine(db_url)
        
        # Initialize tables once
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Ensure tables exist"""
        from sqlalchemy import text
        
        with self._engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS kernels (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS kernel_versions (
                    id TEXT PRIMARY KEY,
                    kernel_id TEXT NOT NULL,
                    parent_version_id TEXT,
                    commit_message TEXT,
                    content_hash TEXT,
                    created_at TIMESTAMP,
                    created_by TEXT
                )
            """))
            conn.commit()
    
    def create_kernel(self, name: str, description: str = "") -> str:
        """Create a new kernel and return its ID."""
        from sqlalchemy import text
        from datetime import datetime
        
        kernel_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        with self._engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO kernels (id, name, description, created_at, updated_at)
                VALUES (:id, :name, :description, :created_at, :updated_at)
            """), {
                "id": kernel_id,
                "name": name,
                "description": description,
                "created_at": now,
                "updated_at": now,
            })
            conn.commit()
        
        return kernel_id
    
    def create_version(self, kernel_id: str, content: Dict[str, Any]) -> str:
        """Create a new version of a kernel."""
        from sqlalchemy import text
        from datetime import datetime
        import hashlib
        
        version_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Calculate content hash
        content_str = json.dumps(content, sort_keys=True)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        
        with self._engine.connect() as conn:
            # Get parent version
            result = conn.execute(text("""
                SELECT id FROM kernel_versions
                WHERE kernel_id = :kernel_id
                ORDER BY created_at DESC
                LIMIT 1
            """), {"kernel_id": kernel_id}).fetchone()
            
            parent_version_id = result[0] if result else None
            
            conn.execute(text("""
                INSERT INTO kernel_versions (id, kernel_id, parent_version_id, commit_message, content_hash, created_at, created_by)
                VALUES (:id, :kernel_id, :parent_version_id, :commit_message, :content_hash, :created_at, :created_by)
            """), {
                "id": version_id,
                "kernel_id": kernel_id,
                "parent_version_id": parent_version_id,
                "commit_message": "Version update",
                "content_hash": content_hash,
                "created_at": now,
                "created_by": "system",
            })
            conn.commit()
        
        return version_id
    
    def list_versions(self, kernel_id: str) -> List[KernelVersion]:
        """List all versions of a kernel."""
        from sqlalchemy import text
        
        with self._engine.connect() as conn:
            results = conn.execute(text("""
                SELECT id, kernel_id, parent_version_id, commit_message, 
                       content_hash, created_at, created_by
                FROM kernel_versions
                WHERE kernel_id = :kernel_id
                ORDER BY created_at DESC
            """), {"kernel_id": kernel_id}).fetchall()
            
            versions = []
            for row in results:
                versions.append(KernelVersion(
                    version_id=row[0],
                    kernel_id=row[1],
                    parent_version_id=row[2],
                    commit_message=row[3],
                    created_at=row[5],
                    created_by=row[6],
                    content_hash=row[4],
                ))
            
            return versions
    
    def get_latest_version(self, kernel_id: str) -> Optional[KernelVersion]:
        """Get the latest version of a kernel."""
        from sqlalchemy import text
        
        with self._engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, kernel_id, parent_version_id, commit_message,
                       content_hash, created_at, created_by
                FROM kernel_versions
                WHERE kernel_id = :kernel_id
                ORDER BY created_at DESC
                LIMIT 1
            """), {"kernel_id": kernel_id}).fetchone()
            
            if result:
                return KernelVersion(
                    version_id=result[0],
                    kernel_id=result[1],
                    parent_version_id=result[2],
                    commit_message=result[3],
                    created_at=result[5],
                    created_by=result[6],
                    content_hash=result[4],
                )
            
            return None
