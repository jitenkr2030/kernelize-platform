# KERNELIZE API Documentation

## Overview

The KERNELIZE API provides access to the world's first Knowledge Compression Infrastructure, enabling 100×-10,000× compression of knowledge with zero semantic loss.

**Base URL:** `https://api.kernelize.com/v2`

**Authentication:** Bearer token in Authorization header

**Rate Limits:**
- Free Tier: 100 requests/hour
- Pro Tier: 10,000 requests/hour  
- Enterprise: Custom limits

---

## Authentication

### API Key Management

```bash
# Generate API key
curl -X POST https://api.kernelize.com/v2/auth/keys \
  -H "Authorization: Bearer sk-ker_existing_key" \
  -H "Content-Type: application/json"

# Response
{
  "api_key": "sk-ker_xxxxxxxxxxxxxxxxxxxx",
  "created_at": "2025-12-10T14:01:49Z",
  "last_used": "2025-12-10T14:01:49Z",
  "rate_limit": "10000/hour"
}
```

### Headers

```bash
Authorization: Bearer sk-ker_xxxxxxxxxxxxxxxxxxxx
Content-Type: application/json
X-Request-ID: unique-request-identifier
```

---

## Core Endpoints

### 1. Compression API

#### `POST /v2/compress`

Compress raw knowledge into semantic intelligence kernels.

**Request Body:**
```json
{
  "input": "string (required) - Raw text, URL, or base64 data",
  "input_type": "string (required) - text|url|file|audio|video",
  "compression_level": "string - standard|high|ultra",
  "preserve_context": "boolean - Maintain contextual relationships",
  "domain": "string - Knowledge domain classification",
  "options": {
    "maintain_causality": "boolean - Preserve cause-effect relationships",
    "preserve_relationships": "boolean - Keep semantic connections",
    "extract_patterns": "boolean - Identify recurring patterns"
  }
}
```

**Response:**
```json
{
  "kernel_id": "string - Unique kernel identifier",
  "compression_ratio": "number - Achieved compression ratio",
  "original_size": "number - Original data size in bytes",
  "compressed_size": "number - Compressed size in KB",
  "semantic_fidelity": "number - Semantic preservation percentage (0-100)",
  "processing_time_ms": "number - Processing time in milliseconds",
  "kernel_data": {
    "concepts": "array - Extracted semantic concepts",
    "causal_chains": "array - Cause-effect relationship chains",
    "temporal_relationships": "array - Time-based connections",
    "spatial_relationships": "array - Location-based connections"
  },
  "api_usage": {
    "tokens_used": "number - Processing tokens consumed",
    "cost_usd": "number - Cost in USD"
  }
}
```

**Example:**
```bash
curl -X POST https://api.kernelize.com/v2/compress \
  -H "Authorization: Bearer sk-ker_xxxxxxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Climate change causes global temperature rise...",
    "input_type": "text",
    "compression_level": "ultra",
    "preserve_context": true,
    "domain": "environmental_science",
    "options": {
      "maintain_causality": true,
      "preserve_relationships": true,
      "extract_patterns": true
    }
  }'
```

---

### 2. Query API

#### `GET /v2/kernels/{kernel_id}/query`

Query compressed knowledge without decompression.

**Query Parameters:**
```json
{
  "query": "string (required) - Search query",
  "query_type": "string - semantic|causal|temporal|spatial",
  "max_results": "number - Maximum results (default: 10)",
  "confidence_threshold": "number - Minimum confidence (0-1)",
  "include_explanations": "boolean - Include reasoning explanations",
  "response_format": "string - json|xml|markdown"
}
```

**Response:**
```json
{
  "query_id": "string - Unique query identifier",
  "execution_time_ms": "number - Query execution time",
  "results": [
    {
      "answer": "string - Direct answer to query",
      "confidence": "number - Confidence score (0-1)",
      "supporting_evidence": "number - Number of supporting sources",
      "concept_matches": "array - Matched concept IDs",
      "explanation": "string - Reasoning explanation",
      "related_concepts": "array - Related concept suggestions",
      "source_citations": "array - Original source references"
    }
  ],
  "metadata": {
    "kernel_version": "string - Kernel version used",
    "sources_checked": "number - Total sources examined",
    "semantic_coverage": "number - Coverage percentage",
    "query_complexity": "string - simple|moderate|complex"
  }
}
```

**Example:**
```bash
curl "https://api.kernelize.com/v2/kernels/ker_4a7b2c8d9e1f3g6h/query?query=What%20causes%20climate%20change&query_type=semantic&max_results=5" \
  -H "Authorization: Bearer sk-ker_xxxxxxxxxxxxxxxxxxxx"
```

---

### 3. Merge API

#### `POST /v2/kernels/{kernel_id}/merge`

Incrementally update kernels without recompression.

**Request Body:**
```json
{
  "new_knowledge": "string (required) - New knowledge to merge",
  "merge_strategy": "string - incremental|replacement|hybrid",
  "conflict_resolution": "string - latest_source_priority|confidence_based|manual",
  "preserve_history": "boolean - Maintain merge history",
  "merge_validation": "boolean - Validate against existing knowledge"
}
```

**Response:**
```json
{
  "merge_id": "string - Unique merge identifier",
  "merge_time_ms": "number - Merge execution time",
  "conflicts_resolved": "number - Number of conflicts resolved",
  "new_concepts_added": "number - New concepts integrated",
  "relationships_updated": "number - Updated relationships",
  "kernel_version": "string - New kernel version",
  "backward_compatibility": "boolean - Maintains API compatibility",
  "validation_results": {
    "consistency_score": "number - Knowledge consistency check",
    "contradictions_found": "number - Logical contradictions detected",
    "quality_improvement": "number - Overall quality delta"
  }
}
```

---

### 4. Distillation API

#### `POST /v2/kernels/{kernel_id}/distill`

Inject kernels into LLMs for offline intelligence.

**Request Body:**
```json
{
  "target_model": "string (required) - LLM model identifier",
  "distillation_method": "string - knowledge_injection|weight_merging|fine_tuning",
  "performance_target": "string - maintain_accuracy|maximize_compression|balance",
  "deployment_target": "string - mobile|edge|cloud|embedded",
  "quantization_level": "string - none|int8|int4|binary",
  "batch_size": "number - Processing batch size"
}
```

**Response:**
```json
{
  "distillation_id": "string - Unique distillation identifier",
  "distillation_time_min": "number - Time taken in minutes",
  "model_size_reduction": "number - Percentage size reduction",
  "performance_improvement": "number - Inference speed improvement %",
  "distilled_model_path": "string - Path to distilled model",
  "inference_speedup": "number - Inference speed multiplier",
  "accuracy_retention": "number - Accuracy preservation %",
  "memory_reduction": "number - Memory usage reduction %",
  "deployment_artifacts": {
    "model_file": "string - Model file path",
    "config_file": "string - Configuration file",
    "performance_benchmarks": "object - Benchmark results"
  }
}
```

---

### 5. Marketplace API

#### `GET /v2/marketplace/kernels`

Browse available pre-built domain kernels.

**Query Parameters:**
```json
{
  "domain": "string - Filter by knowledge domain",
  "min_rating": "number - Minimum rating filter",
  "sort": "string - popularity|rating|price|recent",
  "price_range": "string - 0-1000|1000-5000|5000+",
  "compression_ratio": "number - Minimum compression ratio",
  "page": "number - Page number",
  "limit": "number - Results per page"
}
```

**Response:**
```json
{
  "kernels": [
    {
      "kernel_id": "string - Kernel identifier",
      "name": "string - Human-readable name",
      "domain": "string - Knowledge domain",
      "description": "string - Detailed description",
      "size_mb": "number - Compressed size in MB",
      "compression_ratio": "number - Achieved compression ratio",
      "rating": "number - User rating (1-5)",
      "review_count": "number - Number of reviews",
      "price_monthly_usd": "number - Monthly subscription price",
      "coverage": "string - Scope of knowledge covered",
      "last_updated": "string - Last update timestamp",
      "version": "string - Current version",
      "supported_queries": "array - Supported query types",
      "sample_queries": "array - Example queries"
    }
  ],
  "pagination": {
    "page": "number - Current page",
    "total_pages": "number - Total pages",
    "total_kernels": "number - Total kernels",
    "has_next": "boolean - Has next page"
  }
}
```

#### `GET /v2/marketplace/kernels/{kernel_id}/purchase`

Purchase kernel subscription.

**Response:**
```json
{
  "purchase_id": "string - Purchase identifier",
  "kernel_id": "string - Purchased kernel ID",
  "subscription": {
    "status": "string - active|pending|cancelled",
    "start_date": "string - Subscription start",
    "end_date": "string - Subscription end",
    "auto_renew": "boolean - Auto-renewal enabled",
    "monthly_cost": "number - Monthly cost in USD"
  },
  "access_credentials": {
    "api_endpoint": "string - Kernel API endpoint",
    "access_token": "string - Kernel-specific access token",
    "permissions": "array - Available operations"
  }
}
```

---

### 6. Analytics API

#### `GET /v2/analytics/usage`

Monitor API usage and performance metrics.

**Query Parameters:**
```json
{
  "period": "string - 1h|24h|7d|30d|90d|custom",
  "start_date": "string - Custom start date (ISO 8601)",
  "end_date": "string - Custom end date (ISO 8601)",
  "metrics": "string - requests|cost|performance|errors|all",
  "granularity": "string - hourly|daily|weekly"
}
```

**Response:**
```json
{
  "period": "string - Reporting period",
  "generated_at": "string - Report generation timestamp",
  "total_requests": "number - Total API requests",
  "total_cost_usd": "number - Total cost in USD",
  "average_response_time_ms": "number - Average response time",
  "success_rate": "number - Success rate percentage",
  "error_rate": "number - Error rate percentage",
  "breakdown": {
    "compress_requests": "number - Compression API calls",
    "query_requests": "number - Query API calls",
    "merge_requests": "number - Merge API calls",
    "distill_requests": "number - Distillation API calls"
  },
  "cost_breakdown": {
    "compression": "number - Compression costs",
    "queries": "number - Query costs",
    "storage": "number - Storage costs",
    "distillation": "number - Distillation costs"
  },
  "performance_metrics": {
    "p50_response_time_ms": "number - 50th percentile response time",
    "p95_response_time_ms": "number - 95th percentile response time",
    "p99_response_time_ms": "number - 99th percentile response time",
    "throughput_rps": "number - Requests per second"
  },
  "usage_trends": [
    {
      "timestamp": "string - Time bucket",
      "requests": "number - Request count",
      "cost": "number - Cost in this period"
    }
  ]
}
```

---

## Error Handling

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (invalid API key)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found (kernel doesn't exist)
- `409` - Conflict (merge conflict)
- `422` - Unprocessable Entity (validation error)
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error
- `503` - Service Unavailable (maintenance)

### Error Response Format

```json
{
  "error": {
    "code": "string - Machine-readable error code",
    "message": "string - Human-readable error message",
    "details": "object - Additional error details",
    "request_id": "string - Request ID for support",
    "timestamp": "string - Error timestamp"
  }
}
```

### Common Error Codes

```json
{
  "INVALID_API_KEY": "The provided API key is invalid or expired",
  "RATE_LIMIT_EXCEEDED": "API rate limit exceeded for your plan",
  "KERNEL_NOT_FOUND": "The specified kernel does not exist",
  "INSUFFICIENT_CREDITS": "Insufficient credits for the requested operation",
  "COMPRESSION_FAILED": "Knowledge compression failed due to invalid input",
  "QUERY_TIMEOUT": "Query execution exceeded time limit",
  "MERGE_CONFLICT": "Merge operation encountered unresolved conflicts",
  "DISTILLATION_FAILED": "Model distillation failed",
  "INVALID_DOMAIN": "Specified knowledge domain is not supported"
}
```

---

## SDKs and Libraries

### Python SDK

```bash
pip install kernelize-python
```

```python
from kernelize import KernelizeClient

client = KernelizeClient(api_key="sk-ker_xxxxxxxxxxxxxxxxxxxx")

# Compress knowledge
result = client.compress(
    input="Climate change is caused by greenhouse gas emissions...",
    input_type="text",
    compression_level="ultra",
    domain="environmental_science"
)

print(f"Compression ratio: {result.compression_ratio}")
print(f"Kernel ID: {result.kernel_id}")

# Query kernel
response = client.query(
    kernel_id=result.kernel_id,
    query="What are the main causes of global warming?",
    query_type="semantic"
)

print(f"Answer: {response.results[0].answer}")
print(f"Confidence: {response.results[0].confidence}")
```

### JavaScript SDK

```bash
npm install @kernelize/javascript-sdk
```

```javascript
import { KernelizeClient } from '@kernelize/javascript-sdk';

const client = new KernelizeClient({
  apiKey: 'sk-ker_xxxxxxxxxxxxxxxxxxxx'
});

// Compress knowledge
const result = await client.compress({
  input: 'Climate change is caused by greenhouse gas emissions...',
  inputType: 'text',
  compressionLevel: 'ultra',
  domain: 'environmental_science'
});

console.log(`Compression ratio: ${result.compressionRatio}`);
console.log(`Kernel ID: ${result.kernelId}`);

// Query kernel
const response = await client.query({
  kernelId: result.kernelId,
  query: 'What are the main causes of global warming?',
  queryType: 'semantic'
});

console.log(`Answer: ${response.results[0].answer}`);
console.log(`Confidence: ${response.results[0].confidence}`);
```

### Go SDK

```bash
go get github.com/kernelize/go-sdk
```

```go
package main

import (
    "fmt"
    "github.com/kernelize/go-sdk"
)

func main() {
    client := kernelize.NewClient("sk-ker_xxxxxxxxxxxxxxxxxxxx")
    
    // Compress knowledge
    result, err := client.Compress(kernelize.CompressRequest{
        Input: "Climate change is caused by greenhouse gas emissions...",
        InputType: "text",
        CompressionLevel: "ultra",
        Domain: "environmental_science",
    })
    
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Compression ratio: %f\n", result.CompressionRatio)
    fmt.Printf("Kernel ID: %s\n", result.KernelID)
    
    // Query kernel
    response, err := client.Query(kernelize.QueryRequest{
        KernelID: result.KernelID,
        Query: "What are the main causes of global warming?",
        QueryType: "semantic",
    })
    
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Answer: %s\n", response.Results[0].Answer)
    fmt.Printf("Confidence: %f\n", response.Results[0].Confidence)
}
```

---

## Rate Limiting

### Tier Limits

**Free Tier:**
- 100 requests/hour
- Basic compression (up to 1MB)
- Standard query speed
- Community support

**Pro Tier ($99/month):**
- 10,000 requests/hour
- Advanced compression (up to 10MB)
- Priority query speed
- Email support
- Custom domains

**Enterprise Tier ($999/month):**
- Unlimited requests
- Unlimited compression size
- Dedicated infrastructure
- Phone support
- Custom integrations
- SLA guarantees

### Rate Limit Headers

```bash
X-RateLimit-Limit: 10000
X-RateLimit-Remaining: 9999
X-RateLimit-Reset: 1640995200
X-RateLimit-Retry-After: 3600
```

---

## Webhooks

### Setup Webhooks

```bash
curl -X POST https://api.kernelize.com/v2/webhooks \
  -H "Authorization: Bearer sk-ker_xxxxxxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-app.com/webhooks/kernelize",
    "events": ["kernel.compressed", "kernel.updated", "query.completed"],
    "secret": "your-webhook-secret"
  }'
```

### Webhook Events

**kernel.compressed:**
```json
{
  "event": "kernel.compressed",
  "kernel_id": "ker_4a7b2c8d9e1f3g6h",
  "compression_ratio": 1247.3,
  "timestamp": "2025-12-10T14:01:49Z"
}
```

**query.completed:**
```json
{
  "event": "query.completed",
  "query_id": "qry_8b3c2d1e9f5g7h2a",
  "kernel_id": "ker_4a7b2c8d9e1f3g6h",
  "execution_time_ms": 0.07,
  "timestamp": "2025-12-10T14:01:49Z"
}
```

---

## Testing

### Sandbox Environment

**Base URL:** `https://sandbox-api.kernelize.com/v2`

- No real charges
- Limited compression ratios (max 100x)
- Synthetic data only
- Fast response times for testing

### Postman Collection

Import our Postman collection for easy testing:

```bash
curl -L https://api.kernelize.com/v2/docs/postman-collection.json -o kernelize.postman_collection.json
```

### Test Data

Use these sample inputs for testing:

```json
{
  "climate_text": "Climate change is primarily caused by greenhouse gas emissions from human activities such as burning fossil fuels, deforestation, and industrial processes.",
  "medical_text": "Type 2 diabetes is a metabolic disorder characterized by insulin resistance and high blood sugar levels.",
  "financial_text": "The stock market experienced a 15% decline following the Federal Reserve's interest rate decision.",
  "legal_text": "Constitutional law establishes the fundamental principles and structure of government and protects individual rights."
}
```

---

## Support

- **Documentation:** https://docs.kernelize.com
- **API Status:** https://status.kernelize.com
- **Support Email:** support@kernelize.com
- **Community Forum:** https://community.kernelize.com
- **GitHub:** https://github.com/kernelize/api

---

*For enterprise support, custom integrations, or partnership inquiries, contact enterprise@kernelize.com*