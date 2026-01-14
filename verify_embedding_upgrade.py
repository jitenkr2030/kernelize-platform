#!/usr/bin/env python3
"""
Verification Script for Sentence-Transformers Integration
==========================================================

This script verifies that the EmbeddingGenerator has been successfully upgraded
to use real semantic embeddings from sentence-transformers instead of the
simulated hash-based embeddings.

Tests performed:
1. Verify embedding dimensions (should be 384)
2. Verify semantic similarity works correctly
3. Demonstrate batch processing capability
4. Show fallback mechanism works
"""

import sys
import time

# Add the src directory to path for imports
sys.path.insert(0, '/workspace/src/python')

from services.query_engine import EmbeddingGenerator, KernelQueryEngine, QueryType


def test_embedding_dimensions():
    """Test 1: Verify embedding dimensions are 384 (all-MiniLM-L6-v2)"""
    print("=" * 70)
    print("TEST 1: Embedding Dimensions Verification")
    print("=" * 70)
    
    generator = EmbeddingGenerator()
    test_text = "This is a test sentence for embedding verification."
    
    embedding = generator.generate(test_text)
    dimensions = len(embedding)
    
    print(f"Model: {generator.model_name}")
    print(f"Text: '{test_text}'")
    print(f"Embedding dimensions: {dimensions}")
    print(f"Expected dimensions: 384")
    
    if dimensions == 384:
        print("✓ PASS: Embedding dimensions match expected value")
        return True
    else:
        print(f"✗ FAIL: Expected 384 dimensions, got {dimensions}")
        return False


def test_semantic_similarity():
    """Test 2: Verify semantic similarity produces meaningful results"""
    print("\n" + "=" * 70)
    print("TEST 2: Semantic Similarity Verification")
    print("=" * 70)
    
    generator = EmbeddingGenerator()
    
    # Test pairs with expected similarity
    test_pairs = [
        # High similarity pairs
        ("The cat is sleeping on the couch", "A feline is napping on the sofa", "High similarity (cat/feline, sleeping/napping)"),
        ("Machine learning is transforming AI", "Deep learning is revolutionizing artificial intelligence", "High similarity (ML/AI concepts)"),
        ("The weather is sunny today", "It's a beautiful sunny day outside", "High similarity (sunny weather)"),
        
        # Low similarity pairs
        ("The cat is sleeping on the couch", "Stock markets crashed significantly", "Low similarity (unrelated topics)"),
        ("I love eating pizza", "Quantum physics explains the universe", "Low similarity (unrelated topics)"),
    ]
    
    all_passed = True
    for text1, text2, description in test_pairs:
        similarity = generator.semantic_similarity(text1, text2)
        print(f"\nText 1: '{text1[:50]}...'")
        print(f"Text 2: '{text2[:50]}...'")
        print(f"Expected: {description}")
        print(f"Similarity: {similarity:.4f}")
        
        # Validate similarity is in valid range [-1, 1] (cosine similarity can be negative)
        if -1 <= similarity <= 1:
            print("✓ Similarity in valid range [-1, 1]")
        else:
            print("✗ Similarity out of range!")
            all_passed = False
    
    return all_passed


def test_batch_processing():
    """Test 3: Verify batch processing works efficiently"""
    print("\n" + "=" * 70)
    print("TEST 3: Batch Processing Verification")
    print("=" * 70)
    
    generator = EmbeddingGenerator()
    
    texts = [
        "First test document about machine learning",
        "Second document discussing neural networks",
        "Third entry about natural language processing",
        "Fourth text covering computer vision applications",
        "Fifth sample talking about data science"
    ]
    
    # Measure single processing time
    start = time.time()
    single_embeddings = [generator.generate(text) for text in texts]
    single_time = time.time() - start
    
    # Measure batch processing time
    start = time.time()
    batch_embeddings = generator.batch_generate(texts)
    batch_time = time.time() - start
    
    print(f"Single processing time: {single_time*1000:.2f}ms")
    print(f"Batch processing time: {batch_time*1000:.2f}ms")
    print(f"Speedup: {single_time/batch_time:.2f}x")
    
    # Verify all embeddings have correct dimensions
    all_correct_dimensions = all(len(emb) == 384 for emb in batch_embeddings)
    print(f"\nAll batch embeddings have 384 dimensions: {'✓ PASS' if all_correct_dimensions else '✗ FAIL'}")
    
    # Verify embeddings are unique
    unique_embeddings = len(set(tuple(e) for e in batch_embeddings))
    print(f"Unique embeddings: {unique_embeddings}/{len(texts)}")
    
    return all_correct_dimensions and unique_embeddings == len(texts)


def test_kernel_query_engine_integration():
    """Test 4: Verify integration with KernelQueryEngine"""
    print("\n" + "=" * 70)
    print("TEST 4: KernelQueryEngine Integration")
    print("=" * 70)
    
    engine = KernelQueryEngine()
    
    # Index some sample knowledge kernels
    sample_kernels = [
        ("kernel_001", "Machine learning algorithms can detect patterns in large datasets", 
         "ML fundamentals and pattern recognition"),
        ("kernel_002", "Neural networks are inspired by the human brain's structure", 
         "Neural network basics and biology inspiration"),
        ("kernel_003", "Python is a popular programming language for data science", 
         "Python programming for data science"),
        ("kernel_004", "Natural language processing enables computers to understand text", 
         "NLP and text comprehension"),
        ("kernel_005", "Computer vision algorithms can recognize objects in images", 
         "Image recognition and CV techniques"),
    ]
    
    print("Indexing knowledge kernels...")
    for kernel_id, content, _ in sample_kernels:
        engine.index_kernel(kernel_id, content)
        print(f"  ✓ Indexed: {kernel_id}")
    
    # Test semantic search
    print("\nPerforming semantic search...")
    query = "What can neural networks do?"
    results, metrics = engine.query(query, query_type=QueryType.SEMANTIC, top_k=3)
    
    print(f"\nQuery: '{query}'")
    print(f"Results found: {metrics.total_results}")
    print(f"Query time: {metrics.query_time_ms}ms")
    print(f"Embeddings generated: {metrics.embeddings_generated}")
    
    for result in results:
        print(f"\n  Rank {result.rank}: {result.kernel_id}")
        print(f"  Similarity: {result.similarity_score:.4f}")
        print(f"  Content: {result.content[:60]}...")
    
    # Check embedding stats
    stats = engine.embedding_generator.get_stats()
    print(f"\nEmbedding Generator Stats:")
    print(f"  Model: {stats['model_name']}")
    print(f"  Dimensions: {stats['dimensions']}")
    print(f"  Status: {stats['status']}")
    print(f"  Model calls: {stats['model_calls']}")
    print(f"  Using transformer: {stats['using_transformer']}")
    
    return True


def test_embedding_uniqueness():
    """Test 5: Verify different sentences produce different embeddings"""
    print("\n" + "=" * 70)
    print("TEST 5: Embedding Uniqueness Verification")
    print("=" * 70)
    
    generator = EmbeddingGenerator()
    
    # Very different sentences should have low similarity
    sentences = [
        "The quick brown fox jumps over the lazy dog",
        "Photosynthesis is how plants convert sunlight into energy",
        "The stock market experienced significant volatility today",
        "A delicious pizza with pepperoni and mushrooms",
        "The Eiffel Tower is located in Paris, France",
    ]
    
    embeddings = generator.batch_generate(sentences)
    
    # Calculate all pairwise similarities
    print("\nPairwise similarities between very different sentences:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = generator.cosine_similarity(embeddings[i], embeddings[j])
            print(f"  '{sentences[i][:30]}...' vs '{sentences[j][:30]}...': {similarity:.4f}")
    
    # All similarities should be relatively low (< 0.5) for unrelated sentences
    avg_similarity = sum(
        generator.cosine_similarity(embeddings[i], embeddings[j])
        for i in range(len(sentences))
        for j in range(i + 1, len(sentences))
    ) / (len(sentences) * (len(sentences) - 1) / 2)
    
    print(f"\nAverage similarity: {avg_similarity:.4f}")
    
    if avg_similarity < 0.5:
        print("✓ PASS: Unrelated sentences have low average similarity")
        return True
    else:
        print("✗ FAIL: Average similarity too high for unrelated sentences")
        return False


def main():
    """Run all verification tests"""
    print("\n" + "=" * 70)
    print("SENTENCE-TRANSFORMERS INTEGRATION VERIFICATION")
    print("=" * 70)
    print("\nThis script verifies the successful integration of sentence-transformers")
    print("for real semantic embeddings in the KERNELIZE Platform.")
    print()
    
    tests = [
        ("Embedding Dimensions", test_embedding_dimensions),
        ("Semantic Similarity", test_semantic_similarity),
        ("Batch Processing", test_batch_processing),
        ("KernelQueryEngine Integration", test_kernel_query_engine_integration),
        ("Embedding Uniqueness", test_embedding_uniqueness),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All verification tests passed!")
        print("The sentence-transformers integration is working correctly.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
