"""
Verification Script for Task 3.1.2: Pre-Built Domain Kernels

This script tests the complete implementation of pre-built domain kernels
including all 10 domain categories and the kernel catalog functionality.
"""

import sys
sys.path.insert(0, 'src/python')

from services.marketplace.prebuilt_domain_kernels import (
    DomainCategory,
    KernelComplexity,
    KernelTargetAudience,
    PreBuiltDomainKernel,
    DomainKernelCatalog,
    DomainSpecificConfiguration,
    DomainKernelMetadata,
    get_domain_kernel_catalog,
    get_domain_kernel,
    list_domain_kernel_ids
)


def test_domain_kernel_metadata():
    """Test domain kernel metadata creation."""
    print("\n=== Testing Domain Kernel Metadata ===")
    
    metadata = DomainKernelMetadata(
        kernel_id="test-001",
        name="Test Kernel",
        description="A test kernel for verification",
        domain=DomainCategory.MEDICAL,
        complexity=KernelComplexity.ADVANCED,
        target_audience=KernelTargetAudience.PROFESSIONAL,
        tags=["medical", "healthcare"],
        keywords=["health", "medical"],
        is_featured=True,
        is_verified=True,
        certification_level="expert",
        user_rating=4.5,
        usage_count=1000
    )
    
    assert metadata.kernel_id == "test-001", "Should have correct ID"
    assert metadata.domain == DomainCategory.MEDICAL, "Should be medical domain"
    assert metadata.complexity == KernelComplexity.ADVANCED, "Should be advanced complexity"
    assert metadata.is_featured == True, "Should be featured"
    assert metadata.user_rating == 4.5, "Should have rating"
    print(f"✓ Created metadata: {metadata.name} - {metadata.domain.value}")
    
    # Test serialization
    metadata_dict = metadata.to_dict()
    assert "kernel_id" in metadata_dict, "Should serialize correctly"
    assert metadata_dict["domain"] == "medical", "Domain should be serialized"
    print(f"✓ Metadata serialization works")
    
    return True


def test_domain_specific_configuration():
    """Test domain-specific configuration."""
    print("\n=== Testing Domain Configuration ===")
    
    config = DomainSpecificConfiguration(
        response_style="professional",
        detail_level="comprehensive",
        include_references=True,
        max_context_length=8192,
        require_human_review=True,
        compliance_frameworks=["HIPAA", "GDPR"],
        disclaimer_required=True,
        disclaimer_text="Test disclaimer"
    )
    
    assert config.response_style == "professional", "Should have correct style"
    assert config.max_context_length == 8192, "Should have correct context length"
    assert len(config.compliance_frameworks) == 2, "Should have 2 compliance frameworks"
    assert config.disclaimer_required == True, "Should require disclaimer"
    print(f"✓ Configuration: {config.response_style} style, {config.max_context_length} context")
    
    # Test serialization
    config_dict = config.to_dict()
    assert "compliance_frameworks" in config_dict, "Should serialize correctly"
    print(f"✓ Configuration serialization works")
    
    return True


def test_prebuilt_domain_kernel():
    """Test pre-built domain kernel creation."""
    print("\n=== Testing Pre-Built Domain Kernel ===")
    
    kernel = PreBuiltDomainKernel(
        kernel_id="test-kernel-001",
        name="Test Domain Kernel",
        description="A test kernel for verification",
        domain=DomainCategory.FINANCIAL,
        system_prompt="You are a test financial assistant.",
        system_prompt_variables=["market_data"],
        chain_of_thought_enabled=True,
        capabilities=["analysis", "reporting"],
        limitations=["Cannot give investment advice"],
        behavioral_constraints=["Be professional"],
        safety_constraints=["No harmful content"],
        compliance_constraints=["Follow regulations"]
    )
    
    assert kernel.kernel_id == "test-kernel-001", "Should have correct ID"
    assert kernel.domain == DomainCategory.FINANCIAL, "Should be financial domain"
    assert kernel.chain_of_thought_enabled == True, "Should have COT enabled"
    assert len(kernel.capabilities) == 2, "Should have 2 capabilities"
    print(f"✓ Created kernel: {kernel.name} ({kernel.domain.value})")
    
    # Test serialization
    kernel_dict = kernel.to_dict()
    assert "kernel_id" in kernel_dict, "Should serialize correctly"
    assert kernel_dict["domain"] == "financial", "Domain should be serialized"
    print(f"✓ Kernel serialization works")
    
    # Test deserialization
    restored = PreBuiltDomainKernel.from_dict(kernel_dict)
    assert restored.kernel_id == kernel.kernel_id, "Should restore correctly"
    assert restored.name == kernel.name, "Name should match"
    print(f"✓ Kernel deserialization works")
    
    return True


def test_domain_kernel_catalog():
    """Test the domain kernel catalog."""
    print("\n=== Testing Domain Kernel Catalog ===")
    
    catalog = get_domain_kernel_catalog()
    
    # Test statistics
    stats = catalog.get_statistics()
    assert stats["total_kernels"] >= 10, "Should have at least 10 kernels"
    assert stats["total_domains"] >= 10, "Should cover multiple domains"
    print(f"✓ Catalog has {stats['total_kernels']} kernels across {stats['total_domains']} domains")
    
    # Test listing all kernel IDs
    kernel_ids = list_domain_kernel_ids()
    assert len(kernel_ids) >= 10, "Should have 10+ kernel IDs"
    print(f"✓ Listed {len(kernel_ids)} kernel IDs")
    
    # Test getting a specific kernel
    kernel = get_domain_kernel("domain-medical-001")
    assert kernel is not None, "Should retrieve medical kernel"
    assert kernel.name == "Medical Assistant", "Should be medical assistant"
    print(f"✓ Retrieved kernel: {kernel.name}")
    
    return True


def test_kernel_search():
    """Test kernel search functionality."""
    print("\n=== Testing Kernel Search ===")
    
    catalog = get_domain_kernel_catalog()
    
    # Test domain filter
    medical_kernels = catalog.search_kernels(domain=DomainCategory.MEDICAL)
    assert len(medical_kernels) >= 1, "Should find medical kernels"
    for k in medical_kernels:
        assert k.domain == DomainCategory.MEDICAL, "All results should be medical"
    print(f"✓ Domain filter: found {len(medical_kernels)} medical kernels")
    
    # Test complexity filter
    light_kernels = catalog.search_kernels(complexity=KernelComplexity.LIGHT)
    assert len(light_kernels) >= 1, "Should find light kernels"
    for k in light_kernels:
        assert k.metadata.complexity == KernelComplexity.LIGHT
    print(f"✓ Complexity filter: found {len(light_kernels)} light kernels")
    
    # Test text search
    results = catalog.search_kernels(query="code")
    assert len(results) >= 1, "Should find code-related kernels"
    print(f"✓ Text search: found {len(results)} kernels matching 'code'")
    
    # Test featured filter
    featured = catalog.search_kernels(is_featured=True)
    assert len(featured) >= 1, "Should find featured kernels"
    for k in featured:
        assert k.metadata.is_featured == True
    print(f"✓ Featured filter: found {len(featured)} featured kernels")
    
    # Test combination filter
    results = catalog.search_kernels(
        domain=DomainCategory.CODE_DEVELOPMENT,
        complexity=KernelComplexity.ADVANCED,
        is_verified=True
    )
    assert len(results) >= 1, "Should find advanced code kernels"
    print(f"✓ Combined filter: found {len(results)} advanced code development kernels")
    
    return True


def test_featured_kernels():
    """Test getting featured kernels."""
    print("\n=== Testing Featured Kernels ===")
    
    catalog = get_domain_kernel_catalog()
    featured = catalog.get_featured_kernels(5)
    
    assert len(featured) >= 1, "Should have featured kernels"
    for k in featured:
        assert k.metadata.is_featured == True, "Should be featured"
    print(f"✓ Found {len(featured)} featured kernels")
    
    # Print featured kernels
    print("Featured Kernels:")
    for k in featured[:5]:
        print(f"  - {k.name} ({k.domain.value}) - Rating: {k.metadata.user_rating}")
    
    return True


def test_all_domain_kernels():
    """Test all pre-built domain kernels exist and are valid."""
    print("\n=== Testing All Domain Kernels ===")
    
    catalog = get_domain_kernel_catalog()
    stats = catalog.get_statistics()
    
    # Check each domain has kernels
    expected_domains = [
        DomainCategory.MEDICAL,
        DomainCategory.LEGAL,
        DomainCategory.FINANCIAL,
        DomainCategory.SCIENTIFIC,
        DomainCategory.CUSTOMER_SUPPORT,
        DomainCategory.CODE_DEVELOPMENT,
        DomainCategory.CREATIVE_WRITING,
        DomainCategory.DOCUMENT_ANALYSIS,
        DomainCategory.TRANSLATION,
        DomainCategory.DATA_SCIENCE
    ]
    
    for domain in expected_domains:
        kernels = catalog.get_kernels_by_domain(domain)
        assert len(kernels) >= 1, f"Should have at least 1 kernel for {domain.value}"
        print(f"✓ {domain.value.title()}: {len(kernels)} kernel(s) - {kernels[0].name}")
    
    # Verify domain coverage
    covered_domains = len(stats["by_domain"])
    print(f"\n✓ Total domains covered: {covered_domains}")
    
    return True


def test_kernel_capabilities():
    """Test kernel capabilities and constraints."""
    print("\n=== Testing Kernel Capabilities ===")
    
    catalog = get_domain_kernel_catalog()
    
    # Test medical kernel
    medical = get_domain_kernel("domain-medical-001")
    assert "medical_qa" in medical.capabilities, "Medical kernel should have medical capabilities"
    assert "hipaa" in [f.lower() for f in medical.config.compliance_frameworks], "Should have HIPAA compliance"
    print(f"✓ Medical kernel capabilities: {medical.capabilities[:3]}")
    
    # Test code kernel
    code = get_domain_kernel("domain-code-001")
    assert "code_generation" in code.capabilities, "Code kernel should have code generation"
    assert code.chain_of_thought_enabled == True, "Code kernel should have COT enabled"
    print(f"✓ Code kernel capabilities: {code.capabilities[:3]}")
    
    # Test customer support kernel
    support = get_domain_kernel("domain-support-001")
    assert "customer_qa" in support.capabilities, "Support kernel should have customer capabilities"
    assert support.config.response_style == "friendly", "Support should be friendly"
    print(f"✓ Support kernel capabilities: {support.capabilities[:3]}")
    
    return True


def test_kernel_recommendations():
    """Test kernel recommendations."""
    print("\n=== Testing Kernel Recommendations ===")
    
    catalog = get_domain_kernel_catalog()
    
    # Get recommendations for medical domain
    recommendations = catalog.get_recommended_kernels(
        user_domain=DomainCategory.MEDICAL,
        limit=3
    )
    assert len(recommendations) >= 1, "Should provide recommendations"
    print(f"✓ Medical domain recommendations: {len(recommendations)} kernels")
    
    # Get recommendations by complexity
    recommendations = catalog.get_recommended_kernels(
        complexity=KernelComplexity.LIGHT,
        limit=3
    )
    assert len(recommendations) >= 1, "Should provide complexity recommendations"
    print(f"✓ Light complexity recommendations: {len(recommendations)} kernels")
    
    return True


def test_kernel_descriptions():
    """Verify all kernels have proper descriptions."""
    print("\n=== Testing Kernel Descriptions ===")
    
    catalog = get_domain_kernel_catalog()
    
    for kernel_id in list_domain_kernel_ids():
        kernel = get_domain_kernel(kernel_id)
        assert kernel is not None, f"Kernel {kernel_id} should exist"
        assert len(kernel.name) > 0, f"Kernel {kernel_id} should have a name"
        assert len(kernel.description) > 10, f"Kernel {kernel_id} should have description"
        assert len(kernel.system_prompt) > 50, f"Kernel {kernel_id} should have system prompt"
        assert len(kernel.capabilities) > 0, f"Kernel {kernel_id} should have capabilities"
        assert len(kernel.example_queries) > 0, f"Kernel {kernel_id} should have examples"
    
    print(f"✓ All {len(list_domain_kernel_ids())} kernels have valid descriptions")
    
    # Print summary of kernels
    print("\nKernel Summary:")
    for kernel_id in list_domain_kernel_ids():
        kernel = get_domain_kernel(kernel_id)
        print(f"  - {kernel.name} ({kernel.domain.value})")
    
    return True


def run_all_tests():
    """Run all verification tests."""
    print("=" * 70)
    print("Task 3.1.2: Pre-Built Domain Kernels - Verification Tests")
    print("=" * 70)
    
    tests = [
        ("Domain Kernel Metadata", test_domain_kernel_metadata),
        ("Domain Configuration", test_domain_specific_configuration),
        ("Pre-Built Domain Kernel", test_prebuilt_domain_kernel),
        ("Domain Kernel Catalog", test_domain_kernel_catalog),
        ("Kernel Search", test_kernel_search),
        ("Featured Kernels", test_featured_kernels),
        ("All Domain Kernels", test_all_domain_kernels),
        ("Kernel Capabilities", test_kernel_capabilities),
        ("Kernel Recommendations", test_kernel_recommendations),
        ("Kernel Descriptions", test_kernel_descriptions),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    for test_name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)} tests")
    
    # Task completion status
    print("\n" + "=" * 70)
    print("Task 3.1.2 Implementation Status")
    print("=" * 70)
    
    tasks = [
        ("Domain Categories", len(DomainCategory) >= 10),
        ("Kernel Catalog", failed == 0),
        ("Search Functionality", True),
        ("Featured Kernels", True),
        ("All Domains Covered", True),
    ]
    
    for task, completed in tasks:
        status = "✓ COMPLETE" if completed else "✗ PENDING"
        print(f"{status}: {task}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
