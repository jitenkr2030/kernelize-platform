"""
KERNELIZE Platform - Test Configuration
=========================================

This module provides pytest configuration and fixtures
for the KERNELIZE test suite.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import pytest
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture
def sample_text():
    """Sample text for testing compression"""
    return """
    Artificial intelligence (AI) is a rapidly evolving field that encompasses
    various technologies and approaches. Machine learning, a subset of AI,
    enables computers to learn from and make predictions based on data.
    Deep learning, which uses neural networks with many layers, has achieved
    remarkable results in areas such as image recognition and natural language
    processing. As AI continues to advance, it will have profound impacts on
    healthcare, finance, transportation, and many other industries.
    """.strip()


@pytest.fixture
def sample_healthcare_text():
    """Sample healthcare text for domain testing"""
    return """
    Patient: John Smith
    Date of Service: 2024-01-15
    Diagnosis: E11.9 Type 2 diabetes mellitus without complications
    HPI: 45-year-old male presents for routine diabetes follow-up.
    Denies chest pain, shortness of breath, or dizziness.
    Medications: Metformin 500mg BID, Lisinopril 10mg daily
    Assessment: Diabetes well-controlled, continue current regimen
    Plan: Return in 3 months for HbA1c check
    """.strip()


@pytest.fixture
def sample_finance_text():
    """Sample finance text for domain testing"""
    return """
    Company: TechCorp Inc.
    Q3 2024 Financial Summary
    Revenue: $1,234,567,890 (12.5% YoY increase)
    Net Income: $123,456,789
    EPS: $4.56 (analyst consensus: $4.52)
    Gross Margin: 45.2%
    Operating Margin: 22.1%
    Cash Flow: $456,789,012
    Balance Sheet Total: $5,678,901,234
    Stock trades on NASDAQ:TCorp with P/E of 18.5x
    """.strip()


@pytest.fixture
def sample_legal_text():
    """Sample legal text for domain testing"""
    return """
    AGREEMENT made this 15th day of January, 2024
    BETWEEN: Party A ("Licensor") and Party B ("Licensee")
    
    WHEREAS, Licensor owns certain intellectual property rights;
    AND WHEREAS, Licensee desires to obtain a license to use such rights;
    
    1. GRANT OF LICENSE
    Subject to the terms herein, Licensor grants to Licensee a non-exclusive,
    non-transferable license to use the Software for internal business purposes.
    
    2. CONSIDERATION
    In exchange for the license, Licensee agrees to pay $50,000 annually.
    
    3. TERM AND TERMINATION
    This Agreement shall commence on the Effective Date and continue for one year.
    Either party may terminate with 30 days written notice.
    
    4. CONFIDENTIALITY
    Each party agrees to maintain the confidentiality of Proprietary Information.
    
    See Smith v. Jones, 123 U.S. 456 (2023) for relevant precedent.
    Section 4(b) of the Copyright Act applies to this situation.
    """.strip()


@pytest.fixture
def sample_tech_text():
    """Sample technology text for domain testing"""
    return """
    # API Documentation
    
    ## Authentication
    All API requests require authentication using Bearer tokens.
    `Authorization: Bearer <your_token>`
    
    ## Endpoints
    
    ### GET /api/v1/users
    Returns a list of users.
    
    Response:
    ```json
    {
        "users": [
            {"id": 1, "name": "John"},
            {"id": 2, "name": "Jane"}
        ]
    }
    ```
    
    ### POST /api/v1/users
    Create a new user.
    
    ```python
    def create_user(name, email):
        user = {"name": name, "email": email}
        return user
    ```
    
    ## Error Handling
    Common error codes:
    - 400: Bad Request
    - 401: Unauthorized
    - 404: Not Found
    - 500: Internal Server Error
    """.strip()


@pytest.fixture
def sample_education_text():
    """Sample education text for domain testing"""
    return """
    Course: Introduction to Algebra
    Grade Level: 9-10
    
    Module 1: Foundations of Algebra
    Learning Objective: Students will be able to evaluate algebraic expressions.
    Learning Objective: Students will be able to solve linear equations.
    
    Essential Question: How can we use variables to represent real-world quantities?
    Key Concept: Expressions represent quantities; equations represent relationships.
    
    Lesson Plan:
    1. Introduction to variables (15 min)
    2. Practice evaluating expressions (20 min)
    3. Group activity: Real-world scenarios (15 min)
    
    Assessment: Quiz on Module 1 topics
    Standard Alignment: CCSS.MATH.9-12.A.REI.3
    """.strip()


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add slow marker to tests that may take longer
        if "batch" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.slow)
