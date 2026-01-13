"""
KERNELIZE Platform - Finance Domain Processor
==============================================

This module implements the finance-specific knowledge compression
processor for the KERNELIZE Platform. It preserves financial metrics,
regulatory compliance markers, and numerical precision.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .base import BaseDomainProcessor, DomainContext, DomainSchema


# ==================== Finance Schema ====================

class FinanceMetadata(DomainSchema):
    """Validation schema for finance domain metadata"""
    report_period: Optional[datetime] = Field(
        None,
        description="Financial reporting period end date",
    )
    fiscal_year: Optional[int] = Field(
        None,
        description="Fiscal year",
        ge=1900,
        le=2100,
    )
    currency_code: Optional[str] = Field(
        default="USD",
        description="ISO 4217 currency code",
        pattern="^[A-Z]{3}$",
    )
    compliance_level: Optional[str] = Field(
        default="standard",
        description="Regulatory compliance level",
        pattern="^(standard|gaap|ifrs|sox)$",
    )
    report_type: Optional[str] = Field(
        None,
        description="Type of financial report",
        pattern="^(balance_sheet|income_statement|cash_flow|10-k|10-q|annual_report)$",
    )
    auditor: Optional[str] = Field(None, description="Auditing firm")
    entity_name: Optional[str] = Field(None, description="Reporting entity name")


class FinanceProcessor(BaseDomainProcessor):
    """
    Finance Domain Knowledge Processor
    
    Specialized compression for financial statements, reports,
    and market analysis. Key features:
    - Preservation of numerical precision
    - Currency and percentage format protection
    - Regulatory compliance markers (SOX, GAAP, IFRS)
    - Ticker symbol and market data preservation
    """
    
    DOMAIN_NAME = "finance"
    DISPLAY_NAME = "Finance"
    DESCRIPTION = "Specialized compression for financial statements and market analysis"
    
    def _configure_patterns(self) -> None:
        """Configure finance-specific preservation patterns"""
        # Currency and numerical formats
        self.preserve_patterns.extend([
            r'\$[\d,]+\.?\d*',  # Currency amounts ($1,234.56)
            r'[\d,]+\.?\d*%\s*',  # Percentages (12.5%)
            r'\b[A-Z]{1,5}\s+(?:USD|EUR|GBP|JPY|CNY)\b',  # Currency codes
            r'\b(?:Q[1-4]|FY)\s*\d{2,4}\b',  # Fiscal periods
        ])
        
        # Financial metrics and ratios
        self.preserve_patterns.extend([
            r'\b(?:EBITDA|ROI|ROE|ROA|P/E|EPS|DPS|CFPS)\b[:\s]*[\d.]+',  # Key metrics
            r'\b(?:revenue|net income|gross profit|operating income|ebitda)\b[:\s]*\$?[\d,]+',  # Line items
            r'\b(?:assets|liabilities|equity|cash flow|working capital)\b[:\s]*\$?[\d,]+',  # Balance sheet items
        ])
        
        # Market data
        self.preserve_patterns.extend([
            r'\b(?:NASDAQ|NYSE|AMEX|S&P|DOW|JONES):?[A-Z]+\b',  # Index/ticker
            r'\b[A-Z]{1,5}\b(?:\s*(?:stock|share|equity|bond|etf))?',  # Ticker symbols
            r'\b(?:market cap|shares outstanding|beta|dividend yield)\b[:\s]*[\d.,]+',  # Market metrics
        ])
        
        # Regulatory markers
        self.critical_terms.extend([
            "material", "materiality", "restatement", "recast",  # Quality of earnings
            "gaap", "ifrs", "sox", "sec", "fasb", "iiasb",  # Compliance
            "qualified opinion", "unqualified opinion", "adverse opinion",  # Audit opinions
            "contingent liability", "off-balance sheet", "related party",  # Risk disclosures
        ])
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate finance-specific metadata.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            FinanceMetadata(**metadata)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_domain_schema(self) -> type:
        """Get the Pydantic schema for finance validation"""
        return FinanceMetadata
    
    def score_sentence_importance(
        self,
        sentence: str,
        entities: List[str],
        position: int,
        total: int,
        context: Optional[DomainContext] = None,
    ) -> float:
        """
        Score sentence importance with finance-specific adjustments.
        
        Financial documents typically have front-loaded key metrics
        and detailed footnotes that may be compressed more aggressively.
        """
        if context is None:
            context = self.create_context()
        
        # Base score
        score = super().score_sentence_importance(
            sentence, entities, position, total, context
        )
        
        sentence_lower = sentence.lower()
        
        # Boost sentences with key metrics
        if re.search(r'\$[\d,]+', sentence):
            score += 1.5
        
        # Boost sentences with percentage changes
        if re.search(r'\d+\.?\d*%', sentence):
            score += 1.5
        
        # Boost compliance and risk language
        if any(term in sentence_lower for term in [
            "material", "contingent", "liability", "risk", "uncertainty",
            "compliance", "regulation", "material weakness", "significant deficiency"
        ]):
            score += 2.0
        
        # Boost forward-looking statements and guidance
        if any(term in sentence_lower for term in [
            "outlook", "guidance", "outlook", "forecast", "expects", "anticipates",
            "non-gaap", "reconciliation", "adjusted"
        ]):
            score += 1.5
        
        # Slightly reduce score for detailed footnotes (can be compressed)
        if "footnote" in sentence_lower or "note" in sentence_lower:
            if position > total * 0.5:  # Later in document
                score -= 0.5
        
        return score
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get finance processor information.
        
        Returns:
            Dictionary with processor details
        """
        info = super().get_processor_info()
        
        info.update({
            "supported_standards": ["GAAP", "IFRS", "SOX", "SEC"],
            "supported_report_types": [
                "Balance Sheet",
                "Income Statement",
                "Cash Flow Statement",
                "10-K",
                "10-Q",
            ],
            "preserved_formats": [
                "Currency ($X,XXX.XX)",
                "Percentages (XX.X%)",
                "Ticker Symbols",
                "Financial Ratios",
            ],
        })
        
        return info
