"""Diagnostics for information upper bounds with a frozen VideoLLM.

The package deliberately lives in one repository folder so the complete
experiment (adapters, condition construction, feature extraction, scoring, and
statistics) can be audited independently of the main benchmark harness.
"""

from .schema import SCHEMA_VERSION, ValidationIssue, validate_record

__all__ = ["SCHEMA_VERSION", "ValidationIssue", "validate_record"]
