"""Shared failures for exact corpus workload materialization."""


class WorkloadProviderError(ValueError):
    """Raised when an owned workload cannot be materialized exactly."""
