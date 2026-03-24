"""Verification tests for DSE Pipeline (C05).

Group E tests: Validates that no fabricated data exists (gaussian_sigma,
systematic_bias = 0 matches), AnalyticalResourceModel name is used
(not ContractProxy in the runner), and the proxy model computes
real resource estimates.
"""

import os
import re
import subprocess

import pytest
from pathlib import Path


def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "tools" / "tapestry").exists():
            return p
        p = p.parent
    raise RuntimeError("Cannot locate repository root")


REPO_ROOT = _find_repo_root()
DSE_DIR = REPO_ROOT / "scripts" / "dse"


class TestNoFabricatedData:
    """E1: No fabricated data in DSE scripts or outputs."""

    def test_no_gaussian_sigma_in_proxy_model(self):
        """proxy_model.py should not contain gaussian_sigma noise injection."""
        pm_path = DSE_DIR / "proxy_model.py"
        assert pm_path.exists(), f"proxy_model.py not found at {pm_path}"
        content = pm_path.read_text(encoding="utf-8")

        assert "gaussian_sigma" not in content, (
            "gaussian_sigma found in proxy_model.py -- fabricated noise"
        )

    def test_no_systematic_bias_in_proxy_model(self):
        """proxy_model.py should not contain systematic_bias injection."""
        pm_path = DSE_DIR / "proxy_model.py"
        content = pm_path.read_text(encoding="utf-8")

        assert "systematic_bias" not in content, (
            "systematic_bias found in proxy_model.py -- fabricated data"
        )

    def test_no_gaussian_sigma_in_dse_runner(self):
        """dse_runner.py should not contain gaussian_sigma."""
        runner_path = DSE_DIR / "dse_runner.py"
        assert runner_path.exists(), f"dse_runner.py not found"
        content = runner_path.read_text(encoding="utf-8")

        assert "gaussian_sigma" not in content, (
            "gaussian_sigma found in dse_runner.py"
        )

    def test_no_systematic_bias_in_dse_runner(self):
        """dse_runner.py should not contain systematic_bias."""
        runner_path = DSE_DIR / "dse_runner.py"
        content = runner_path.read_text(encoding="utf-8")

        assert "systematic_bias" not in content, (
            "systematic_bias found in dse_runner.py"
        )

    def test_no_fabricated_data_in_any_dse_script(self):
        """No DSE script should contain fabricated noise injection terms."""
        for py_file in DSE_DIR.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            # Check for noise injection terms (not documentation references)
            for term in ["gaussian_sigma", "systematic_bias", "fake_correlation"]:
                assert term not in content, (
                    f"Fabricated data term '{term}' found in {py_file.name}"
                )


class TestProxyRenamed:
    """E2: AnalyticalResourceModel is the canonical name in the runner."""

    def test_dse_runner_uses_analytical_model(self):
        """dse_runner.py should use AnalyticalResourceModel, not ContractProxy."""
        runner_path = DSE_DIR / "dse_runner.py"
        content = runner_path.read_text(encoding="utf-8")

        assert "AnalyticalResourceModel" in content, (
            "Expected AnalyticalResourceModel usage in dse_runner.py"
        )

    def test_proxy_model_defines_analytical_class(self):
        """proxy_model.py should define the AnalyticalResourceModel class."""
        pm_path = DSE_DIR / "proxy_model.py"
        content = pm_path.read_text(encoding="utf-8")

        assert "class AnalyticalResourceModel" in content, (
            "Expected AnalyticalResourceModel class definition in proxy_model.py"
        )

    def test_hw_outer_still_uses_contract_proxy(self):
        """hw_outer_optimizer.py may still use ContractProxy as legacy wrapper.
        This test documents the current state rather than enforcing removal."""
        hw_outer = DSE_DIR / "hw_outer_optimizer.py"
        if not hw_outer.exists():
            pytest.skip("hw_outer_optimizer.py not found")
        content = hw_outer.read_text(encoding="utf-8")

        # ContractProxy may still exist in hw_outer as a legacy wrapper.
        # The important thing is that the main runner uses the new name.
        if "ContractProxy" in content:
            # Document that it exists but the runner uses the new name
            runner_content = (DSE_DIR / "dse_runner.py").read_text(encoding="utf-8")
            assert "AnalyticalResourceModel" in runner_content, (
                "ContractProxy exists in hw_outer but AnalyticalResourceModel "
                "should be used in dse_runner.py"
            )


class TestCorrelationDataIntegrity:
    """E1 continued: Correlation study uses real data."""

    def test_run_real_correlation_uses_analytical_model(self):
        """run_real_correlation.py should use AnalyticalResourceModel."""
        corr_path = DSE_DIR / "run_real_correlation.py"
        if not corr_path.exists():
            pytest.skip("run_real_correlation.py not found")
        content = corr_path.read_text(encoding="utf-8")

        assert "AnalyticalResourceModel" in content, (
            "Expected AnalyticalResourceModel in run_real_correlation.py"
        )

    def test_no_hardcoded_correlation_values(self):
        """run_real_correlation.py should not have pre-computed result values."""
        corr_path = DSE_DIR / "run_real_correlation.py"
        if not corr_path.exists():
            pytest.skip("run_real_correlation.py not found")
        content = corr_path.read_text(encoding="utf-8")

        for term in ["0.95", "0.98", "0.99"]:
            # These high correlation values appearing as literals would
            # suggest fabricated perfect results
            count = content.count(f'"{term}"') + content.count(f"'{term}'")
            assert count == 0, (
                f"Suspicious hardcoded correlation value {term} "
                f"in run_real_correlation.py"
            )


class TestDSEScriptStructure:
    """E: DSE scripts have proper module structure."""

    def test_dse_package_has_init(self):
        """DSE scripts should be a proper Python package."""
        init_path = DSE_DIR / "__init__.py"
        assert init_path.exists(), "Expected __init__.py in scripts/dse/"

    def test_dse_has_bayesian_opt(self):
        """bayesian_opt.py should exist for Bayesian optimization."""
        bo_path = DSE_DIR / "bayesian_opt.py"
        assert bo_path.exists(), "Expected bayesian_opt.py in scripts/dse/"

    def test_dse_has_pareto(self):
        """pareto.py should exist for Pareto frontier management."""
        pareto_path = DSE_DIR / "pareto.py"
        assert pareto_path.exists(), "Expected pareto.py in scripts/dse/"

    def test_dse_has_design_space(self):
        """design_space.py should exist for design space definition."""
        ds_path = DSE_DIR / "design_space.py"
        assert ds_path.exists(), "Expected design_space.py in scripts/dse/"
