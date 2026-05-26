"""
test_fusion.py — Unit tests for HealthFusionEngine edge cases.
"""

import pytest
from src.fusion.fusion_engine import HealthFusionEngine


def test_low_risk():
    """Test that low probabilities produce Low Risk tier."""
    engine = HealthFusionEngine()
    result = engine.fuse(0.1, 0.1, 0.1)
    assert result['risk_tier'] == 'Low Risk'
    assert 0 <= result['composite_score'] <= 100


def test_critical_risk():
    """Test that high probabilities produce Critical tier."""
    engine = HealthFusionEngine()
    result = engine.fuse(0.9, 0.9, 0.9)
    assert result['risk_tier'] == 'Critical'


def test_mental_amplification():
    """Test that mental health risk amplifies physical disease risks."""
    engine = HealthFusionEngine()
    r_low_mental = engine.fuse(0.5, 0.5, 0.0)
    r_high_mental = engine.fuse(0.5, 0.5, 1.0)
    assert r_high_mental['diabetes_risk_pct'] >= r_low_mental['diabetes_risk_pct']


def test_recommendations_not_empty():
    """Test that recommendations are always generated (at least 3)."""
    engine = HealthFusionEngine()
    result = engine.fuse(0.6, 0.4, 0.7)
    assert len(result['recommendations']) >= 3


def test_moderate_risk():
    """Test that moderate probabilities produce Moderate Risk tier."""
    engine = HealthFusionEngine()
    result = engine.fuse(0.35, 0.35, 0.35)
    assert result['risk_tier'] == 'Moderate Risk'


def test_high_risk():
    """Test that high probabilities produce High Risk tier."""
    engine = HealthFusionEngine()
    result = engine.fuse(0.7, 0.7, 0.7)
    assert result['risk_tier'] in ['High Risk', 'Critical']


def test_zero_inputs():
    """Test that zero inputs produce Low Risk tier."""
    engine = HealthFusionEngine()
    result = engine.fuse(0.0, 0.0, 0.0)
    assert result['risk_tier'] == 'Low Risk'
    assert result['composite_score'] == 0.0


def test_max_inputs():
    """Test that maximum inputs are capped at 100%."""
    engine = HealthFusionEngine()
    result = engine.fuse(1.0, 1.0, 1.0)
    assert result['diabetes_risk_pct'] <= 100.0
    assert result['heart_risk_pct'] <= 100.0
    assert result['mental_confidence_pct'] == 100.0


def test_mental_status_healthy():
    """Test mental status is Healthy when mental prob < 0.5."""
    engine = HealthFusionEngine()
    result = engine.fuse(0.5, 0.5, 0.3)
    assert result['mental_status'] == 'Healthy'


def test_mental_status_depressed():
    """Test mental status is Depressed/Stressed when mental prob >= 0.5."""
    engine = HealthFusionEngine()
    result = engine.fuse(0.5, 0.5, 0.7)
    assert result['mental_status'] == 'Depressed/Stressed'


def test_health_summary_not_empty():
    """Test that health summary is always generated."""
    engine = HealthFusionEngine()
    result = engine.fuse(0.5, 0.5, 0.5)
    assert len(result['health_summary']) > 50


def test_amplify_risk_bounded():
    """Test that amplified risk never exceeds 1.0."""
    engine = HealthFusionEngine()
    result = engine.amplify_risk(0.99, 1.0, alpha=0.5)
    assert result <= 1.0


def test_custom_weights():
    """Test fusion with custom weight initialization."""
    engine = HealthFusionEngine(w_diabetes=0.5, w_heart=0.3, w_mental=0.2)
    result = engine.fuse(0.5, 0.5, 0.5)
    assert 0 <= result['composite_score'] <= 100
