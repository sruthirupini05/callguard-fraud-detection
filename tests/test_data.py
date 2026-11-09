"""
Unit tests for CallGuard data validation and leakage-free splitting.
"""

import os
import pytest
import pandas as pd
from src.data.dataset import CallGuardDatasetManager


def test_leakage_free_splits(tmp_path):
    raw_file = str(tmp_path / "raw.json")
    proc_dir = str(tmp_path / "processed")
    man_dir = str(tmp_path / "manifests")
    
    manager = CallGuardDatasetManager(raw_file, proc_dir, man_dir)
    
    # Generate 50 dummy calls
    data = []
    for i in range(50):
        data.append({
            "call_id": f"CALL_{i:04d}",
            "label": i % 2,
            "language": "en" if i % 3 == 0 else ("hi" if i % 3 == 1 else "te"),
            "category": "test",
            "duration_seconds": 30.0,
            "turns": [
                {"speaker": "caller", "text": "Hello this is a call", "start_time": 0, "end_time": 5},
                {"speaker": "recipient", "text": "Yes speaking", "start_time": 5, "end_time": 10}
            ]
        })
        
    df = manager.load_and_validate(data)
    assert len(df) == 50
    
    train_df, val_df, test_df = manager.create_leakage_free_splits(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    
    # Check split sizes
    assert len(train_df) == 35
    assert len(val_df) + len(test_df) == 15
    
    # Check strictly ZERO overlap across call_id
    train_set = set(train_df["call_id"])
    val_set = set(val_df["call_id"])
    test_set = set(test_df["call_id"])
    
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
