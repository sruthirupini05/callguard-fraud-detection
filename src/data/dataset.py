"""
Dataset processing, validation, and call-level stratified splitting for CallGuard.
Strictly ensures no data leakage: all turns and prefixes of a call remain within the same split.
"""

import json
import logging
import os
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.data.normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class CallGuardDatasetManager:
    """Manages raw conversation dialogues, cleaning, validation, and leakage-free splits."""

    def __init__(self, raw_path: str, processed_dir: str, manifest_dir: str):
        self.raw_path = raw_path
        self.processed_dir = processed_dir
        self.manifest_dir = manifest_dir
        self.normalizer = TextNormalizer(lowercase=True, remove_punct=False)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.manifest_dir, exist_ok=True)

    def load_and_validate(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Inspects, cleans, deduplicates, and validates transcript entries."""
        records = []
        seen_calls = set()
        
        for item in data:
            call_id = str(item.get("call_id", "")).strip()
            if not call_id:
                continue
            if call_id in seen_calls:
                logger.warning(f"Duplicate call_id detected: {call_id}. Skipping duplicate.")
                continue
            seen_calls.add(call_id)
            
            raw_transcript = item.get("transcript") or item.get("full_transcript")
            if not raw_transcript and "turns" in item:
                # Reconstruct from turns
                turn_texts = [f"{t.get('speaker', 'caller').upper()}: {t.get('text', '')}" for t in item["turns"]]
                raw_transcript = " \n".join(turn_texts)
                
            cleaned_transcript = self.normalizer.clean(raw_transcript)
            if not cleaned_transcript:
                logger.warning(f"Call {call_id} has empty transcript after cleaning. Dropping.")
                continue
                
            label = item.get("label")
            if isinstance(label, str):
                label_clean = 1 if label.lower() in ["fraud", "fraudulent", "1", "true"] else 0
            else:
                label_clean = 1 if int(label) == 1 else 0
                
            language = item.get("language", "en").lower().strip()
            category = item.get("category", "unknown")
            duration_seconds = float(item.get("duration_seconds", 30.0))
            suspicious_phrases = item.get("suspicious_phrases", [])
            
            records.append({
                "call_id": call_id,
                "label": label_clean,
                "language": language,
                "category": category,
                "duration_seconds": duration_seconds,
                "raw_transcript": raw_transcript,
                "cleaned_transcript": cleaned_transcript,
                "suspicious_phrases": json.dumps(suspicious_phrases),
                "num_turns": len(item.get("turns", []))
            })
            
        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} validated conversations.")
        return df

    def analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates class and language distributions."""
        class_counts = df["label"].value_counts().to_dict()
        lang_counts = df["language"].value_counts().to_dict()
        category_counts = df["category"].value_counts().to_dict()
        
        total = len(df)
        stats = {
            "total_calls": total,
            "fraud_count": class_counts.get(1, 0),
            "legitimate_count": class_counts.get(0, 0),
            "fraud_ratio": round(class_counts.get(1, 0) / total, 4) if total > 0 else 0,
            "languages": lang_counts,
            "categories": category_counts,
            "avg_duration": round(float(df["duration_seconds"].mean()), 2),
            "avg_turns": round(float(df["num_turns"].mean()), 2)
        }
        return stats

    def create_leakage_free_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset by unique call_id using stratified sampling on label and language.
        Guarantees that all parts of a call reside in exactly one split.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        
        # Combined stratum: label + language
        df["strata"] = df["label"].astype(str) + "_" + df["language"]
        
        # First split train vs (val + test)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            stratify=df["strata"],
            random_state=seed
        )
        
        # Second split val vs test
        relative_test_ratio = test_ratio / temp_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            stratify=temp_df["strata"],
            random_state=seed
        )
        
        train_df = train_df.drop(columns=["strata"]).copy()
        val_df = val_df.drop(columns=["strata"]).copy()
        test_df = test_df.drop(columns=["strata"]).copy()
        
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"
        
        # Verify no overlap of call_ids
        train_calls = set(train_df["call_id"])
        val_calls = set(val_df["call_id"])
        test_calls = set(test_df["call_id"])
        
        assert len(train_calls.intersection(val_calls)) == 0, "Data leakage detected: train and val overlap!"
        assert len(train_calls.intersection(test_calls)) == 0, "Data leakage detected: train and test overlap!"
        assert len(val_calls.intersection(test_calls)) == 0, "Data leakage detected: val and test overlap!"
        
        # Save split assignments
        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        split_manifest_path = os.path.join(self.manifest_dir, "call_splits.csv")
        full_df[["call_id", "split", "label", "language", "category"]].to_csv(split_manifest_path, index=False)
        
        train_df.to_csv(os.path.join(self.processed_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.processed_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_dir, "test.csv"), index=False)
        
        logger.info(f"Split complete (seed={seed}): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
