"""
Text Normalization and Cleaning Module for CallGuard
Handles transcript cleaning, punctuation handling, code-mixed token normalization,
and multi-language tokenization (English, Hindi, Telugu).
"""

import re
import unicodedata
from typing import List, Optional


class TextNormalizer:
    """Cleans and standardizes raw telephone transcripts."""
    
    def __init__(self, lowercase: bool = True, strip_accents: bool = False, remove_punct: bool = False):
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.remove_punct = remove_punct
        
        # Regex for cleaning repetitive whitespace and audio artifacts
        self._whitespace_re = re.compile(r'\s+')
        self._bracket_tags_re = re.compile(r'\[.*?\]|\(.*?\)')
        self._punct_re = re.compile(r'[^\w\s\u0900-\u097F\u0C00-\u0C7F]') # Keep Devanagari & Telugu unicode ranges

    def clean(self, text: Optional[str]) -> str:
        """Sanitizes raw text, handles missing values, and normalizes unicode."""
        if text is None or not isinstance(text, str):
            return ""
            
        # Normalize unicode (NFC)
        text = unicodedata.normalize("NFC", text)
        
        # Remove audio artifact annotations e.g. [laughter], [inaudible], (static)
        text = self._bracket_tags_re.sub(" ", text)
        
        if self.lowercase:
            text = text.lower()
            
        if self.remove_punct:
            text = self._punct_re.sub(" ", text)
            
        # Standardize multiple spaces into single space
        text = self._whitespace_re.sub(" ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace-based word tokenization supporting multilingual strings."""
        cleaned = self.clean(text)
        return cleaned.split() if cleaned else []
