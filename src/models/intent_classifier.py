"""
Semantic Intent Classifier, Speaker Diarization, and Anti-Warning Detector for CallGuard.
Implements:
1. Dialogue Turn Diarization (separates CALLER from YOU/USER turns)
2. Semantic Intent Classification across 5 scam vectors:
   - Credential Request (OTP, password, PIN, AnyDesk code)
   - Urgency & Panic Induction (immediate action, deadlines)
   - Authority Impersonation (Police, CBI, ED, Bank security, DoT)
   - Isolation Tactic ("don't tell anyone", confidential case, stay on line)
   - Coercive Threat (account freeze, SIM block, arrest warrant, parcel confiscation)
3. Anti-Warning / Negation Logic:
   - Identifies protective educational notices ("bank will never ask for OTP", "never share", "beware of scams")
   - Prevents false positive alerts on legitimate bank advisories (e.g. L04)
4. Multilingual Keyword & Syntactic Pattern Matchers (English, Hindi, Telugu, Hinglish)
5. Scam Taxonomy Classification (Bank OTP Fraud, Digital Arrest, Customs Parcel, Utility Cutoff, Advisory)
"""

import re
from typing import Dict, List, Tuple, Any, Optional


# ── Diarization Parser ───────────────────────────────────────────────────────

def parse_dialogue_turns(transcript: str) -> Dict[str, List[str]]:
    """
    Separates a dialogue transcript into caller turns and user/receiver turns.
    Expected speaker tags: 'CALLER:', 'YOU:', 'USER:', 'OPERATOR:', 'AGENT:'
    """
    lines = transcript.split('\n')
    caller_turns: List[str] = []
    user_turns: List[str] = []
    current_speaker = 'caller'

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        upper = line.upper()
        if upper.startswith('CALLER:') or upper.startswith('OPERATOR:') or upper.startswith('AGENT:'):
            current_speaker = 'caller'
            content = re.sub(r'^(CALLER|OPERATOR|AGENT):\s*', '', line, flags=re.IGNORECASE)
            if content.strip():
                caller_turns.append(content.strip())
        elif upper.startswith('YOU:') or upper.startswith('USER:') or upper.startswith('CUSTOMER:'):
            current_speaker = 'user'
            content = re.sub(r'^(YOU|USER|CUSTOMER):\s*', '', line, flags=re.IGNORECASE)
            if content.strip():
                user_turns.append(content.strip())
        else:
            # Continuation line
            if current_speaker == 'caller':
                caller_turns.append(line)
            else:
                user_turns.append(line)

    # If no explicit tags were present, treat full transcript as caller speech
    if not caller_turns and not user_turns:
        caller_turns = [transcript.strip()]

    return {
        "caller": caller_turns,
        "user": user_turns
    }


# ── Multilingual Intent Patterns ─────────────────────────────────────────────

INTENT_PATTERNS = {
    "credential_request": [
        # English
        r'\b(?:share|tell|give|provide|read|enter|send|verify)\b.{0,25}\b(?:otp|one[- ]time[- ]password|pin|cvv|password|passcode|code|six[- ]digit|6[- ]digit|digits?)\b',
        r'\b(?:download|install)\b.{0,25}\b(?:anydesk|teamviewer|quicksupport|remote|apk)\b',
        # Hindi / Hinglish
        r'(?:ओटीपी|otp|pin|पासवर्ड|code).{0,25}(?:बताइए|शेयर|बताओ|दीजिए|share|batao|bataiye|send)',
        r'(?:anydesk|एनीडेस्क|teamviewer).{0,25}(?:डाउनलोड|install|karein|kijiye)',
        # Telugu
        r'(?:ఓటీపీ|otp|పిన్|కోడ్).{0,25}(?:చెప్పండి|పంపండి|షేర్|share|ivvandi|cheppandi)',
    ],
    "urgency": [
        # English
        r'\b(?:immediately|right now|urgently|at once|hurry|without delay|no time|within \d+ (?:minutes?|hours?)|today only)\b',
        r'\b(?:before it is too late|last warning|instant action)\b',
        # Hindi / Hinglish
        r'(?:तुरंत|जल्दी|अभी|abhi|turant|jaldi|fauran|sirf aaj|aaj hi|bina deri)',
        # Telugu
        r'(?:వెంటనే|ఇప్పుడే|త్వరగా|ventane|ippude|tvaraga|aagakandi)',
    ],
    "authority_impersonation": [
        # English
        r'\b(?:police|cbi|ed|enforcement directorate|customs|telecom regulatory|trai|dot|supreme court|high court|bank security|cyber crime|crime branch|rbi|fraud prevention)\b',
        # Hindi / Hinglish
        r'(?:पुलिस|सीबीआई|कस्टम्स|साइबर क्राइम|police|cbi|customs|crime branch|rbi|bank officer)',
        # Telugu
        r'(?:పోలీస్|కస్టమ్స్|సిబిఐ|సైబర్ క్రైమ్|police|customs|cbi|bank adhikaari)',
    ],
    "isolation_tactic": [
        # English
        r'\b(?:do not (?:tell|inform|share with) anyone|keep this confidential|secret investigation|do not disconnect|stay on the line|digital arrest|room arrest|do not hang up)\b',
        # Hindi / Hinglish
        r'(?:किसी को मत बताना|फोन मत काटना|लाइन पर रहिए|kisi ko mat batana|phone mat kaatna|disconnect mat karna|digital arrest|secret inquiry)',
        # Telugu
        r'(?:ఎవరికీ చెప్పకండి|కాల్ కట్ చేయకండి|లైన్ లో ఉండండి|evariki cheppakandi|disconnect cheyyakandi|digital arrest)',
    ],
    "coercive_threat": [
        # English
        r'\b(?:account (?:will be|is) (?:blocked|frozen|suspended)|arrest warrant|fir (?:lodged|registered)|illegal parcel|drugs found|electricity (?:will be )?cut|sim (?:card )?deactivated)\b',
        # Hindi / Hinglish
        r'(?:खाता बंद|अकाउंट ब्लॉक|अरेस्ट वारंट|गैरकानूनी पार्सल|ड्रग्स|बिजली कट|account block|fir|arrest warrant|sim block)',
        # Telugu
        r'(?:ఖాతా బ్లాక్|అకౌంట్ రద్దు|అరెస్ట్ వారెంట్|డ్రగ్స్ పార్సెల్|కరెంట్ కట్|account block|fir)',
    ]
}

# ── Anti-Warning Negation Patterns ───────────────────────────────────────────

ANTI_WARNING_PATTERNS = [
    # English educational/defensive warnings
    r'\b(?:bank will never (?:ask|call|request)|never share your (?:otp|pin|cvv|password)|beware of (?:scams?|fraud|fake calls?)|do not share (?:otp|pin|credentials)|caution|awareness advisory|fraud alert|security tip)\b',
    r'\b(?:if someone asks (?:for|to share) (?:otp|pin|password),? do not)\b',
    r'\b(?:we never ask for any confidential details)\b',
    # Hindi
    r'(?:बैंक कभी भी (?:ओटीपी|पिन) नहीं मांगता|किसी के साथ ओटीपी शेयर न करें|धोखाधड़ी से सावधान|सतर्क रहें)',
    r'(?:kisi ko bhi otp share na karein|bank kabhi otp nahi maangta|cyber fraud se bachein)',
    # Telugu
    r'(?:బ్యాంక్ ఎప్పుడూ ఓటీపీ అడగదు|మీ ఓటీపీ ఎవరితోనూ పంచుకోకండి|మోసాల పట్ల జాగ్రత్త|హెచ్చరిక)',
]


# ── Core Classifier Functions ────────────────────────────────────────────────

def detect_anti_warning(text: str) -> Tuple[bool, List[str]]:
    """
    Detects whether the utterance is a protective advisory warning users NOT to disclose credentials.
    """
    matches = []
    text_lower = text.lower()
    for pattern in ANTI_WARNING_PATTERNS:
        found = re.findall(pattern, text_lower)
        if found:
            matches.extend(found)
    return len(matches) > 0, matches


def score_caller_intents(caller_turns: List[str]) -> Dict[str, float]:
    """
    Analyzes caller turns and scores each malicious intent dimension in [0.0, 1.0].
    """
    combined_caller_text = " ".join(caller_turns).lower()
    intent_scores: Dict[str, float] = {}

    for intent, patterns in INTENT_PATTERNS.items():
        hits = 0
        for pattern in patterns:
            found = re.findall(pattern, combined_caller_text)
            hits += len(found)
        # Saturating non-linear score: 0 hits -> 0.0, 1 hit -> 0.65, 2+ hits -> 0.85-0.98
        if hits == 0:
            score = 0.0
        elif hits == 1:
            score = 0.68
        elif hits == 2:
            score = 0.88
        else:
            score = min(0.99, 0.88 + (hits - 2) * 0.04)
        intent_scores[intent] = round(score, 3)

    return intent_scores


def classify_scam_taxonomy(intent_scores: Dict[str, float], caller_text: str, is_anti_warning: bool) -> str:
    """
    Assigns an actionable fraud taxonomy category based on detected semantic intents.
    """
    if is_anti_warning:
        return "Legitimate Bank Advisory / Notice"

    lower = caller_text.lower()

    if intent_scores.get("isolation_tactic", 0) >= 0.5 and (
        "police" in lower or "cbi" in lower or "parcel" in lower or "digital arrest" in lower or "drugs" in lower
    ):
        return "Digital Arrest / Police Extortion"

    if intent_scores.get("coercive_threat", 0) >= 0.5 and (
        "parcel" in lower or "customs" in lower or "courier" in lower or "narcotics" in lower or "fedex" in lower
    ):
        return "Customs / Narcotics Illegal Parcel Scam"

    if intent_scores.get("credential_request", 0) >= 0.5 and (
        "otp" in lower or "kyc" in lower or "bank" in lower or "card" in lower or "account" in lower
    ):
        return "Bank OTP / KYC Suspension Fraud"

    if "electricity" in lower or "power" in lower or "bill" in lower or "meter" in lower:
        return "Utility / Electricity Cutoff Scam"

    if "lottery" in lower or "prize" in lower or "winner" in lower or "crore" in lower:
        return "Lottery / Imposter Reward Scam"

    # Default based on overall intent level
    max_intent = max(intent_scores.values()) if intent_scores else 0.0
    if max_intent >= 0.65:
        return "Social Engineering Threat"
    elif max_intent >= 0.35:
        return "Suspicious Solicitation"
    else:
        return "Legitimate Customer Dialogue"


def analyze_conversation_intents(transcript: str) -> Dict[str, Any]:
    """
    Main entrypoint: Performs diarization, anti-warning check, intent scoring,
    and returns full semantic audit metadata.
    """
    diarized = parse_dialogue_turns(transcript)
    caller_text = " ".join(diarized["caller"])

    is_anti_warning, warning_cues = detect_anti_warning(transcript)
    intent_scores = score_caller_intents(diarized["caller"])

    # If it is clearly an advisory message from an official organization, suppress malicious intent
    if is_anti_warning:
        for k in intent_scores:
            intent_scores[k] = round(intent_scores[k] * 0.15, 3)

    scam_taxonomy = classify_scam_taxonomy(intent_scores, caller_text, is_anti_warning)

    # Compute composite semantic fraud intent score [0, 1]
    if is_anti_warning:
        composite_intent = 0.08
    else:
        # Weighted combination of intent dimensions
        composite_intent = (
            intent_scores.get("credential_request", 0) * 0.35 +
            intent_scores.get("coercive_threat", 0) * 0.25 +
            intent_scores.get("urgency", 0) * 0.15 +
            intent_scores.get("authority_impersonation", 0) * 0.15 +
            intent_scores.get("isolation_tactic", 0) * 0.10
        )
        composite_intent = min(1.0, composite_intent)

    return {
        "diarization": diarized,
        "is_anti_warning": is_anti_warning,
        "warning_cues": warning_cues,
        "intents": intent_scores,
        "composite_intent_score": round(composite_intent, 3),
        "scam_taxonomy": scam_taxonomy
    }
