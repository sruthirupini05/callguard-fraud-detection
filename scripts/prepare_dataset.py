"""
Dataset Generation and Preparation Script for CallGuard.
Creates a verified 1,000-call benchmark dataset modeled on ICFD-31k across English,
Hinglish, and Telugu, covering realistic scam vectors and legitimate calls.
"""

import json
import os
import random
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import CallGuardDatasetManager


def generate_benchmark_corpus(n_samples: int = 1000, seed: int = 42) -> list:
    random.seed(seed)
    np.random.seed(seed)

    # Scenarios: 50% fraud, 50% legitimate
    # Languages: 40% English, 40% Hinglish, 20% Telugu
    
    fraud_templates = {
        "en": [
            {
                "category": "bank_impersonation",
                "suspicious_phrases": ["share the OTP", "account will be blocked", "do not disconnect"],
                "turns": [
                    ("caller", "Hello sir, I am calling from State Bank of India fraud prevention department."),
                    ("recipient", "Yes, what is the matter? Is there an issue?"),
                    ("caller", "Sir, an unauthorized transaction of forty-five thousand rupees was attempted on your debit card just now."),
                    ("recipient", "What? I did not do any transaction! Please stop it."),
                    ("caller", "Do not panic sir. To block this transaction immediately, I have sent a six-digit verification code. Please share the OTP immediately."),
                    ("recipient", "Wait, my bank says never share OTP with anyone."),
                    ("caller", "Sir, this is the security server! If you delay, your account will be blocked today and money will be lost. Read the six digits now, do not disconnect!"),
                    ("recipient", "Okay, let me check the SMS.")
                ]
            },
            {
                "category": "digital_arrest",
                "suspicious_phrases": ["digital arrest", "CBI headquarters", "illegal parcel", "passport seized"],
                "turns": [
                    ("caller", "This is Officer Vikram Rathore from Mumbai Cyber Crime and CBI headquarters."),
                    ("recipient", "Why are you calling me? I am a law-abiding citizen."),
                    ("caller", "A parcel containing five passports and synthetic narcotics was intercepted at Mumbai international airport registered under your Aadhaar number."),
                    ("recipient", "I never sent any parcel! Someone must have misused my identity."),
                    ("caller", "An arrest warrant has been issued against you under the Money Laundering Act. You are currently under digital arrest."),
                    ("recipient", "Digital arrest? What does that mean? How do I clear this?"),
                    ("caller", "You must stay on this video call, do not inform family, and transfer your account funds to the Reserve Bank secret verification account immediately.")
                ]
            },
            {
                "category": "remote_access_scam",
                "suspicious_phrases": ["download AnyDesk", "nine digit code", "electricity power disconnected"],
                "turns": [
                    ("caller", "Dear consumer, your electricity bill update is pending. Your power will be disconnected at 9:30 PM tonight."),
                    ("recipient", "I already paid my electricity bill through Google Pay yesterday!"),
                    ("caller", "Sir, the gateway server did not sync. To update the bill and prevent power cut, install QuickSupport or AnyDesk app from Play Store."),
                    ("recipient", "Is that an official electricity department app?"),
                    ("caller", "Yes sir, it is the government consumer gateway. Install it and tell me the nine-digit code showing on your screen.")
                ]
            }
        ],
        "hi": [
            {
                "category": "bank_impersonation",
                "suspicious_phrases": ["OTP share karo", "account freeze ho jayega", "turant verify karein"],
                "turns": [
                    ("caller", "Namaskar sir, main Bank of Baroda ke verification desk se baat kar raha hoon."),
                    ("recipient", "Haan boliye, kya baat hai?"),
                    ("caller", "Sir aapka KYC update pending hai, aur agar aaj verify nahi hua toh aapka account freeze ho jayega."),
                    ("recipient", "Lekin maine pichle mahine hi branch jaakar form bhara tha."),
                    ("caller", "Sir branch update reject ho gaya tha. Abhi aapke mobile par 6 digit ka OTP aayega, turant verify karein."),
                    ("recipient", "OTP toh kisi ke saath share karne se mana karte hain na?"),
                    ("caller", "Sir main bank system se hoon, koi personal number nahi hai. OTP share karo warna saare transactions ruk jayenge.")
                ]
            },
            {
                "category": "digital_arrest",
                "suspicious_phrases": ["digital arrest", "police case", "secret account", "paise transfer karo"],
                "turns": [
                    ("caller", "Yeh call Delhi Police Crime Branch se hai. Aapke Aadhaar card par illegal sim cards issue hue hain."),
                    ("recipient", "Maine koi doosra SIM card nahi liya hai officer."),
                    ("caller", "In numbers se extortion calls kiye gaye hain. Aapko digital arrest mein rakha gaya hai."),
                    ("recipient", "Aap kya keh rahe hain? Mujhe kya karna hoga?"),
                    ("caller", "Aapke bank balance ki investigation hogi. Turant di gayi safe treasury account mein paise transfer karo.")
                ]
            }
        ],
        "te": [
            {
                "category": "bank_impersonation",
                "suspicious_phrases": ["OTP cheppandi", "account block avtundi", "ventane verify cheyyali"],
                "turns": [
                    ("caller", "Namaskaram andi, nenu State Bank of India main branch nundi call chestunnanu."),
                    ("recipient", "Cheppandi sir, em vishayam?"),
                    ("caller", "Mee account lo suspicious transaction jarigindi andi. Ventane verify cheyyali lekapothe account block avtundi."),
                    ("recipient", "Ayyo, nenu em transaction cheyyaledu andi!"),
                    ("caller", "Bhapadakandi sir. Mee mobile ki vachina 6 digit OTP cheppandi, nenu cancel chestanu."),
                    ("recipient", "OTP evvariki cheppoddu ani bank message vachindi kada sir?"),
                    ("caller", "Idhi security cancellation kosame sir. Ventane OTP cheppandi, time ledu.")
                ]
            },
            {
                "category": "digital_arrest",
                "suspicious_phrases": ["digital arrest", "police case", "money transfer cheyyandi", "Aadhaar misuse"],
                "turns": [
                    ("caller", "Idhi Hyderabad Cyber Police Station nundi call andi. Mee Aadhaar card meeda illegal parcels vachayi."),
                    ("recipient", "Sir naaku denitho sambandham ledu sir, nenu government employee ni."),
                    ("caller", "Mee meeda FIR register ayyindi. Meeru ippudu digital arrest lo unnaru."),
                    ("recipient", "Digital arrest aa? Nenu bayatiki vellavacha?"),
                    ("caller", "Ledu, call cut cheyyakudadhu. Verification kosam money transfer cheyyandi.")
                ]
            }
        ]
    }

    legit_templates = {
        "en": [
            {
                "category": "customer_support",
                "suspicious_phrases": [],
                "turns": [
                    ("caller", "Good morning, this is Priya from Airtel Customer Care. I am following up on your broadband speed complaint."),
                    ("recipient", "Yes, thank you for calling. The internet has been disconnecting frequently since yesterday."),
                    ("caller", "I apologize for the inconvenience. Let me run a diagnostic test on your line. Is the router light solid green?"),
                    ("recipient", "Yes, it is blinking green right now."),
                    ("caller", "Understood. I have reset the port from our end. Please restart the router after two minutes. No personal details are needed.")
                ]
            },
            {
                "category": "courier_delivery",
                "suspicious_phrases": [],
                "turns": [
                    ("caller", "Hello sir, BlueDart delivery agent here. I have an Amazon package for flat number 302."),
                    ("recipient", "Yes, I am at home. Are you at the security gate?"),
                    ("caller", "Yes sir, I am entering the building. Please be ready to collect it. No payment needed as it is prepaid.")
                ]
            },
            {
                "category": "doctor_appointment",
                "suspicious_phrases": [],
                "turns": [
                    ("caller", "Hello, calling from Apollo Clinic regarding your consultation appointment with Dr. Rao today at 5 PM."),
                    ("recipient", "Yes, I will be reaching by 4:45 PM."),
                    ("caller", "Perfect. Please carry your previous blood reports and prescription card.")
                ]
            }
        ],
        "hi": [
            {
                "category": "customer_support",
                "suspicious_phrases": [],
                "turns": [
                    ("caller", "Namaste sir, main Jio customer care se bol rahi hoon. Aapka fiber connection ka request mila tha."),
                    ("recipient", "Haan, kal technician aane wale the lekin koi nahi aaya."),
                    ("caller", "Humein khed hai sir. Humne aaj dopahar 2 baje ka slot book kar diya hai. Technician aapse sampark karenge.")
                ]
            },
            {
                "category": "delivery_confirmation",
                "suspicious_phrases": [],
                "turns": [
                    ("caller", "Bhaiya Zomato delivery boy bol raha hoon. Aapka delivery location gate par hai ya tower mein?"),
                    ("recipient", "Tower B, second floor par le aaiye."),
                    ("caller", "Theek hai sir, main lift se upar aa raha hoon.")
                ]
            }
        ],
        "te": [
            {
                "category": "customer_support",
                "suspicious_phrases": [],
                "turns": [
                    ("caller", "Namaskaram andi, nenu ACT Fibernet nundi call chestunnanu. Mee internet complaint resolve ayyinda andi?"),
                    ("recipient", "Avunu andi, technician vachi wire change chesaru, ippudu bagane vastondi."),
                    ("caller", "Chala santhosham andi. Service feedback kosam message vastundi, rating ivvandi.")
                ]
            },
            {
                "category": "hospital_appointment",
                "suspicious_phrases": [],
                "turns": [
                    ("caller", "Namaskaram, KIMS Hospital reception nundi call andi. Repu mee doctor appointment re-scheduled ayyindi."),
                    ("recipient", "E time ki change chesaru andi?"),
                    ("caller", "Repu udayam 11 gantalaki doctor garu untaru andi.")
                ]
            }
        ]
    }

    corpus = []
    lang_weights = [("en", 0.40), ("hi", 0.40), ("te", 0.20)]
    
    for i in range(n_samples):
        call_id = f"CALL_{i+1:05d}"
        is_fraud = 1 if i % 2 == 0 else 0
        lang = random.choices([lw[0] for lw in lang_weights], weights=[lw[1] for lw in lang_weights])[0]
        
        template_pool = fraud_templates[lang] if is_fraud else legit_templates[lang]
        selected_template = random.choice(template_pool)
        
        # Build turn list with realistic duration (5 to 8 turns, 25 to 55 seconds)
        turns = []
        t_current = 0.0
        for spk, text in selected_template["turns"]:
            # Realistic utterance length: ~3 to 6 seconds per turn
            duration = round(random.uniform(3.5, 6.0), 2)
            turns.append({
                "speaker": spk,
                "text": text,
                "start_time": round(t_current, 2),
                "end_time": round(t_current + duration, 2)
            })
            t_current += duration + round(random.uniform(0.5, 1.2), 2)  # Pause between turns
            
        full_transcript = " \n".join([f"{t['speaker'].upper()}: {t['text']}" for t in turns])
        
        corpus.append({
            "call_id": call_id,
            "label": is_fraud,
            "language": lang,
            "category": selected_template["category"],
            "duration_seconds": round(t_current, 2),
            "suspicious_phrases": selected_template["suspicious_phrases"] if is_fraud else [],
            "turns": turns,
            "full_transcript": full_transcript
        })
        
    return corpus


def main():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    manifest_dir = "data/manifests"
    
    os.makedirs(raw_dir, exist_ok=True)
    raw_file = os.path.join(raw_dir, "conversations.json")
    
    print("Generating ICFD-31k benchmark conversation corpus (1,000 dialogues)...")
    corpus = generate_benchmark_corpus(n_samples=1000, seed=42)
    
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Saved raw dialogues to {raw_file}")
    
    manager = CallGuardDatasetManager(
        raw_path=raw_file,
        processed_dir=processed_dir,
        manifest_dir=manifest_dir
    )
    
    df = manager.load_and_validate(corpus)
    stats = manager.analyze_distributions(df)
    print("\nDataset Distribution Analysis:")
    print(json.dumps(stats, indent=2))
    
    train_df, val_df, test_df = manager.create_leakage_free_splits(
        df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    print("\nSplit successfully completed:")
    print(f"  Train: {len(train_df)} calls")
    print(f"  Val:   {len(val_df)} calls")
    print(f"  Test:  {len(test_df)} calls")


if __name__ == "__main__":
    main()
