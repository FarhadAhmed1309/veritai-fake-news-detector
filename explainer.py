"""
explainer.py — Rule-based signals that explain WHY an article was
flagged FAKE or REAL: sensational language, sentiment, clickbait
phrases, and a blended credibility score.
"""

import re

SENSATIONAL_WORDS = [
    "shocking", "unbelievable", "exposed", "secret", "miracle",
    "breaking", "urgent", "banned", "conspiracy", "hoax",
    "hidden", "explosive", "bombshell", "scandalous", "leaked",
    "warning", "alert", "exclusive", "stunning", "terrifying",
    "outrage", "scandal", "betrayal", "chaos", "crisis", "hysteria",
    "sensational", "dramatic", "radical", "extreme", "insane"
]

POSITIVE_WORDS = [
    "good", "great", "excellent", "happy", "success", "win", "best",
    "love", "hope", "wonderful", "amazing", "positive", "benefit",
    "improve", "safe", "trust", "support", "peace", "growth", "proud",
    "fantastic", "brilliant", "outstanding", "superb", "joyful",
    "celebrate", "achieve", "progress", "healthy", "strong", "helpful"
]

NEGATIVE_WORDS = [
    "bad", "terrible", "awful", "hate", "fail", "worst", "evil",
    "corrupt", "attack", "fear", "danger", "threat", "lie", "false",
    "fraud", "crime", "death", "kill", "destroy", "collapse", "panic",
    "disaster", "catastrophe", "violence", "brutal", "cruel", "toxic",
    "devastating", "horrific", "sinister", "traitor", "rotten", "disgusting"
]

URDU_FAKE_WORDS = [
    "jhoot", "jhoota", "farzi", "fake", "jali", "dhoka", "bewaqoof",
    "haqeeqat", "sach", "iftira", "propaganda", "breaking", "asal",
    "nakli", "sazish", "azaab", "fitna", "fasad", "beghairat"
]

EXAGGERATIONS = [
    "never before", "you won't believe", "everyone is saying",
    "they don't want you to know", "share before deleted",
    "100%", "guaranteed", "proven", "scientists baffled",
    "doctors hate", "one weird trick", "going viral"
]

CLICKBAIT_PHRASES = [
    "what happened next", "you need to see this", "mind blowing",
    "jaw dropping", "game changer", "this is huge", "must watch",
    "wake up", "open your eyes", "they are hiding"
]

# Pre-compute sets for O(1) word lookups (used for single-word lists only;
# multi-word phrases still need substring search).
_POSITIVE_SET = set(POSITIVE_WORDS)
_NEGATIVE_SET = set(NEGATIVE_WORDS)


def get_sentiment(text: str):
    """
    Very lightweight lexicon-based sentiment scorer.

    Returns:
        (polarity, subjectivity)
        polarity:     -1.0 (negative) to +1.0 (positive)
        subjectivity:  0.0 (objective) to 1.0 (subjective)
    """
    words = re.findall(r"\b\w+\b", text.lower())
    total = len(words)
    if total == 0:
        return 0.0, 0.0

    pos = sum(1 for w in words if w in _POSITIVE_SET)
    neg = sum(1 for w in words if w in _NEGATIVE_SET)
    emotional = pos + neg

    polarity = round((pos - neg) / emotional, 3) if emotional else 0.0
    subjectivity = round(min(emotional / (total * 0.1), 1.0), 3)
    return polarity, subjectivity


def get_credibility_scores(text: str, label: str, confidence: float):
    """
    Blend several rule-based signals into a 0-100 credibility score
    per category, used for the "Credibility Breakdown" bars in the UI.
    """
    text_lower = text.lower()
    words = text.split()

    # 1. Language score — fewer sensational words is better
    found_sensational = [w for w in SENSATIONAL_WORDS if w in text_lower]
    lang_score = max(0, 100 - len(found_sensational) * 15)

    # 2. Sentiment score — neutral, objective tone is more credible
    polarity, subjectivity = get_sentiment(text)
    sentiment_score = max(0, int(100 - abs(polarity) * 60 - subjectivity * 40))

    # 3. Length score — longer, more detailed articles score higher
    wc = len(words)
    if wc >= 300:
        length_score = 100
    elif wc >= 150:
        length_score = 75
    elif wc >= 100:
        length_score = 50
    else:
        length_score = 20

    # 4. Clickbait score — fewer bait phrases/exaggerations is better
    found_clickbait = [p for p in CLICKBAIT_PHRASES if p in text_lower]
    found_exag = [e for e in EXAGGERATIONS if e in text_lower]
    clickbait_score = max(0, 100 - (len(found_clickbait) + len(found_exag)) * 20)

    # 5. ML model's own confidence, normalized so "REAL" confidence
    #    maps to a high score and "FAKE" confidence maps to a low one
    ml_score = int(confidence) if label == "REAL" else int(100 - confidence)

    return {
        "Language":  min(100, lang_score),
        "Sentiment": min(100, sentiment_score),
        "Length":    min(100, length_score),
        "Clickbait": min(100, clickbait_score),
        "ML Model":  min(100, ml_score),
    }


def explain(text: str, label: str):
    """
    Build a human-readable list of reasons behind the verdict.

    Returns:
        (reasons: list[str], polarity: float, subjectivity: float)
    """
    reasons = []
    text_lower = text.lower()

    found_words = [w for w in SENSATIONAL_WORDS if w in text_lower]
    if found_words:
        reasons.append(f"⚠️ Contains sensational words: {', '.join(found_words[:4])}")

    polarity, subjectivity = get_sentiment(text)

    if polarity < -0.3:
        reasons.append(f"⚠️ Strongly negative tone (polarity: {polarity:.2f})")
    elif polarity > 0.3:
        reasons.append(f"ℹ️ Strongly positive tone (polarity: {polarity:.2f})")

    if subjectivity > 0.5:
        reasons.append(f"⚠️ Very subjective writing style (subjectivity: {subjectivity:.2f})")

    headline = text[:100]
    if headline.isupper():
        reasons.append("⚠️ Headline is ALL CAPS — common clickbait pattern")
    if headline.count("!") >= 2:
        reasons.append("⚠️ Excessive exclamation marks in headline")
    if headline.count("?") >= 2:
        reasons.append("⚠️ Multiple question marks — bait headline pattern")

    word_count = len(text.split())
    if word_count < 100:
        reasons.append(f"⚠️ Very short article ({word_count} words) — lacks detail")

    found_exag = [e for e in EXAGGERATIONS if e in text_lower]
    if found_exag:
        reasons.append(f"⚠️ Exaggerated claim: '{found_exag[0]}'")

    found_clickbait = [p for p in CLICKBAIT_PHRASES if p in text_lower]
    if found_clickbait:
        reasons.append(f"⚠️ Clickbait phrase detected: '{found_clickbait[0]}'")

    found_urdu = [w for w in URDU_FAKE_WORDS if w in text_lower]
    if found_urdu:
        reasons.append(f"ℹ️ Roman Urdu emotional words found: {', '.join(found_urdu[:3])}")

    # Headline vs body contradiction check
    lines = text.strip().split("\n")
    if len(lines) > 1:
        headline_words = set(re.findall(r"\b\w+\b", lines[0].lower()))
        body_words = set(re.findall(r"\b\w+\b", " ".join(lines[1:]).lower()))
        overlap = len(headline_words & body_words) / max(len(headline_words), 1)
        if overlap < 0.2:
            reasons.append("⚠️ Headline and body have low overlap — possible contradiction")

    if not reasons:
        if label == "REAL":
            reasons.append("✅ No sensational language detected")
            reasons.append("✅ Neutral and objective tone")
            reasons.append("✅ Sufficient article length and detail")
        else:
            reasons.append("⚠️ ML model detected fake patterns in writing style")

    return reasons, polarity, subjectivity