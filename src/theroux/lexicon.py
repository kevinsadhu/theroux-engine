"""
THE TUNABLE RUBRIC.

This file is the scoring methodology. It is the part of the system worth
protecting and the part worth iterating on. Everything else is plumbing.

Each dimension has a research rationale. Keep the rationale attached to the
lexicon — when someone asks why we measure a thing, the answer lives here.
"""

# ---------------------------------------------------------------------------
# HEDGING
# Qualifiers and conditionals used where a direct answer was available.
# Larcker & Zakolyukina found elevated hedging in calls later restated.
# ---------------------------------------------------------------------------
HEDGES = [
    "i think", "i believe", "probably", "possibly", "perhaps", "maybe",
    "sort of", "kind of", "somewhat", "generally", "broadly", "fair to say",
    "it's fair", "we'll see", "hopefully", "could be", "might", "may be",
    "tend to", "relatively", "arguably", "to some extent", "more or less",
    "i would say", "i'd say", "in some ways", "reasonably", "roughly speaking",
    "for the most part", "by and large", "at this point", "as things stand",
]

# ---------------------------------------------------------------------------
# TOPIC DEFLECTION
# Redirecting from the question asked toward an adjacent, safer frame.
# The classic pre-disclosure move: answer the multi-year question instead.
# ---------------------------------------------------------------------------
DEFLECTORS = [
    "but what i'd really", "the bigger picture", "stepping back", "zoom out",
    "what matters more", "let me reframe", "the real story", "more broadly",
    "i'd point you to", "what i would focus on", "the way i think about",
    "long-term", "over time", "down the road", "in the fullness of time",
    "the durability of", "the multi-year", "structurally",
]

# ---------------------------------------------------------------------------
# CONFIDENCE LANGUAGE
# Overt certainty markers. Scored as its own dimension rather than assumed
# to point one way: it can indicate genuine strength OR overcompensation.
# Let the baseline and the outcome data decide the sign.
# ---------------------------------------------------------------------------
CONFIDENCE = [
    "absolutely", "clearly", "no question", "without a doubt", "certainly",
    "definitely", "obviously", "unquestionably", "for sure", "100%",
    "very confident", "extremely", "tremendous", "never been stronger",
    "couldn't be more", "thrilled",
]

# ---------------------------------------------------------------------------
# PRONOUN DISTANCING
# Movement from first-person ownership to impersonal framing at moments of
# exposure. Reduced self-reference is one of the more replicated findings
# in the deception-linguistics literature.
# ---------------------------------------------------------------------------
FIRST_PERSON = [
    " i ", " we ", " our ", " us ", " i'm ", " we're ", " we've ", " i've ",
    " i'll ", " we'll ", " my ", " i'd ",
]

IMPERSONAL = [
    "the company", "the business", "the organization", "the team",
    "it was decided", "there is", "there are", "one would", "the market",
    "the environment", "conditions", "the segment", "the industry",
    "the category", "the sector", "management",
]

# ---------------------------------------------------------------------------
# SPECIFICITY AVOIDANCE
# Vagueness where a number, date, or named fact would normally appear.
# Measured as the *absence* of concrete markers, which is why it is a regex
# rather than a word list. This is the highest-weighted dimension: a speaker
# who normally quotes basis points and suddenly speaks only in adjectives is
# the cleanest observable signal in the corpus.
# ---------------------------------------------------------------------------
NUMERIC_PATTERN = (
    r"\b\d[\d,\.]*\s*(%|percent|bps|basis points|million|billion|bn|m\b|k\b|x\b)?"
)

PERIOD_PATTERN = (
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|q1|q2|q3|q4|fiscal|first quarter|second quarter|"
    r"third quarter|fourth quarter|full year|year over year|sequentially)\b"
)

# ---------------------------------------------------------------------------
# NORMALISERS
# Divide raw per-100-word rates by these to land dimensions on 0-1.
# Tune against a real corpus; these are calibrated to the seed corpus.
# ---------------------------------------------------------------------------
NORMALISERS = {
    "hedging": 4.0,
    "topic_deflection": 2.0,
    "confidence_language": 2.0,
    "specificity_density_full_marks": 6.0,  # concrete markers/100w scoring 0
}

DIMENSIONS = [
    "hedging",
    "specificity_avoidance",
    "pronoun_distancing",
    "topic_deflection",
    "confidence_language",
]

DIMENSION_LABELS = {
    "hedging": "Hedging",
    "specificity_avoidance": "Specificity avoidance",
    "pronoun_distancing": "Pronoun distancing",
    "topic_deflection": "Topic deflection",
    "confidence_language": "Confidence language",
}
