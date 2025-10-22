import spacy

# Load scispaCy small biomedical model
# Try to load the model
try:
    nlp = spacy.load("en_core_sci_lg")
except OSError:
    # fallback logic if model not installed
    print("⚠️ Model en_core_sci_lg not found. Please install it.")
    raise

# Then use nlp to process text
doc = nlp("Your biomedical/clinical text here …")

def extract_entities(text: str) -> dict:
    """
    Extract named entities from clinical text and return a feature dictionary.
    Example: {"SYMPTOM_shortness_of_breath": 1, "DISEASE_diabetes": 1}
    """
    doc = nlp(text)
    features = {}

    for ent in doc.ents:
        # Clean entity text (lowercase, underscores for spaces)
        ent_text = ent.text.strip().lower().replace(" ", "_")
        ent_label = ent.label_.upper()

        key = f"{ent_label}_{ent_text}"
        features[key] = 1   # binary flag (present)

    return features
