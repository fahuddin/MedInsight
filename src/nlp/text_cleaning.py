import re 
import spacy

nlp = spacy.load("en_core_sci_sm")

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s.,]", "", text)
    text = text.lower().strip()
    return text

def lemmatize(text) -> str:
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])