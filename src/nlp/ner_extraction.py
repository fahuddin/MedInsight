import spacy 

nlp = spacy.load("en_core_sci_sm")

def extract_entities(text):
    return [(ent.text, ent.label_) for ent in nlp(text).ents]