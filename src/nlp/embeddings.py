from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state: shape = [batch_size, seq_len, hidden_dim]
    # mean across all tokens → shape: [hidden_dim]
    return outputs.last_hidden_state.mean(dim=1).squeeze()