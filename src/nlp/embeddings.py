from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device).eval()

@torch.no_grad()
def embed_text(texts, normalize=True, use_last4=False, max_length=512):
    """
    texts: str or list[str]
    returns: (n, hidden) torch.FloatTensor on CPU
    """
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    outputs = model(**inputs, output_hidden_states=use_last4)

    if use_last4:
        # Stack last 4 hidden states and mean them: (batch, seq, hidden)
        hidden = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0)
    else:
        hidden = outputs.last_hidden_state  # (batch, seq, hidden)

    # Masked mean pooling over tokens (ignore PAD)
    special = set(tokenizer.all_special_ids)
    mask = inputs["attention_mask"].clone()
    ids = inputs["input_ids"]
    mask[id.isin(torch.tensor(list(special), device=ids.device))] = 0
    attn = mask.unsqueeze(-1)
    summed = (hidden * attn).sum(dim=1)
    counts = attn.sum(dim=1).clamp(min=1)
    emb = summed / counts  # (batch, hidden)

    if normalize:
        emb = F.normalize(emb, p=2, dim=1)

    return emb.cpu()

