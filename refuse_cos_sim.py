from ast import literal_eval
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to('cuda')
model.eval()

df = pd.read_csv('data/alignscore_dataset.csv')
df['answer'] = df['answer'].apply(literal_eval) # convert string format to list

refuse_text = "I do not have information"
with torch.no_grad():
    ref_tokens = tokenizer(refuse_text, return_tensors='pt', padding=True, truncation=True).to('cuda')
    ref_output = model(**ref_tokens)
    ref_embedding = average_pool(ref_output.last_hidden_state, ref_tokens['attention_mask'])
    ref_embedding = F.normalize(ref_embedding, dim=1)

cos_sim_list = []
for answers in tqdm(df['answer']):
    encoded = tokenizer(
        answers,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to('cuda')

    with torch.no_grad():
        outputs = model(**encoded)
        embeddings = average_pool(outputs.last_hidden_state, encoded['attention_mask'])
        embeddings = F.normalize(embeddings, dim=1)

    similarities = (embeddings @ ref_embedding.T).squeeze().tolist()
    cos_sim_list.append(similarities)

df['cos_sim'] = cos_sim_list
df.to_csv('data/alignscore_dataset.csv', index=False)
