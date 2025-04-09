from collections import defaultdict
from ast import literal_eval
import pandas as pd
from alignscore import AlignScore

scorer = AlignScore(
    model='roberta-base', 
    batch_size=32, 
    device='cuda', 
    ckpt_path='models/AlignScore-base.ckpt', 
    evaluation_mode='nli_sp'
)
df = pd.read_csv('data/em_dataset.csv').drop(columns=['em'])
df['answer'] = df['answer'].apply(literal_eval) # convert string format to list
all_contexts = []
all_claims = []
indices = []

for idx, row in df.iterrows():
    reference = str(row['reference'])
    answers = [ans for ans in row['answer']]
    
    all_contexts.extend(answers)
    all_claims.extend([reference] * len(answers))
    indices.extend([idx] * len(answers))

scores = scorer.score(contexts=all_contexts, claims=all_claims)
score_dict = defaultdict(list)

for idx, score in zip(indices, scores):
    score_dict[idx].append(score)

df['alignscore'] = df.index.map(lambda x: score_dict.get(x, []))
df['alignscore_sum'] = df['alignscore'].apply(sum)
df.to_csv('data/alignscore_dataset.csv', index=False)