from ast import literal_eval
import pandas as pd

def determine_class(row):
    if row['alignscore_sum'] > 9:
        return 1
    elif 2 < row['alignscore_sum'] <= 9:
        return 2
    elif all(val >= 0.75 for val in row['cos_sim']):
        return 3
    else:
        return 4

df = pd.read_csv('data/alignscore_dataset.csv')
df['cos_sim'] = df['cos_sim'].apply(literal_eval) # convert string format to list
df['class'] = df.apply(determine_class, axis=1)
df.to_csv('data/alignscore_dataset.csv', index=False)
