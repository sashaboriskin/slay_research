import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lm_polygraph.utils import WhiteboxModel, estimate_uncertainty
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.estimators import SemanticEntropy

model_path = "meta-llama/Llama-3.1-8B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map='auto',
)
base_model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
df = pd.read_csv('data/SemanticEntropy_dataset.csv')

generation_parameters = GenerationParameters(
    temperature=0.5,
    do_sample=True,
    num_beams=5,
)

model = WhiteboxModel(
    base_model, 
    tokenizer, 
    generation_parameters=generation_parameters
)

ue_method = SemanticEntropy()
answers = []
uncertainties = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    result = estimate_uncertainty(
        model,
        ue_method,
        input_text=row['question']
    )
    answers.append([text.replace('<|end_of_text|>', '') for text in result.generation_text])
    uncertainties.append(result.uncertainty)

df['answer'] = answers
df['SemanticEntropy'] = uncertainties
df.to_csv('data/SemanticEntropy_dataset.csv')
