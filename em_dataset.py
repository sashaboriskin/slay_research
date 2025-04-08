import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_path = "meta-llama/Llama-3.1-8B-Instruct"
df = pd.read_csv("data/adaptive_rag_datasets.csv")
batch_size = 8
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map='auto',
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
generation_config = GenerationConfig(
    temperature=0.5,
    do_sample=True,
    num_beams=10,
    num_return_sequences=10,
    max_new_tokens=100,
)

sample_texts = []
em = []

for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i+batch_size]
    batch_prompts = []
    batch_references = []
    
    for _, row in batch.iterrows():
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}], 
            add_generation_prompt=True, 
            tokenize=False
        )
        batch_prompts.append(prompt)
        batch_references.append(row["reference"])
    
    inputs = tokenizer(
        batch_prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_sequences = outputs.reshape(
        len(batch_prompts), 
        generation_config.num_return_sequences, 
        -1
    )
    
    for idx in range(len(batch_prompts)):
        prompt_length = inputs["input_ids"][idx].shape[0]
        reference = str(batch_references[idx])
        em_count = 0
        
        for seq in generated_sequences[idx]:
            gen_text = tokenizer.decode(
                seq[prompt_length:], 
                skip_special_tokens=True
            )
            if reference.lower() in gen_text.lower():
                em_count += 1
        
        em.append(em_count)
        sample_texts.append(
            [
                tokenizer.decode(seq[prompt_length:], skip_special_tokens=True) 
                for seq in generated_sequences[idx]
            ]
        )

df['answer'] = sample_texts
df['em'] = em
df.to_csv("data/em_dataset.csv", index=False)