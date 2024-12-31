import transformers
import torch
from transformers import pipeline

model = "meta-llama/Meta-Llama-3-8B"
generator = pipeline("text-generation", model=model).to("cuda")
input_text = "Once upon a time"
generated_text = generator(input_text, max_length=100, num_return_sequences=1)

print(generated_text)
