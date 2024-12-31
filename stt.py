import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

# Device setup
device = "mps"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model and processor loading
model_id = "openai/whisper-large-v3-turbo"

start_time = time.time()
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Pipeline creation
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)
pipeline_time = time.time()

# Dataset loading
dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
dataset_time = time.time()

# Sample inference
sample = dataset[0]["audio"]
result = pipe("audio.mp3")
inference_time = time.time()

# Printing results and execution times
print(result["text"])
print(f"Model and processor loading time: {(pipeline_time - start_time) * 1000:.2f} ms")
print(f"Dataset loading time: {(dataset_time - pipeline_time) * 1000:.2f} ms")
print(f"Inference time: {(inference_time - dataset_time) * 1000:.2f} ms")
