import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from io import BytesIO
import time

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Logs to console
)

# Device setup
device = "mps"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model and processor setup
model_id = "openai/whisper-large-v3-turbo"
logging.info("Loading model and processor...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Pipeline setup
logging.info("Setting up the pipeline...")
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Endpoint to accept raw audio buffer
@app.post("/transcribe/")
async def transcribe_audio(request: Request):
    try:
        # Log request received
        logging.info("Received transcription request.")

        # Read raw binary data from request
        audio_data = await request.body()
        
        # Ensure the request contains data
        if not audio_data:
            logging.error("No audio data received in the request.")
            raise HTTPException(status_code=400, detail="No audio data received.")

        # Wrap the raw binary data into a buffer
        audio_buffer = BytesIO(audio_data)
        logging.info("Audio data wrapped into a buffer.")

        # Time the inference
        start_time = time.time()
        logging.info("Starting transcription...")
        result = pipe(audio_buffer)
        inference_time = time.time()

        # Log success
        logging.info("Transcription completed successfully.")

        # Return the result
        return JSONResponse(
            content={
                "transcribed_text": result["text"],
                "inference_time_ms": round((inference_time - start_time) * 1000, 2),
            }
        )

    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
