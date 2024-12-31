import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import torch
import torchaudio
from torchaudio.transforms import Resample
from pydub import AudioSegment
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

# Enable CORS for localhost:3001
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Allow only this origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Device setup
device = "mps"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model and processor setup
model_id = "facebook/s2t-small-librispeech-asr"
logging.info("Loading model and processor...")
model = Speech2TextForConditionalGeneration.from_pretrained(model_id)
model.to(device)
processor = Speech2TextProcessor.from_pretrained(model_id)

# Pipeline setup (we will directly use the model and processor instead of pipeline)
logging.info("Model and processor loaded successfully.")

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

        # Convert the audio buffer to WAV using pydub (ensure it's in a compatible format)
        audio = AudioSegment.from_file(audio_buffer)  # Automatically detects the format (e.g., MP3)
        logging.info("Audio converted to WAV format using pydub.")

        # Export the audio as WAV to a new buffer
        wav_buffer = BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        # Load the audio buffer into a waveform and sampling rate
        waveform, sample_rate = torchaudio.load(wav_buffer)
        logging.info(f"Loaded audio waveform with sample rate {sample_rate}.")

        # If the sample rate is not 16000, resample the audio to 16000 Hz
        if sample_rate != 16000:
            logging.info(f"Resampling audio from {sample_rate} to 16000 Hz.")
            resample_transform = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resample_transform(waveform)
            sample_rate = 16000
            logging.info(f"Resampled audio to 16000 Hz.")

        # Process the audio input for the model
        logging.info("Processing audio input for model...")
        inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt").to(device)

        # Time the inference
        start_time = time.time()
        logging.info("Starting transcription...")
        generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        inference_time = time.time()

        # Log success
        logging.info("Transcription completed successfully.")

        # Return the result
        return JSONResponse(
            content={
                "transcribed_text": transcription[0],  # First result as it is batch decoding
                "inference_time_ms": round((inference_time - start_time) * 1000, 2),
            }
        )

    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
