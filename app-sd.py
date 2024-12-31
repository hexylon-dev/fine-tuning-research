import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pyannote.audio import Pipeline
from io import BytesIO
import time

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Allow only this origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Logs to console
)

# Initialize pyannote.audio diarization pipeline
logging.info("Loading pyannote diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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"  # Replace with your Hugging Face token
)

# Endpoint to accept raw audio buffer for speaker diarization
@app.post("/diarize/")
async def diarize_audio(request: Request):
    try:
        # Log request received
        logging.info("Received diarization request.")

        # Read raw binary data from request
        audio_data = await request.body()

        # Ensure the request contains data
        if not audio_data:
            logging.error("No audio data received in the request.")
            raise HTTPException(status_code=400, detail="No audio data received.")

        # Wrap the raw binary data into a buffer
        audio_buffer = BytesIO(audio_data)
        logging.info("Audio data wrapped into a buffer.")

        # Save the raw audio buffer to a file (temporary step before processing)
        with open("temp_audio.wav", "wb") as temp_audio_file:
            temp_audio_file.write(audio_data)
        logging.info("Audio file saved temporarily for diarization.")

        # Time the diarization process
        start_time = time.time()
        logging.info("Starting diarization...")

        # Run the diarization pipeline on the audio file
        diarization = pipeline("temp_audio.wav")

        # Write the diarization result to RTTM format
        with open("audio.rttm", "w") as rttm_file:
            diarization.write_rttm(rttm_file)

        inference_time = time.time()

        # Log success
        logging.info("Diarization completed successfully.")

        # Return the result
        return JSONResponse(
            content={
                "message": "Diarization completed and saved in 'audio.rttm'",
                "inference_time_ms": round((inference_time - start_time) * 1000, 2),
            }
        )

    except Exception as e:
        logging.error(f"Error during diarization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
