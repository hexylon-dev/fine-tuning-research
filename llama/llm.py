from fastapi import FastAPI, HTTPException, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import pipeline, TextIteratorStreamer, AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from threading import Thread
import torch
import time
import logging
import json
from typing import AsyncIterator
from queue import Queue
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import httpx
from typing import List, Dict, Any
import numpy as np
import os
import io

# Initialize the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# model = ParlerTTSForConditionalGeneration.from_pretrained("HelpingAI/HelpingAI-TTS-v1").to("cpu")
# tokenizer = AutoTokenizer.from_pretrained("HelpingAI/HelpingAI-TTS-v1")
# description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# processor = WhisperProcessor.from_pretrained("openai/whisper-small")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# model.config.forced_decoder_ids = None

# Initialize the text generation pipeline
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
try:
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="mps",
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Define the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from this origin
    # allow_credentials=True,  # Allow cookies and credentials if required
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the request and response schemas
class QueryRequest(BaseModel):
    role: str
    content: str

class QueryResponse(BaseModel):
    response: str
    processing_time: float

class TTSRequest(BaseModel):
    prompt: str
    description: str

async def fetch_rag_context(query: str) -> str:
    """Helper function to fetch context from RAG server."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:6000/query",
                json={"query": query, "limit": 3},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json().get("context", "")
        except Exception as e:
            logger.error(f"Error calling RAG server: {e}")
            raise HTTPException(status_code=503, detail="RAG server error")

async def generate_stream(messages: list) -> AsyncIterator[str]:
    """Generate streaming response."""
    start_time = time.time()
    try:
        # Create a streamer object
        streamer = TextIteratorStreamer(
            tokenizer=text_generation_pipeline.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Prepare the inputs
        inputs = [{"role": message["role"], "content": message["content"]} for message in messages]

        logger.info(inputs)
        
        # Create a separate thread for generation
        def generate():
            text_generation_pipeline(
                inputs,
                max_new_tokens=256,
                streamer=streamer,
                num_return_sequences=1,
            )
        
        # Start generation in a separate thread
        thread = Thread(target=generate)
        thread.start()
        
        # Stream the generated text
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            chunk = new_text
            print(chunk)
            yield chunk
            
    except Exception as e:
        logger.error(f"Error in stream generation: {e}")
        error_chunk = {
            "error": str(e),
            "processing_time": time.time() - start_time
        }
        yield error_chunk

@app.post("/generate/stream")
async def generate_text_stream(request: QueryRequest):
    """Endpoint for streaming text generation."""
    context = await fetch_rag_context(query=request.content)
    
    messages = [
        {"role": "system", "content": context},
        {"role": request.role, "content": request.content}
    ]
    # if len(request.message) > 0:
        # messages.extend(request.message)
    logger.info(messages)
    
    return StreamingResponse(
        generate_stream(messages),
        media_type="text/event-stream"
    )

# async def generate_stream(messages: list) -> AsyncIterator[str]:
#     """Generate streaming response asynchronously without threading."""
#     start_time = time.time()
#     try:
#         # Prepare the inputs for text generation pipeline
#         inputs = [{"role": message["role"], "content": message["content"]} for message in messages]
#         logger.info(inputs)

#         # Create a streamer object
#         streamer = TextIteratorStreamer(
#             tokenizer=text_generation_pipeline.tokenizer,
#             skip_prompt=True,
#             skip_special_tokens=True,
#         )

#         # Generate text directly
#         text_generation_pipeline(
#             inputs,
#             max_new_tokens=256,
#             streamer=streamer,
#             num_return_sequences=1,
#         )

#         # Stream the generated text
#         for new_text in streamer:
#             logger.info(f"text: {new_text}")
#             yield new_text

#     except Exception as e:
#         logger.error(f"Error in stream generation: {e}")
#         yield {"error": str(e), "processing_time": time.time() - start_time}

# @app.websocket("/ws/generate")
# async def websocket_generate_text(websocket: WebSocket):
#     """WebSocket endpoint for streaming text generation."""
#     await websocket.accept()

#     try:
#         while True:
#             # Receive the message from the client
#             data = await websocket.receive_json()
#             logger.info(data)

#             # Validate the received data
#             if "content" not in data or "role" not in data:
#                 await websocket.send_json({"error": "Invalid request format."})
#                 continue

#             # Fetch context (if needed)
#             context = await fetch_rag_context(query=data["content"])

#             # Prepare messages
#             messages = [
#                 {"role": "system", "content": context},
#                 {"role": data["role"], "content": data["content"]},
#             ]

#             logger.info(messages)

#             # Stream the generated text
#             async for chunk in generate_stream(messages):
#                 logger.info(f"chunk: {chunk}")
#                 await websocket.send_text(chunk)

#     except WebSocketDisconnect:
#         logger.info("WebSocket connection closed.")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         await websocket.send_json({"error": str(e)})

# @app.post("/generate", response_model=QueryResponse)
# async def generate_text(request: QueryRequest):
#     """Endpoint that integrates RAG before generation."""
#     start_time = time.time()

#     # First, call the RAG server to get relevant context
#     context = await fetch_rag_context(query=request.content)

#     # Construct messages with RAG context
#     messages = [
#         {"role": "system", "content": f"Relevant context: {context}"},
#         {"role": request.role, "content": request.content},
#     ]

#     try:
#         outputs = text_generation_pipeline(
#             [{"role": message["role"], "content": message["content"]} for message in messages],
#             max_new_tokens=256,
#         )
#         response_text = outputs[0]["generated_text"]

#         logger.info(response_text)
        
#         if isinstance(response_text, list):
#             response_text = response_text[0].get("generated_text", "No response generated.")
#     except Exception as e:
#         logger.error(f"Error generating text: {e}")
#         raise HTTPException(status_code=500, detail="Error generating text")

#     processing_time = time.time() - start_time
#     logger.info(f"Request processed in {processing_time:.2f} seconds")

#     return QueryResponse(response=response_text, processing_time=processing_time)

# def reduce_noise(audio, noise_reduction_factor=0.005):
#     """
#     Reduce background noise using spectral gating.
#     """
#     noise_sample = audio[:int(0.1 * len(audio))]  # First 10% of the audio
#     noise_mean = np.mean(noise_sample)
#     noise_std = np.std(noise_sample)
#     denoised_audio = np.where(
#         np.abs(audio - noise_mean) > noise_reduction_factor * noise_std, 
#         audio, 
#         0
#     )
#     return denoised_audio


# def filter_silence(audio, threshold=0.01):
#     """
#     Filter out silent or very low-amplitude parts of the audio.
#     """
#     return audio[np.abs(audio) > threshold]


# @app.post("/transcribe")
# async def transcribe_audio(request: Request):
#     """
#     API endpoint to transcribe audio.
#     Accepts raw audio buffer as input and returns its transcription.
#     """
#     try:
#         # Read raw audio data from the request body
#         audio_data = await request.body()

#         # Convert raw audio data to a BytesIO object
#         audio_buffer = io.BytesIO(audio_data)

#         # Attempt to load audio using soundfile (sf) as an alternative to torchaudio
#         try:
#             # Read the audio buffer into an array using soundfile (handles more formats)
#             audio_array, sampling_rate = sf.read(audio_buffer)
#         except Exception as load_error:
#             logger.error(f"Error loading raw audio data with soundfile: {load_error}")
#             raise HTTPException(
#                 status_code=400, detail="Error loading raw audio data. Ensure the format is valid."
#             )

#         # Convert to Tensor for further processing in torchaudio
#         waveform = torch.tensor(audio_array).unsqueeze(0)

#         # Resample the audio if necessary (Whisper models expect 16 kHz audio)
#         if sampling_rate != 16000:
#             resampler = torch.hub.load('pytorch/vision', 'torchaudio.transforms.Resample', orig_freq=sampling_rate, new_freq=16000)
#             waveform = resampler(waveform)
#             sampling_rate = 16000

#         # Convert to mono if stereo
#         if waveform.size(0) > 1:
#             waveform = torch.mean(waveform, dim=0, keepdim=True)

#         # Preprocess the audio
#         waveform = waveform.squeeze().numpy()  # Convert from Tensor to NumPy
#         # denoised_waveform = reduce_noise(waveform)  # Apply noise reduction
#         # filtered_waveform = filter_silence(denoised_waveform)  # Remove silence

#         if len(waveform) < sampling_rate * 0.1:  # Less than 100ms of audio
#             raise HTTPException(
#                 status_code=400,
#                 detail="Audio too short or contains only silence after preprocessing."
#             )

#         # Convert the processed audio to input features
#         input_features = processor(
#             waveform,
#             sampling_rate=16000,
#             return_tensors="pt"
#         ).input_features

#         # Generate token IDs and decode them to text
#         predicted_ids = model.generate(input_features)
#         transcription = processor.batch_decode(
#             predicted_ids,
#             skip_special_tokens=True
#         )[0]

#         logger.info(f"Successfully transcribed audio: {transcription[:100]}...")

#         return JSONResponse(content={"transcription": transcription})

#     except Exception as e:
#         logger.error(f"Transcription error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/generate_audio/")
# async def generate_audio(request: TTSRequest):
#     try:
#         # Tokenize inputs
#         input_ids = description_tokenizer(request.description, return_tensors="pt").input_ids.to('cpu')
#         prompt_input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids.to('cpu')

#         # Generate audio
#         generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
#         audio_arr = generation.cpu().numpy().squeeze()

#         # Write audio to a BytesIO buffer
#         audio_buffer = BytesIO()
#         sf.write(audio_buffer, audio_arr, model.config.sampling_rate, format="WAV")
#         audio_buffer.seek(0)  # Move the pointer to the beginning of the buffer

#         # Return the audio file as a streaming response
#         return StreamingResponse(audio_buffer, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=output.wav"})

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")


@app.middleware("http")
async def log_request(request: Request, call_next):
    request_time = time.time()
    response = await call_next(request)
    duration = time.time() - request_time
    logger.info(f"{request.method} {request.url} completed in {duration:.2f}s")
    return response

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4040)