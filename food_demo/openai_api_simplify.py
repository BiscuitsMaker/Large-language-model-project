import time
import torch
import uvicorn
import json
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

MAX_MODEL_LENGTH = 8192

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 1024

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]

class ChatCompletionResponse(BaseModel):
    model: str
    id: Optional[str] = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: Literal["chat.completion"]
    choices: List[ChatCompletionResponseChoice]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    inputs = tokenizer.apply_chat_template(request.messages, add_generation_prompt=True, tokenize=False)
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        stop_token_ids=[151329, 151336, 151338],
    )

    response_text = ""
    async for output in generate_stream_glm4(inputs, sampling_params):
        response_text = output["text"]

    if response_text.startswith("\n"):
        response_text = response_text[1:].strip()

    message = ChatMessage(role="assistant", content=response_text)
    choice_data = ChatCompletionResponseChoice(index=0, message=message, finish_reason="stop")

    return ChatCompletionResponse(
        model=request.model,
        choices=[choice_data],
        object="chat.completion"
    )

@torch.inference_mode()
async def generate_stream_glm4(inputs, sampling_params):
    async for output in engine.generate(prompt=inputs, sampling_params=sampling_params, request_id=f"{time.time()}"):
        yield {"text": output.outputs[0].text}

@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)

if __name__ == "__main__":
    MODEL_PATH = "../glm-4-9b-chat"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        tensor_parallel_size=1,
        dtype="half",
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    uvicorn.run(app, host='0.0.0.0', port=8001, workers=1)

