# main.py
import json
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from graph import app as graph_app

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield 

app = FastAPI(title="Medical Chatbot API", lifespan=lifespan)

# --- CORS Setup ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Model ---
class ChatRequest(BaseModel):
    query: str
    user_context: Dict[str, Any]
    thread_id: str

# --- Chat Endpoint ---
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    async def event_stream():
        config = {"configurable": {"thread_id": req.thread_id}}
        inputs = {
            "messages": [HumanMessage(content=req.query)],
            "user_context": req.user_context 
        }
        
        async for event in graph_app.astream(inputs, config=config, stream_mode="updates"):
            for node, output in event.items():
                if output and "messages" in output:
                    last_msg = output["messages"][-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        if last_msg.type == "ai":
                            cleaned_content = json.dumps(last_msg.content, ensure_ascii=False)
                            yield f"data: {cleaned_content}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)