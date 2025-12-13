# llm_config.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

llm = ChatOpenAI(model="GPT-4.1 mini", temperature=0.1)
# llm = ChatOpenAI(model="gpt-5.2")