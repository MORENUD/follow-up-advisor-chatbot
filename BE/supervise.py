# supervise.py
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llm_config import llm

# --- Models ---
class TopicClassifier(BaseModel):
    decision: Literal["on_topic", "off_topic"] = Field(
        ...,
        description="Return 'on_topic' if relevant to the disease OR if it is a greeting/general chat."
    )

class Router(BaseModel):
    next: Literal["MedicationAgent", "ExerciseAgent", "DietAgent", "TransportAgent", "GeneralChatAgent", "FINISH"]

# --- Chains ---

# 1. Topic Check
topic_prompt = (
    "You are a medical context guardian. Patient has: **{allowed_disease}**.\n\n"
    "RULES:\n"
    "1. **Allow** Greetings, Small talk, Thank you -> 'on_topic'.\n"
    "2. **Allow** Questions about Medication, Diet, Exercise, Travel related to **{allowed_disease}** -> 'on_topic'.\n"
    "3. **REJECT** Questions about OTHER diseases (e.g. Cancer, HIV) -> 'off_topic'.\n"
)

topic_check_chain = (
    ChatPromptTemplate.from_messages([
        ("system", topic_prompt),
        MessagesPlaceholder("messages")
    ])
    | llm.with_structured_output(TopicClassifier)
)

# 2. Supervisor
supervisor_prompt = (
    "Route user input to the right agent:\n"
    "- Symptoms (Fever, Pain, Infection) -> MedicationAgent\n"
    "- Weakness, Fatigue, Rest, Activity -> ExerciseAgent\n"
    "- Hunger, Thirst, Menu, Eating -> DietAgent\n"
    "- Travel, Driving, Flying, Commute, Car -> TransportAgent\n"
    "- Hello, Thanks, Sadness, Small talk -> GeneralChatAgent\n"
    "- Bye -> FINISH"
)

supervisor_chain = (
    ChatPromptTemplate.from_messages([("system", supervisor_prompt), MessagesPlaceholder("messages")])
    | llm.with_structured_output(Router)
)