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
    next: Literal["MedicationAgent", "ExerciseAgent", "DietAgent", "TransportAgent", "AppointmentAgent", "GeneralChatAgent", "FINISH"]

# --- Chains ---

# 1. Topic Check
topic_prompt = (
    "You are a medical context guardian. Patient has: **{allowed_disease}**.\n\n"
    "RULES:\n"
    "1. **Allow** Greetings, Small talk, Thank you -> 'on_topic'.\n"
    "2. **Allow** Questions about Medication, Diet, Exercise, Travel related to **{allowed_disease}** -> 'on_topic'.\n"
    "3. **Allow** Questions about context, for example, current diseases, appointment date.\n"
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
    "You are a router. Choose the agent that best matches the user's **PRIMARY INTENT**.\n"
    "ROUTING GUIDELINES:\n"
    "1. **MedicationAgent**: Drugs, dosage, side effects.\n"
    "2. **ExerciseAgent**: Workout, physical activity, tiredness.\n"
    "3. **DietAgent**: Food, hunger, menu, eating.\n"
    "4. **TransportAgent**: Travel, driving, flying, carrying items.\n"
    "5. **AppointmentAgent**: Scheduling, seeing doctor, postpone, change date.\n"
    "6. **GeneralChatAgent**: Greetings, emotions, small talk, current diseases, appointment date\n\n"
    
    "CONFLICT HANDLING:\n"
    "- 'Can I exercise after taking Insulin?' -> **ExerciseAgent** (Action is exercise)\n"
    "- 'Move my appointment to next week' -> **AppointmentAgent**\n"
)

supervisor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", supervisor_prompt), 
        MessagesPlaceholder("messages")
    ])
    | llm.with_structured_output(Router)
)