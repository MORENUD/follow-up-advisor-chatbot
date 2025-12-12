# supervise.py
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llm_config import llm

# --- Models ---

class TopicClassifier(BaseModel):
    decision: Literal["on_topic", "off_topic"] = Field(
        ...,
        description="Return 'on_topic' if relevant to the disease OR if it is a greeting/general chat. Return 'off_topic' only for totally unrelated diseases (e.g. Cancer, HIV when user has Diabetes)."
    )

class Router(BaseModel):
    reasoning: str = Field(
        ...,
        description="Analyze the conversation history. What has been answered? What is still pending? explain why you choose the next agent."
    )
    next: Literal[
        "MedicationAgent", 
        "ExerciseAgent", 
        "DietAgent", 
        "TransportAgent", 
        "AppointmentAgent", 
        "GeneralChatAgent", 
        "FINISH"
    ]

# --- Chains ---

# 1. Topic Check
topic_prompt = (
    "You are a medical context guardian. Patient has: **{allowed_disease}**.\n\n"
    "RULES:\n"
    "1. **Allow** Greetings, Small talk, Thank you -> 'on_topic'.\n"
    "2. **Allow** Questions about Medication, Diet, Exercise, Travel related to **{allowed_disease}** -> 'on_topic'.\n"
    "3. **Allow** Questions about context, for example, current diseases, current appointment date -> 'on_topic'.\n"
    "4. **REJECT** Questions about OTHER diseases (e.g. Cancer, HIV) -> 'off_topic'.\n"
)

topic_check_chain = (
    ChatPromptTemplate.from_messages([
        ("system", topic_prompt),
        MessagesPlaceholder("messages")
    ])
    | llm.with_structured_output(TopicClassifier)
)

# 2. Supervisor (Router)
supervisor_prompt = (
    "You are a router. Your goal is to manage the conversation flow to address ALL user needs.\n"
    "Current Agents:\n"
    "1. **MedicationAgent**: Drugs, dosage, side effects.\n"
    "2. **ExerciseAgent**: Workout, physical activity, tiredness.\n"
    "3. **DietAgent**: Food, hunger, menu, eating.\n"
    "4. **TransportAgent**: Travel, driving, flying, carrying items.\n"
    "5. **AppointmentAgent**: Scheduling, seeing doctor, postpone, change date.\n"
    "6. **GeneralChatAgent**: Greetings, emotions, small talk.\n\n"
    
    "ROUTING LOGIC (IMPORTANT):\n"
    "- Analyze the user's latest message AND the conversation history.\n"
    "- If the user asks MULTIPLE questions (e.g., 'Change appointment AND what to eat'), you must address them ONE BY ONE.\n"
    "- **STEP 1**: Pick the first relevant agent.\n"
    "- **STEP 2**: Wait for that agent to respond (look at the history).\n"
    "- **STEP 3**: If there are still unanswered parts of the question, pick the NEXT relevant agent.\n"
    "- **FINISH**: Select 'FINISH' ONLY when ALL parts of the user's input have been addressed.\n\n"

    "EXAMPLE:\n"
    "User: 'Change appointment to Monday and is it okay to eat Durian?'\n"
    "Turn 1: Select 'AppointmentAgent' (Reasoning: I need to handle the appointment change first.)\n"
    "...AppointmentAgent responds...\n"
    "Turn 2: Select 'DietAgent' (Reasoning: Appointment is done, but the user also asked about eating Durian.)\n"
    "...DietAgent responds...\n"
    "Turn 3: Select 'FINISH' (Reasoning: Both appointment and diet questions have been answered.)\n"
)

supervisor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", supervisor_prompt), 
        MessagesPlaceholder("messages")
    ])
    | llm.with_structured_output(Router)
)