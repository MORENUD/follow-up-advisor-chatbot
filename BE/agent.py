# agent.py
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from llm_config import llm
from tools import med_tools, exercise_tools, diet_tools, transport_tools, appointment_tools

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_context: Dict[str, Any]

def create_system_prompt(template: str, context: Dict[str, Any]) -> str:
    """Helper เพื่อจัดการ Default Value และ Format String"""
    return template.format(
        user_name=context.get("user_name", "คนไข้"),
        disease=context.get("disease", "ไม่ทราบโรคที่เป็น"),
        current_schedule=context.get("current_schedule", "ยังไม่ได้นัดหมาย"),
        is_alert=context.get("is_alert", "Negative"),
        is_cardio=context.get("is_cardio", "Negative"),
        is_gi_liver=context.get("is_gi_liver", "Negative"),
        is_infectious=context.get("is_infectious", "Negative")
    )

def build_agent(llm, tools, system_template):
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot(state: AgentState):
        ctx = state.get("user_context", {})
        prompt = create_system_prompt(system_template, ctx)
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", chatbot)
    
    if tools:
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_edge("tools", "agent")
        workflow.add_conditional_edges("agent", tools_condition)
    else:
        workflow.add_edge("agent", END)

    workflow.add_edge(START, "agent")
    return workflow.compile()

# --- Base Template ---
base_template = """
คุณคือผู้เชี่ยวชาญดูแลคนไข้ชื่อ: {user_name} (โรค: {disease} อาการกำเริบ: {is_alert})
มีนัดหมายกับแพทย์ในวันที่: {current_schedule}

ความเสี่ยงของโรคแทรกซ้อน:
1. โรคเกี่ยวกับหัวใจ: {is_cardio}
2. โรคเกี่ยวกับระบบหายใจ: {is_gi_liver}
3. การติดเชื้อ: {is_infectious}

กฎ:
1. Empathy: นุ่มนวล ห่วงใย
2. Follow-up: จบด้วยคำถามกลับเสมอ
"""

# --- Build Agents ---
med_agent = build_agent(llm, med_tools, base_template + "\nหน้าที่: ยา/อาการป่วย")
exercise_agent = build_agent(llm, exercise_tools, base_template + "\nหน้าที่: กายภาพ/พักผ่อน")
diet_agent = build_agent(llm, diet_tools, base_template + "\nหน้าที่: อาหารการกิน")
transport_agent = build_agent(llm, transport_tools, base_template + "\nหน้าที่: การเดินทาง")
general_agent = build_agent(llm, [], base_template + "\nหน้าที่: พูดคุยทั่วไป")

appointment_agent = build_agent(
    llm, 
    appointment_tools, 
    base_template + """
    \nหน้าที่: จัดการเลื่อนนัดหมาย (Reschedule)
    
    \n***STRICT RULES***
    1. ห้ามเรียกใช้ Tool 'reschedule_appointment' เด็ดขาด หากผู้ใช้ **"ยังไม่ระบุวันที่จะมาใหม่"**
    2. หากผู้ใช้บอกแค่ว่า "ขอเลื่อนนัด" แต่ไม่บอกวันที่:
       - **ห้าม** คิดวันเอง
       - **ต้อง** ตอบกลับไปถามผู้ใช้ว่า "สะดวกเป็นวันไหนครับ?" หรือ "ต้องการเลื่อนไปเป็นวันที่เท่าไหร่ครับ?" เท่านั้น
    3. เมื่อได้วันที่ครบถ้วนแล้ว จึงค่อยเรียก Tool
    """
)

agent_runnables = {
    "MedicationAgent": med_agent,
    "ExerciseAgent": exercise_agent,
    "DietAgent": diet_agent,
    "TransportAgent": transport_agent,
    "AppointmentAgent": appointment_agent,
    "GeneralChatAgent": general_agent,
}