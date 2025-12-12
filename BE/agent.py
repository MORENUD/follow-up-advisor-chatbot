from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from llm_config import llm
from tools import med_tools, exercise_tools, diet_tools, transport_tools, appointment_tools

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_context: Dict[str, Any]

def parse_bool(value):
    if str(value).lower() == 'true':
        return True
    return False

def build_agent(llm, tools, system_template):
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot(state: AgentState):
        ctx = state.get("user_context", {})
        
        user_name = ctx.get("user_name", "คุณ")
        disease = ctx.get("disease", "โรคประจำตัว")
        current_schedule = ctx.get("current_schedule", "ไม่ระบุ")
        is_alert = parse_bool(ctx.get("is_alert", "false"))
        is_cardio = parse_bool(ctx.get("is_cardio", "false"))
        is_gi_liver = parse_bool(ctx.get("is_gi_liver", "false"))
        is_infectious = parse_bool(ctx.get("is_infectious", "false"))

        prompt = system_template.format(user_name=user_name,
                                        disease=disease,
                                        is_alert=is_alert,                                        
                                        current_schedule=current_schedule,
                                        is_cardio=is_cardio,
                                        is_gi_liver=is_gi_liver,
                                        is_infectious=is_infectious)
        
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
        workflow.add_edge("agent", "__end__")

    workflow.set_entry_point("agent")
    return workflow.compile()

# --- Base Template ---
base_template = """
คุณคือผู้เชี่ยวชาญดูแลคนไข้ชื่อ: {user_name} (โรค: {disease} อาการกำเริบ: {is_alert})
มีนัดหมายกับแพทย์ในวันที่: {current_schedule} (Appointment date)

อาการอื่นแทรกซ้อน
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
general_agent = build_agent(llm, [], base_template + "\nหน้าที่: พูดคุยทั่วไป และตอบข้อมูลเบื้องต้นของผู้ใช้ เช่น ชื่อคนไข้ โรค อาการกำเริบ วันนัดหมาย และอาการอื่นแทรกซ้อน")

# New Appointment Agent
appointment_agent = build_agent(
    llm, 
    appointment_tools, 
    base_template + """
    \nหน้าที่: จัดการเลื่อนนัดหมาย เมื่อเลื่อนนัดเสร็จแล้ว ให้ผู้ใช้เข้าไปตรวจสอบที่ xxx.com
    \nสำคัญ: ถ้าลูกค้ายังไม่ระบุวันใหม่ ให้ถามก่อนเรียก Tool 'reschedule_appointment'
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