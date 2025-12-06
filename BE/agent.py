# agent.py
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from llm_config import llm
from tools import med_tools, exercise_tools, diet_tools, transport_tools

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_context: Dict[str, Any]

def build_agent(llm: ChatOpenAI, tools: list, system_template: str):
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: AgentState):
        ctx = state.get("user_context", {})
        prompt = system_template.format(
            user_name=ctx.get("user_name", "คุณ"), 
            disease=ctx.get("disease", "โรคประจำตัว"), 
            alert=ctx.get("alert_level", "0")
        )
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", chatbot)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile()

# --- Context Template ---
base_template = """
คุณคือผู้เชี่ยวชาญดูแลคนไข้ชื่อ: {user_name}
โรคประจำตัว: {disease} (Alert: {alert})

คำแนะนำการตอบ:
1. อิงบริบทโรค '{disease}' เสมอ
2. **Empathy:** ใช้ภาษาที่นุ่มนวล เป็นกันเอง แสดงความห่วงใย (เช่น "เข้าใจเลยครับว่าทานยาก...")
3. **Follow-up:** ห้ามตอบห้วนๆ! **ต้องจบประโยคด้วยคำถามกลับเสมอ** เพื่อชวนคุยต่อ หรือประเมินอาการเพิ่ม
   (ตัวอย่าง: "...ทานได้ครับแต่ต้องน้อยหน่อย ช่วงนี้คุณ {user_name} คุมน้ำตาลได้ดีไหมครับ?")
"""

# Medicine Agents
med_agent = build_agent(llm, med_tools, base_template + "\nหน้าที่: ให้คำปรึกษาเรื่องยา/อาการป่วย")

# Exercise Agents
exercise_agent = build_agent(llm, exercise_tools, base_template + "\nหน้าที่: ให้คำปรึกษาเรื่องกายภาพ/การพักผ่อน")

# Diet Agents
diet_agent = build_agent(llm, diet_tools, base_template + "\nหน้าที่: ให้คำปรึกษาเรื่องอาหารการกิน")

# Transport Agent
transport_agent = build_agent(llm, transport_tools, base_template + "\nหน้าที่: ให้คำปรึกษาเรื่องการเดินทาง การขับรถ หรือการเตรียมตัวขึ้นเครื่องบิน")

# General Chat Agent
general_agent = build_agent(llm, [], base_template + "\nหน้าที่: พูดคุยทั่วไป ทักทาย ให้กำลังใจ รับฟังอาการเบื้องต้น (ไม่ต้องใช้ Tools)")

agent_runnables = {
    "MedicationAgent": med_agent,
    "ExerciseAgent": exercise_agent,
    "DietAgent": diet_agent,
    "TransportAgent": transport_agent,
    "GeneralChatAgent": general_agent,
}