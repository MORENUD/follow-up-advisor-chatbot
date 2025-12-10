# graph.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from agent import AgentState, agent_runnables
from supervise import topic_check_chain, supervisor_chain

# --- 1. Alert Check ---
def check_alert_node(state: AgentState):
    ctx = state.get("user_context", {})
    try:
        alert_val = float(ctx.get("alert_level", 0.0))
    except:
        alert_val = 0.0
    
    if alert_val > 0.4:
        warning_msg = (
            f"🚨 **แจ้งเตือนความปลอดภัย:** ระดับความเสี่ยงของคุณสูง ({alert_val}) \n"
            "เพื่อความปลอดภัยสูงสุด **กรุณาไปพบแพทย์โดยด่วนนะครับ** ทางเราเป็นห่วงสุขภาพของคุณครับ"
            "โดยเราได้เลื่อนนัดให้คุณแล้ว สามารถเข้าไปตรวจสอบได้ที่ xxx.com"
        )
        return {"messages": [AIMessage(content=warning_msg)], "next": "END"}
    
    return {"next": "topic"}

# --- 2. Topic Check ---
def topic_node(state: AgentState):
    ctx = state.get("user_context", {})
    user_disease = ctx.get("disease", "Unknown")
    
    res = topic_check_chain.invoke({
        "messages": state["messages"],
        "allowed_disease": user_disease 
    })
    
    if res.decision == "off_topic":
        msg = (
            f"ขอโทษด้วยนะครับ เนื่องจากประวัติการรักษาของคุณเน้นที่ **{user_disease}** เป็นหลัก"
            f"หมอเลยขออนุญาตให้คำแนะนำเฉพาะเรื่อง **{user_disease}** เพื่อความปลอดภัยนะครับ"
            f"ว่าแต่ช่วงนี้อาการ **{user_disease}** ของคุณเป็นอย่างไรบ้างครับ?"
        )
        return {"messages": [AIMessage(content=msg)], "next": "END"}
    
    return {"next": "supervisor"}

# --- 3. Supervisor ---
def supervisor_node(state: AgentState):
    if not isinstance(state['messages'][-1], HumanMessage):
        return {"next": "END"}
        
    res = supervisor_chain.invoke({"messages": state["messages"]})
    if res.next == "FINISH":
        return {"next": "END"}
    return {"next": res.next}

# Helper to run agents
def run_agent(state: AgentState, agent_name: str):
    result = agent_runnables[agent_name].invoke(state)
    return {"messages": result["messages"]}

# --- Assembly ---
graph = StateGraph(AgentState)
graph.add_node("check_alert", check_alert_node)
graph.add_node("topic", topic_node)
graph.add_node("supervisor", supervisor_node)

# Dynamic Agent Node Creation
for name in agent_runnables:
    graph.add_node(name, lambda state, n=name: run_agent(state, n))

# Edges
graph.set_entry_point("check_alert")
graph.add_conditional_edges("check_alert", lambda x: x["next"], {"topic": "topic", "END": END})
graph.add_conditional_edges("topic", lambda x: x["next"], {"supervisor": "supervisor", "END": END})

# Router Edges
graph.add_conditional_edges("supervisor", lambda x: x["next"], {**{n: n for n in agent_runnables}, "END": END})

# Return edges
for name in agent_runnables:
    graph.add_edge(name, "supervisor")

memory = MemorySaver()
app = graph.compile(checkpointer=memory)