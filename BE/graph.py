# graph.py
from functools import partial
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from agent import AgentState, agent_runnables
from supervise import topic_check_chain, supervisor_chain

# --- 1. Alert Check ---
def check_alert_node(state: AgentState):
    ctx = state.get("user_context", {})
    alert_status = str(ctx.get("is_alert", "Negative")).strip().lower()
    
    if alert_status == "positive":
        warning_msg = (
            "🚨 **แจ้งเตือนความปลอดภัย:** ระบบตรวจพบความเสี่ยงอาการกำเริบ\n"
            "กรุณาไปพบแพทย์โดยด่วน ทางเราได้ดำเนินการเลื่อนนัดให้แล้ว ตรวจสอบที่ xxx.com"
        )
        return {
            "messages": [AIMessage(content=warning_msg)],
            "next": "END"
        }
    
    return {"next": "topic"}

# --- 2. Topic Check ---
def topic_node(state: AgentState):
    ctx = state.get("user_context", {})
    user_disease = ctx.get("disease", "Unknown")
    last_message = state["messages"][-1]
    
    res = topic_check_chain.invoke({
        "messages": [last_message],
        "allowed_disease": user_disease 
    })
    
    if res.decision == "off_topic":
        msg = f"ขออภัยครับ หมอขออนุญาตให้คำแนะนำเฉพาะเรื่อง **{user_disease}** เพื่อความปลอดภัยนะครับ"
        return {
            "messages": [AIMessage(content=msg)],
            "next": "END"
        }
    
    return {"next": "supervisor"}

# --- 3. Supervisor ---
def supervisor_node(state: AgentState):
    if not state['messages'] or not isinstance(state['messages'][-1], HumanMessage):
        return {"next": "END"}
        
    res = supervisor_chain.invoke({"messages": state["messages"]})
    return {"next": res.next}

# Helper to run agents
def run_agent_node(state: AgentState, agent_name: str):
    result = agent_runnables[agent_name].invoke(state)
    return {"messages": result["messages"]}

# --- Assembly ---
graph = StateGraph(AgentState)

graph.add_node("check_alert", check_alert_node)
graph.add_node("topic", topic_node)
graph.add_node("supervisor", supervisor_node)

# Dynamic Agent Node Creation
for name in agent_runnables:
    graph.add_node(name, partial(run_agent_node, agent_name=name))
    graph.add_edge(name, "supervisor")

# Set Entry Point
graph.add_edge(START, "check_alert")

# Conditional Edges
def route_after_alert(x):
    return END if x.get("next") == "END" else "topic"

def route_after_topic(x):
    return END if x.get("next") == "END" else "supervisor"

def route_supervisor(x):
    destination = x.get("next")
    if destination == "FINISH" or destination not in agent_runnables:
        return END
    return destination

graph.add_conditional_edges("check_alert", route_after_alert)
graph.add_conditional_edges("topic", route_after_topic)
graph.add_conditional_edges("supervisor", route_supervisor)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)