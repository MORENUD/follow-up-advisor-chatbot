from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from agent import AgentState, agent_runnables
from supervise import topic_check_chain, supervisor_chain

# --- 1. Alert Check ---
def check_alert_node(state: AgentState):
    ctx = state.get("user_context", {})
    
    is_alert_bool = str(ctx.get("is_alert", "false")).lower() == "true"
    try:
        alert_val = float(ctx.get("alert_level", 0.0))
    except:
        alert_val = 0.0
    
    if is_alert_bool or alert_val > 0.4:
        warning_msg = (
            f"🚨 **แจ้งเตือนความปลอดภัย:** ตรวจพบความเสี่ยงสูง ({alert_val}) \n"
            "กรุณาไปพบแพทย์โดยด่วนครับ"
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
        msg = f"ขออภัยครับ หมอขออนุญาตให้คำแนะนำเฉพาะเรื่อง **{user_disease}** นะครับ"
        return {"messages": [AIMessage(content=msg)], "next": "END"}
    
    return {"next": "supervisor"}

# --- 3. Supervisor ---
def supervisor_node(state: AgentState):
    """
    Supervisor จะอ่าน History ทั้งหมด แล้วตัดสินใจว่า:
    1. ส่งต่อให้ Specialist (Cardio, GI, etc.)
    2. หรือจบงาน (FINISH) เมื่อข้อมูลครบถ้วนแล้ว
    """

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

# Add Nodes
graph.add_node("check_alert", check_alert_node)
graph.add_node("topic", topic_node)
graph.add_node("supervisor", supervisor_node)

# Add Agent Nodes dynamically
for name in agent_runnables:
    graph.add_node(name, lambda state, n=name: run_agent(state, n))

# --- Edges & Wiring ---

# Entry Point
graph.set_entry_point("check_alert")

# 1. Alert -> Topic -> Supervisor
graph.add_conditional_edges(
    "check_alert", 
    lambda x: x["next"], 
    {"topic": "topic", "END": END}
)

graph.add_conditional_edges(
    "topic", 
    lambda x: x["next"], 
    {"supervisor": "supervisor", "END": END}
)

# 2. Supervisor Routing
mapping = {k: k for k in agent_runnables}
mapping["END"] = END
graph.add_conditional_edges("supervisor", lambda x: x["next"], mapping)

# 3. The Loop Back
for name in agent_runnables:
    graph.add_edge(name, "supervisor") 

# Compile
memory = MemorySaver()
app = graph.compile(checkpointer=memory)