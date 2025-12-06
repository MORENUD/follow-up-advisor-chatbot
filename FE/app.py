import streamlit as st
import requests
import uuid
import json
import os

st.set_page_config(page_title="Medical AI", page_icon="🏥")

# --- Configuration ---
BACKEND_URL = os.getenv("API_URL", "http://127.0.0.1:8000") 

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

params = st.query_params
user_context = {
    "user_name": params.get("user_name", "คุณผู้ใช้"),
    "disease": params.get("disease", "Unknown"),
    "alert_level": params.get("alert", "0.0")
}

# Welcome Message
if "messages" not in st.session_state:
    welcome = (
        f"สวัสดีครับคุณ **{user_context['user_name']}** 😊\n\n"
        f"วันนี้เป็นอย่างไรบ้างครับ? ผมพร้อมดูแลเรื่อง **{user_context['disease']}** ของคุณนะครับ "
        "ไม่ว่าจะเป็นเรื่องอาหาร ยา การเดินทาง หรือแค่อยากระบายอาการป่วย ก็พิมพ์มาได้เลยนะครับ"
    )
    st.session_state.messages = [{"role": "assistant", "content": welcome}]

# Chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("พิมพ์ข้อความ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_res = ""
        
        try:
            payload = {
                "query": prompt,
                "user_context": user_context,
                "thread_id": st.session_state.session_id
            }

            api_endpoint = f"{BACKEND_URL}/chat"
            
            with requests.post(api_endpoint, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data: "):
                            json_str = decoded[6:]
                            try:
                                content = json.loads(json_str)
                                full_res = content
                                placeholder.markdown(full_res)
                            except json.JSONDecodeError:
                                pass
            
            st.session_state.messages.append({"role": "assistant", "content": full_res})

        except Exception as e:
            st.error(f"Error: {e}")