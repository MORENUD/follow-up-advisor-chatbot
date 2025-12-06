# tools.py
from langchain_core.tools import tool

# --- Medication ---
@tool
def get_diabetes_medication(query: str) -> str:
    """Use ONLY for Diabetes (เบาหวาน) medication."""
    return "ยาเบาหวาน: Metformin (ลดน้ำตาล), อินซูลิน. *ห้ามปรับยาเอง ระวังภาวะน้ำตาลต่ำ*"
@tool
def get_typhoid_medication(query: str) -> str:
    """Use ONLY for Typhoid (ไทฟอยด์) medication."""
    return "ยาไทฟอยด์: ต้องทานยาปฏิชีวนะ (Antibiotics) เช่น Ciprofloxacin ให้ครบโดส และทานพาราเซตามอลลดไข้"

# --- Exercise ---
@tool
def get_diabetes_exercise(query: str) -> str:
    """Use ONLY for Diabetes (เบาหวาน) exercise."""
    return "ออกกำลังกายเบาหวาน: แอโรบิก 30 นาที/วัน (เดินเร็ว, ว่ายน้ำ) ช่วยเพิ่มความไวอินซูลิน"
@tool
def get_typhoid_exercise(query: str) -> str:
    """Use ONLY for Typhoid (ไทฟอยด์) exercise."""
    return "ออกกำลังกายไทฟอยด์: **งดออกกำลังกาย** ต้องนอนพักผ่อน (Bed Rest) จนกว่าไข้จะลด"

# --- Diet ---
@tool
def get_diabetes_diet(query: str) -> str:
    """Use ONLY for Diabetes (เบาหวาน) diet."""
    return "อาหารเบาหวาน: เลี่ยงหวาน/แป้งขัดขาว. เน้นผักใบเขียว ข้าวกล้อง"
@tool
def get_typhoid_diet(query: str) -> str:
    """Use ONLY for Typhoid (ไทฟอยด์) diet."""
    return "อาหารไทฟอยด์: เน้น **'สุก ร้อน สะอาด'**. ทานอาหารอ่อน (โจ๊ก). ห้ามผักสด/ของหมักดอง/น้ำแข็ง"

# --- Groups ---
med_tools = [get_diabetes_medication, get_typhoid_medication]
exercise_tools = [get_diabetes_exercise, get_typhoid_exercise]
diet_tools = [get_diabetes_diet, get_typhoid_diet]
transport_tools = [get_diabetes_transport, get_typhoid_transport]