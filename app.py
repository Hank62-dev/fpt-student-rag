# app.py
import streamlit as st
from src.agent import ask_question
import time

st.set_page_config(page_title="FPT Student Assistant", page_icon="🧑‍🎓", layout="centered")

st.title("🧑‍🎓 FPT Student Assistant")
st.caption("AI RAG Domain-Specific - Hỗ trợ sinh viên FPT | Dữ liệu từ FAP.fpt.edu.vn")

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input chat
if prompt := st.chat_input("Hỏi bất kỳ về FAP (lịch học, quy chế, thủ tục...)"):
    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi Agent (backend)
    with st.chat_message("assistant"):
        with st.spinner("🤖 Đang tra cứu FAP và suy nghĩ..."):
            start_time = time.time()
            response = ask_question(prompt)
            elapsed = round(time.time() - start_time, 1)
        
        st.markdown(response.response)
        st.caption(f"⏱️ {elapsed}s | Agentic RAG")

        # Hiển thị nguồn (rất quan trọng cho hackathon)
        with st.expander("📚 Nguồn tham khảo (trích dẫn)"):
            for node in response.source_nodes:
                file_name = node.metadata.get("file_name", "Unknown")
                score = round(node.score, 3) if node.score else "N/A"
                st.write(f"• **{file_name}** (độ tương đồng: {score})")

    # Lưu vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": response.response})