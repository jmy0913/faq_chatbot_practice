import streamlit as st
from rag import vector_retriever_create, ai_faq_rag



st.title("📦 주문/배송 FAQ 챗봇")

# 메시지 기록 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 받기
user_input = st.chat_input("궁금한 점을 입력하세요 (예: 반품은 어떻게 하나요?)")

if user_input:
    # 사용자 메시지 출력
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 응답 처리
    with st.chat_message("assistant"):
        with st.spinner("답변을 찾는 중입니다..."):
            retriever = vector_retriever_create()
            response_gen = ai_faq_rag(user_input, retriever)

            full_response = ""
            placeholder = st.empty()

            for chunk in response_gen:
                full_response += chunk
                placeholder.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })


else:
    st.warning('질문을 입력해주세요.')

