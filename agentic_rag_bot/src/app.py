import streamlit as st
from mains import GraphState, workflow  

# 初始化會話狀態
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 處理使用者輸入
if user_input := st.chat_input("請輸入你的問題："):
    # 顯示用戶輸入
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 初始化狀態
    init_state = GraphState(
        chat_history=[{"role": "user", "content": user_input}],
        question=user_input,
        route="",
        product_to_search=None,
        products_to_search=[],
        documents=[],
        response=None,
        verdict="refuse",
        retries=0,
    )

    # 編譯工作流程
    compiled_workflow = workflow.compile()

    # 執行工作流程
    result = compiled_workflow.invoke(init_state)  # 使用 compile 方法來執行工作流程

    # 顯示機器人回應
    with st.chat_message("assistant"):
        st.markdown(result["response"])
    
    # 儲存機器人回應到會話歷史中
    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
