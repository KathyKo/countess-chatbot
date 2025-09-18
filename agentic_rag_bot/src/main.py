import os
from pathlib import Path
from typing import TypedDict, List, Literal, Optional, Union
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationSummaryMemory

# ======================================
# 環境變數
# ======================================
load_dotenv()
PRODUCTS = ["天絲", "天絲棉", "莫代爾棉", "絲光棉", "品牌故事", "其他"]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
INDEX_NAME = os.getenv("INDEX_NAME", "index")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

if not VECTOR_STORE_PATH:
    raise RuntimeError("請在 .env 設定 VECTOR_STORE_PATH")

vec_dir = Path(VECTOR_STORE_PATH)
faiss_file = vec_dir / f"{INDEX_NAME}.faiss"
pkl_file = vec_dir / f"{INDEX_NAME}.pkl"

if not faiss_file.exists() or not pkl_file.exists():
    raise FileNotFoundError("找不到向量庫檔案，請先執行 ingestion.py 產生索引")

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
vector_store = FAISS.load_local(
    str(vec_dir),
    embeddings,
    allow_dangerous_deserialization=True,
    index_name=INDEX_NAME,
)
llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_HOST)

# ======================================
# 記憶體設定：ConversationSummaryMemory
# ======================================
summary_memory = ConversationSummaryMemory(
    llm=llm,
    ai_prefix="AI",
    human_prefix="Human",
    return_messages=True,
)

# ======================================
# 工具函式
# ======================================
def clean_response(text: str) -> str:
    """移除 <think> ... </think> 區塊，保持輸出乾淨。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def format_docs(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        prod = d.metadata.get("product", "未知產品")
        lines.append(f"[{prod}] {d.page_content}")
    return "\n\n".join(lines)

# ======================================
# 狀態定義
# ======================================
class GraphState(TypedDict):
    chat_history: List[Union[HumanMessage, AIMessage]]
    question: str
    route: str
    product_to_search: Optional[str]
    products_to_search: List[str]
    documents: List[Document]
    response: Optional[str]
    verdict: Literal["ok", "retry", "refuse"]
    retries: int

# ======================================
# Router（規則優先 + LLM 後援）
# ======================================
router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是路由助理。請將用戶問題路由到最相關的產品類別，或判定為 'out_of_scope'。\n"
     "選項：{products}\n"
     "規則：\n"
     "1) 僅能輸出一個字串，不能解釋。\n"
     "2) 多個產品用逗號分隔，例如 '天絲,莫代爾棉'。\n"
     "3) 無關則輸出 'out_of_scope'。\n"
     "4) 若同時出現『天絲棉』與『天絲』，以『天絲棉』為準。"),
    ("user", "{question}")
])
router_chain = router_prompt | llm | StrOutputParser()

def route_question(state: GraphState) -> GraphState:
    q = state["question"]

    # ===== 規則優先 =====
    if "天絲棉" in q:
        state["route"] = "single_product"
        state["product_to_search"] = "天絲棉"
        print("[Router] 規則命中：天絲棉")
        return state
    if ("天絲" in q) and ("天絲棉" not in q):
        state["route"] = "single_product"
        state["product_to_search"] = "天絲"
        print("[Router] 規則命中：天絲")
        return state
    if ("莫代爾棉" in q) or ("cm" in q.lower()):
        state["route"] = "single_product"
        state["product_to_search"] = "莫代爾棉"
        print("[Router] 規則命中：CM/莫代爾棉")
        return state
    if any(keyword in q for keyword in ["Countess", "countess", "品牌", "品牌故事", "成立", "創立", "歷史", "門市"]):
        state["route"] = "single_product"
        state["product_to_search"] = "品牌故事"
        print("[Router] 規則命中：品牌故事")
        return state

    # ===== LLM 後援 =====
    raw = router_chain.invoke({"question": q, "products": ", ".join(PRODUCTS)}).strip()
    cleaned = re.sub(r"<[^>]+>", "", raw).strip()
    route = cleaned.split("\n")[-1].strip()
    print(f"[Router] 決策: {route}")

    state["products_to_search"] = []
    state["product_to_search"] = None
    state["documents"] = []
    state["response"] = None
    state["verdict"] = "refuse"

    if "," in route:
        state["route"] = "multi_product"
        state["products_to_search"] = [p.strip() for p in route.split(",") if p.strip()]
    elif route.lower() == "out_of_scope":
        state["route"] = "out_of_scope"
    elif route in PRODUCTS:
        state["route"] = "single_product"
        state["product_to_search"] = route
    else:
        state["route"] = "out_of_scope"
    return state

# ======================================
# 單產品 / 多產品 RAG
# ======================================
single_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "你是一個寢具專家。請根據提供的內容簡潔回答用戶問題。\n"
     "規則：\n"
     "1) 不要輸出 <think> 或思考過程。\n"
     "2) 直接條列或短段落作答，保持精簡。\n"
     "3) 若內容不足，請說「抱歉，文件中沒有相關資訊」。"),
    ("user", "上下文: {context}\n\n問題: {question}")
])

multi_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "你是一個多產品專家。請根據多個產品的內容，簡潔比較並回答問題。\n"
     "規則：\n"
     "1) 不要輸出 <think> 或思考過程。\n"
     "2) 直接回答，保持簡短。\n"
     "3) 若內容不足，請說「抱歉，文件中沒有相關資訊」。"),
    ("user", "上下文: {context}\n\n問題: {question}")
])

def single_product_rag(state: GraphState) -> GraphState:
    docs_with_scores = vector_store.similarity_search_with_score(
        state["question"], k=6, filter={"product": state["product_to_search"]}
    )
    docs = [d for d, s in docs_with_scores if s <= 1.0]
    if not docs:
        return {**state, "documents": [], "response": None}

    response = (single_rag_prompt | llm | StrOutputParser()).invoke(
        {"context": format_docs(docs), "question": state["question"]}
    )
    response = clean_response(response)
    return {**state, "documents": docs, "response": response}

def multi_product_rag(state: GraphState) -> GraphState:
    all_docs: List[Document] = []
    for product in state.get("products_to_search", []):
        docs_with_scores = vector_store.similarity_search_with_score(
            state["question"], k=4, filter={"product": product}
        )
        all_docs.extend([d for d, s in docs_with_scores if s <= 1.0])
    if not all_docs:
        return {**state, "documents": [], "response": None}

    response = (multi_rag_prompt | llm | StrOutputParser()).invoke(
        {"context": format_docs(all_docs), "question": state["question"]}
    )
    response = clean_response(response)
    return {**state, "documents": all_docs, "response": response}

# ======================================
# Verifier
# ======================================
verifier_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是答案審核員。檢查回答是否嚴格依據上下文。\n"
     "僅輸出以下之一：ok / retry / refuse"),
    ("user", "上下文：\n{context}\n\n回答：\n{answer}")
])
verifier_chain = verifier_prompt | llm | StrOutputParser()

def verifier_node(state: GraphState) -> GraphState:
    if not state.get("documents"):
        return {**state, "verdict": "refuse"}
    verdict_raw = verifier_chain.invoke({
        "context": format_docs(state["documents"]),
        "answer": state.get("response") or ""
    }).strip().lower()
    verdict_raw = clean_response(verdict_raw)
    verdict = "ok" if "ok" in verdict_raw else "retry" if "retry" in verdict_raw else "refuse"
    print(f"[Verifier] 判定：{verdict_raw} -> {verdict}")
    return {**state, "verdict": verdict}

# ======================================
# 拒答
# ======================================
def refusal_node(state: GraphState) -> GraphState:
    return {**state, "response": "抱歉，我只能回答與 Countess 寢具產品相關、且文件中有明確依據的問題。"}

# ======================================
# 流程
# ======================================
workflow = StateGraph(GraphState)
workflow.add_node("route_question", route_question)
workflow.add_node("single_product_rag", single_product_rag)
workflow.add_node("multi_product_rag", multi_product_rag)
workflow.add_node("verifier_node", verifier_node)
workflow.add_node("refusal_node", refusal_node)

workflow.set_entry_point("route_question")
workflow.add_conditional_edges("route_question", lambda s: s["route"], {
    "single_product": "single_product_rag",
    "multi_product": "multi_product_rag",
    "out_of_scope": "refusal_node",
})
workflow.add_edge("single_product_rag", "verifier_node")
workflow.add_edge("multi_product_rag", "verifier_node")

def after_verify(state: GraphState) -> str:
    if state.get("verdict") == "ok":
        return END
    if state.get("verdict") == "retry" and state.get("retries", 0) < 1:
        state["retries"] = state.get("retries", 0) + 1
        return "single_product_rag" if state["route"] == "single_product" else "multi_product_rag"
    return "refusal_node"

workflow.add_conditional_edges("verifier_node", after_verify, {
    "single_product_rag": "single_product_rag",
    "multi_product_rag": "multi_product_rag",
    "refusal_node": "refusal_node",
    END: END,
})
workflow.add_edge("refusal_node", END)
app = workflow.compile()

# ======================================
# 互動執行
# ======================================
if __name__ == "__main__":
    while True:
        user_question = input("請輸入你的問題：").strip()
        if user_question.lower() in ["end", "退出", "結束"]:
            print("結束對話。")
            break

        init_state: GraphState = {
            "chat_history": [],
            "question": user_question,
            "route": "",
            "product_to_search": None,
            "products_to_search": [],
            "documents": [],
            "response": None,
            "verdict": "refuse",
            "retries": 0,
        }
        result = app.invoke(init_state)
        print("\n=== 最終輸出 ===")
        print(result["response"])
