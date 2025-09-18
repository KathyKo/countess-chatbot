# ingestion.py
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

CHINESE_SEPARATORS: List[str] = ["\n\n", "\n", "。", "？", "！", "；", "：", "…", "、", " "]

def get_product_name(p: Path) -> str:
    return p.stem

def create_vector_store_with_metadata(
    data_dir: str,
    vector_store_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = "nomic-embed-text",
    index_name: str = "index",
) -> None:
    print("--- 開始資料攝入（Data Ingestion）流程 ---")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"找不到資料夾：{data_path}")

    out_dir = Path(vector_store_path)
    out_dir.mkdir(parents=True, exist_ok=True)  # 自動建立，不靠桌面

    docx_files = list(data_path.rglob("*.docx"))
    if not docx_files:
        raise FileNotFoundError(f"在 {data_path} 找不到任何 .docx 檔案")

    documents = []
    for fp in docx_files:
        print(f"處理檔案: {fp}")
        loader = Docx2txtLoader(str(fp))
        for d in loader.load():
            d.metadata["product"] = get_product_name(fp)
            d.metadata["source"] = str(fp)
            documents.append(d)

    print(f"已載入 {len(documents)} 個文件")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=CHINESE_SEPARATORS
    )
    chunks = splitter.split_documents(documents)
    print(f"文件已拆分為 {len(chunks)} 個區塊")

    base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    print(f"正在載入嵌入模型: {embedding_model} (Ollama: {base_url})")
    embeds = OllamaEmbeddings(model=embedding_model, base_url=base_url)

    print("正在建立 FAISS 向量庫...")
    vs = FAISS.from_documents(chunks, embeds, normalize_L2=True)

    vs.save_local(str(out_dir), index_name=index_name)
    print(f"向量庫已儲存至 {out_dir}")
    print("--- 資料攝入流程結束 ---")

if __name__ == "__main__":
    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    INDEX_NAME = os.getenv("INDEX_NAME", "index")

    if not DATA_DIR or not VECTOR_STORE_PATH:
        raise RuntimeError("請在 .env 設定 DATA_DIR 與 VECTOR_STORE_PATH")

    create_vector_store_with_metadata(
        data_dir=DATA_DIR,
        vector_store_path=VECTOR_STORE_PATH,
        chunk_size=500,
        chunk_overlap=50,
        embedding_model=EMBEDDING_MODEL,
        index_name=INDEX_NAME,
    )
