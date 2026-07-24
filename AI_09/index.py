import os
import sys

# Khắc phục lỗi in tiếng Việt trên console Windows
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Lấy .env từ thư mục gốc AI_training (hoặc có thể tạo .env mới trong AI_09)
load_dotenv(dotenv_path="../.env")
load_dotenv()

# Langchain Google GenAI sử dụng GOOGLE_API_KEY, lấy từ GEMINI_API_KEY nếu có
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

def main():
    print("=== BƯỚC 1: ĐANG LOAD TÀI LIỆU ===")
    # Load tất cả các file .txt trong thư mục data (đã tạo sẵn file company_info.txt)
    loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()
    print(f"Đã load {len(docs)} tài liệu.\n")

    print("=== BƯỚC 2: ĐANG SPLIT (CẮT) TÀI LIỆU ===")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150, 
        chunk_overlap=30
    )
    splits = text_splitter.split_documents(docs)
    print(f"Đã chia thành {len(splits)} chunks.\n")

    print("=== BƯỚC 3 & 4: ĐANG EMBEDDING VÀ LƯU VÀO CHROMA ===")
    # Dùng CÙNG một embedding model của Gemini cho cả index và ask
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Lưu vector vào thư mục ./chroma_db
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Thành công! Đã lưu Database vào thư mục ./chroma_db")

if __name__ == "__main__":
    main()
