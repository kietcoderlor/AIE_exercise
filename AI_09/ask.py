import os
import sys

# Khắc phục lỗi in tiếng Việt trên console Windows
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API key
load_dotenv(dotenv_path="../.env")
load_dotenv()

# Langchain Google GenAI sử dụng GOOGLE_API_KEY, lấy từ GEMINI_API_KEY nếu có
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

def format_docs(docs):
    """
    Hàm ghép các chunks thành một chuỗi văn bản.
    Đồng thời in kèm nguồn ra màn hình để kiểm chứng.
    """
    context = []
    print("\n--- [Debug] Retriever tìm được các nguồn sau ---")
    for i, doc in enumerate(docs):
        # Lấy metadata 'source' (hoặc 'page' nếu là PDF)
        source = doc.metadata.get("source", "Unknown")
        print(f"Chunk {i+1} [Nguồn: {source}]: {doc.page_content}")
        # Đưa cả nội dung và nguồn vào ngữ cảnh cho LLM
        context.append(f"[Nguồn: {source}]\n{doc.page_content}")
    print("------------------------------------------------\n")
    return "\n\n".join(context)

def main():
    print("1. Đang mở kho dữ liệu Chroma (không embed lại)...")
    # Sử dụng đúng CÙNG model embedding lúc index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Mở kho Chroma đã lưu trong thư mục
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Tạo retriever lấy ra top 2 chunks liên quan nhất
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 2. Khởi tạo mô hình LLM với temperature=0 (để ổn định, ít bịa)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # 3. Tạo Prompt Grounding
    template = """
Bạn là một trợ lý thông minh hỗ trợ trả lời khách hàng dựa trên tài liệu.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng CHỈ DỰA VÀO phần "NGỮ CẢNH" được cung cấp bên dưới.
Nếu thông tin trong ngữ cảnh KHÔNG đủ để trả lời, bạn PHẢI nói trung thực là "Tôi không biết" hoặc "Tôi không có thông tin về vấn đề này". Tuyệt đối không được bịa đặt.
Nếu có thể, hãy chỉ ra thông tin được lấy từ nguồn nào.

=== NGỮ CẢNH ===
{context}
================

Câu hỏi của người dùng: {question}
Trả lời:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. Thiết lập chuỗi chain LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    def ask_bot(question: str):
        print(f"❓ Câu hỏi: {question}")
        response = rag_chain.invoke(question)
        print(f"🤖 Bot trả lời: {response}")
        print("="*50)

    # 5. Chạy thử 3 ca kiểm thử
    print("\n\n======== BẮT ĐẦU CHẠY THỬ 3 CA ========\n")
    
    print("CA 1: HỎI CÂU CÓ TRONG TÀI LIỆU")
    ask_bot("Trợ lý ảo thông minh của công ty tên là gì và gói cơ bản có giá bao nhiêu?")
    
    print("\nCA 2: DIỄN ĐẠT KHÁC TỪ TRONG TÀI LIỆU (Tìm kiếm ngữ nghĩa)")
    ask_bot("Bao giờ thì công ty tính vươn ra khu vực DNA?")
    
    print("\nCA 3: HỎI NGOÀI TÀI LIỆU (Kiểm tra xem có trả lời 'không biết' không)")
    ask_bot("Công ty có mở chi nhánh văn phòng ở Đà Nẵng không?")

if __name__ == "__main__":
    main()
