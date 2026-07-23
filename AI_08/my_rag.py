import os
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# 1. Tài liệu thật (8-15 chunks)
CHUNKS = [
    "Dịch vụ AI Premium có giá 200,000 VND một tháng.",
    "Khách hàng có thể hủy gói AI Premium bất cứ lúc nào qua phần Cài đặt tài khoản.",
    "Chức năng Voice-to-Text hỗ trợ 50 ngôn ngữ, bao gồm tiếng Việt và tiếng Anh.",
    "Để đổi mật khẩu, người dùng cần nhấp vào 'Quên mật khẩu' ở màn hình đăng nhập.",
    "Gói AI Premium cho phép tạo ra tối đa 1000 hình ảnh mỗi ngày.",
    "Nếu gặp lỗi thanh toán, người dùng nên kiểm tra lại số dư thẻ tín dụng hoặc liên hệ ngân hàng.",
    "Công ty AI Tech được thành lập vào năm 2020 tại Hà Nội.",
    "Dữ liệu của người dùng được mã hóa đầu cuối và không bao giờ được chia sẻ với bên thứ ba.",
    "Ứng dụng AI Tech hỗ trợ trên cả hai nền tảng iOS (từ bản 14.0) và Android (từ bản 10.0).",
    "Người dùng có thể yêu cầu hoàn tiền trong vòng 7 ngày đầu tiên nếu không hài lòng với dịch vụ."
]

# 2. Indexing: Embed toàn bộ chunk một lần
def get_embeddings(texts: list[str]) -> np.ndarray:
    # 4. Cùng model embedding
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=texts,
        task_type="retrieval_document"
    )
    return np.array(result['embedding'])

print("Đang embedding tài liệu...")
# Lấy embeddings lúc module vừa được chạy
CHUNKS_EMBEDDINGS = get_embeddings(CHUNKS)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

# 3. Cosine + top-k
def search_top_k(query: str, k: int = 2) -> list[tuple[int, float]]:
    # Embed query với cùng model
    query_result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = np.array(query_result['embedding'])
    
    scores = []
    for i, doc_emb in enumerate(CHUNKS_EMBEDDINGS):
        score = cosine_similarity(query_embedding, doc_emb)
        scores.append((i, float(score)))
    
    # Sắp xếp giảm dần theo điểm số
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

def answer_question(query: str):
    print(f"\nCâu hỏi: {query}")
    
    # Tìm top-k chunks
    top_k = search_top_k(query, k=2)
    
    # In cả điểm cosine
    context_texts = []
    print("Nguồn tìm thấy:")
    for idx, score in top_k:
        print(f"- [Score: {score:.4f}] Chunk {idx}: {CHUNKS[idx]}")
        context_texts.append(f"[{idx}] {CHUNKS[idx]}")
        
    context = "\n".join(context_texts)
    
    # 5. Prompt grounding (dựa vào ngữ cảnh, delimiter tách dữ liệu)
    # 6. Đường 'không biết'
    prompt = f"""
Bạn là một trợ lý thông minh hỗ trợ trả lời khách hàng.
Hãy trả lời câu hỏi của người dùng CHỈ DỰA VÀO phần "NGỮ CẢNH" bên dưới.
Nếu thông tin trong ngữ cảnh không đủ hoặc không liên quan để trả lời câu hỏi, hãy trả lời chính xác là "Tôi không tìm thấy thông tin này trong tài liệu" và tuyệt đối KHÔNG ĐƯỢC bịa ra câu trả lời.
Khi trả lời, hãy trích dẫn nguồn bằng cách thêm [số thứ tự] của chunk tương ứng vào cuối câu.

=== NGỮ CẢNH ===
{context}
================

Câu hỏi: {query}
"""

    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    # Sử dụng temperature = 0 cho câu trả lời ổn định, bám sát ngữ cảnh
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.0)
    )
    
    # 7. Hiện nguồn (đã in ở trên và model có trích dẫn trong câu trả lời)
    print(f"Trả lời: {response.text}")

def main():
    # 7. Test 3 ca
    print("\n--- CA 1: TRONG KHO ---")
    answer_question("Gói AI Premium giá bao nhiêu?")
    
    print("\n--- CA 2: DIỄN ĐẠT KHÁC CHỮ ---")
    answer_question("Cho mình hỏi là làm sao để lấy lại password vậy?")
    
    print("\n--- CA 3: NGOÀI KHO ---")
    answer_question("Công ty có chi nhánh ở Hồ Chí Minh không?")

if __name__ == "__main__":
    main()
