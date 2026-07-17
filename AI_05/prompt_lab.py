import os
import json
import tiktoken
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Khởi tạo client OpenRouter
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY", "dummy"),
)

# Chọn model free từ OpenRouter theo hướng dẫn
MODEL = "google/gemini-2.0-flash-lite-preview-02-05:free"

# 1. Pydantic schema
class SentimentResult(BaseModel):
    sentiment: str = Field(description="The sentiment of the review, strictly one of: 'positive', 'negative', 'neutral'")

# 2. Dataset (>= 15 ca)
test_set = [
    ("Đồ ăn rất ngon, phục vụ chu đáo.", "positive"),
    ("Quán bẩn, nhân viên thái độ lồi lõm.", "negative"),
    ("Mình thấy ăn cũng bình thường, không có gì đặc sắc.", "neutral"),
    ("Món gà hơi mặn nhưng bù lại súp rất vừa miệng.", "neutral"),
    ("Chắc chắn sẽ quay lại ủng hộ quán nhiều lần nữa!", "positive"),
    ("Giá quá đắt so với chất lượng.", "negative"),
    ("Tuyệt vời ông mặt trời!", "positive"),
    ("Đợi món hơn 1 tiếng đồng hồ, quá thất vọng.", "negative"),
    ("Không gian rộng rãi, thoáng mát, phù hợp đi gia đình.", "positive"),
    ("Mùi vị thì ổn nhưng bày trí xấu quá.", "neutral"),
    ("Mình không thích đồ ăn ở đây lắm, hơi nhiều dầu mỡ.", "negative"),
    ("Sẽ recommend cho bạn bè.", "positive"),
    ("Ăn xong về bị đau bụng nguyên đêm. Sợ hãi!", "negative"),
    ("Cũng tạm được, tiện đường thì ghé chứ không cố tình đến.", "neutral"),
    ("Chưa bao giờ ăn một bát phở dở như vậy.", "negative"),
    # Tricky cases
    ("Tôi đã từng nghĩ quán này ngon, cho đến khi tôi gọi món bít tết hôm nay.", "negative"),
    ("Ngon thì có ngon đấy, nhưng cái giá thì... ôi chao!", "negative")
]

# 3. Prompt templates
prompt_v1 = """You are an AI assistant.
Classify the following restaurant review into one of three categories: positive, negative, neutral.
Return ONLY a valid JSON object matching this schema:
{{
  "sentiment": "positive | negative | neutral"
}}

Review:
<input>
{review}
</input>
"""

prompt_v2 = """You are an expert linguist and food critic. Your task is to accurately classify the sentiment of restaurant reviews.
Constraints:
- You must return ONLY a valid JSON object.
- The output must match this schema exactly:
  {{ "sentiment": "string (strictly 'positive', 'negative', or 'neutral')" }}
- Rule 1: If the review has mixed feelings but leans heavily towards one side, choose that side.
- Rule 2: If the review is truly mixed or expresses average satisfaction, choose "neutral".
- Rule 3: Pay attention to sarcasm and expectations vs reality (e.g., "was good until...").

Review:
<input>
{review}
</input>
"""

def count_tokens(text: str) -> int:
    try:
        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback if tiktoken fails or isn't downloaded correctly
        return len(text.split())

def predict_sentiment(prompt_template: str, review: str) -> str:
    prompt = prompt_template.format(review=review)
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content
        
        # Parse & validate with Pydantic
        # Extract json if it's wrapped in markdown code blocks
        if content.startswith("```json"):
            content = content.replace("```json\n", "").replace("```", "")
        elif content.startswith("```"):
            content = content.replace("```\n", "").replace("```", "")
            
        data = json.loads(content)
        result = SentimentResult(**data)
        return result.sentiment.lower()
    except (json.JSONDecodeError, ValidationError) as e:
        return f"format_error: {str(e)}"
    except Exception as e:
        return f"api_error: {str(e)}"

def evaluate(prompt_template: str, version_name: str):
    print(f"\n--- Evaluating {version_name} ---")
    correct = 0
    total = len(test_set)
    total_tokens = 0
    
    for i, (review, expected) in enumerate(test_set):
        filled_prompt = prompt_template.format(review=review)
        tokens = count_tokens(filled_prompt)
        total_tokens += tokens
        
        prediction = predict_sentiment(prompt_template, review)
        
        is_correct = (prediction == expected)
        if is_correct:
            correct += 1
            
        # Uncomment below to debug individual cases
        # print(f"[{i+1}] {review[:30]}... | Expected: {expected} | Got: {prediction} | {'✅' if is_correct else '❌'}")
        
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Average tokens per prompt: {total_tokens / total:.1f}")
    
    return accuracy

if __name__ == "__main__":
    print("Bắt đầu chấm điểm các phiên bản prompt...")
    
    # 1. Đếm token thử 1 prompt
    sample_prompt = prompt_v1.format(review=test_set[0][0])
    print(f"Số token của 1 prompt mẫu (v1): {count_tokens(sample_prompt)}")
    
    # 2. Đánh giá v1
    acc_v1 = evaluate(prompt_v1, "Version 1 (Basic)")
    
    # 3. Đánh giá v2
    acc_v2 = evaluate(prompt_v2, "Version 2 (Improved Constraints & Role)")
    
    # 4. So sánh
    print("\n--- KẾT LUẬN ---")
    if acc_v2 > acc_v1:
        print("Version 2 tốt hơn Version 1. Giữ lại Version 2.")
    elif acc_v2 == acc_v1:
        print("Cả 2 version đều có độ chính xác như nhau.")
    else:
        print("Version 1 tốt hơn Version 2. Cần xem lại cách cải thiện ở Version 2.")
