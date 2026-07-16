import argparse
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    model_config = {"env_file": ".env"}

settings = Settings()

def anthropic_call(prompt: str) -> str:
    if not settings.anthropic_api_key:
        return "Bỏ qua: Thiếu ANTHROPIC_API_KEY"
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=settings.anthropic_api_key)
        msg = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        return f"Lỗi: {e}"

def gemini_call(prompt: str) -> str:
    if not settings.gemini_api_key:
        return "Bỏ qua: Thiếu GEMINI_API_KEY"
    try:
        from google import genai
        client = genai.Client(api_key=settings.gemini_api_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return resp.text
    except Exception as e:
        return f"Lỗi: {e}"

def openai_call(prompt: str) -> str:
    if not settings.openai_api_key:
        return "Bỏ qua: Thiếu OPENAI_API_KEY"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Lỗi: {e}"

def openrouter_call(prompt: str) -> str:
    if not settings.openrouter_api_key:
        return "Bỏ qua: Thiếu OPENROUTER_API_KEY"
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key
        )
        resp = client.chat.completions.create(
            model="deepseek/deepseek-chat:free",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Lỗi: {e}"

def ask(provider: str, prompt: str) -> str:
    if provider == "claude":
        return anthropic_call(prompt)
    elif provider == "gemini":
        return gemini_call(prompt)
    elif provider == "openai":
        return openai_call(prompt)
    elif provider == "openrouter":
        return openrouter_call(prompt)
    else:
        return "Unknown provider"

def main():
    parser = argparse.ArgumentParser(description="So sánh các LLM qua một CLI")
    parser.add_argument("prompt", type=str, nargs="?", help="Câu hỏi cho các LLM")
    args = parser.parse_args()

    if args.prompt:
        prompt = args.prompt
    else:
        prompt = input("Nhập câu hỏi: ")

    if not prompt.strip():
        print("Câu hỏi trống.")
        return

    providers = ["claude", "gemini", "openai", "openrouter"]
    
    print(f"\n--- CÂU HỎI ---\n{prompt}\n")
    
    for provider in providers:
        print(f"[{provider.upper()}]")
        answer = ask(provider, prompt)
        print(answer)
        print("-" * 40)

if __name__ == "__main__":
    main()
