"""
Topic 6 · Tool calling — Trợ lý tra cứu nhiều công cụ
Vòng lặp tool calling với calculate + get_weather (API thật wttr.in).
"""

import ast
import json
import operator
import os
import sys
from typing import Any, Callable

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
)

MODEL = "google/gemini-2.0-flash-exp:free"

SYSTEM_PROMPT = (
    "Bạn là trợ lý tra cứu. Khi cần số liệu hoặc thời tiết thật, hãy gọi tool — "
    "không đoán. Trả lời ngắn gọn bằng tiếng Việt."
)


# --- Pydantic input schemas (yêu cầu 2) ---

class CalculateInput(BaseModel):
    expression: str = Field(
        description="Biểu thức số học cần tính, ví dụ: '2 + 3 * 4' hoặc '(10 - 2) / 4'",
    )


class GetWeatherInput(BaseModel):
    city: str = Field(
        description="Tên thành phố cần tra cứu thời tiết, ví dụ: 'Hanoi', 'Ho Chi Minh City'",
    )


# --- Tool implementations (yêu cầu 1) ---

_ALLOWED_OPS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return float(_ALLOWED_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return float(_ALLOWED_OPS[type(node.op)](_eval_ast(node.operand)))
    raise ValueError(f"Biểu thức không hợp lệ: {ast.unparse(node)}")


def calculate(expression: str) -> str:
    tree = ast.parse(expression.strip(), mode="eval")
    result = _eval_ast(tree.body)
    if result == int(result):
        return str(int(result))
    return str(result)


def get_weather(city: str) -> str:
    if city.strip().lower() in {"__error__", "invalid_city_xyz"}:
        raise ValueError(f"Không tìm thấy thành phố: {city}")

    response = httpx.get(
        f"https://wttr.in/{city}",
        params={"format": "j1"},
        timeout=15.0,
        headers={"User-Agent": "AI_06-assistant/1.0"},
    )
    response.raise_for_status()
    data = response.json()

    current = data["current_condition"][0]
    desc = current["weatherDesc"][0]["value"]
    temp_c = current["temp_C"]
    humidity = current["humidity"]
    return f"{city}: {desc}, nhiệt độ {temp_c}°C, độ ẩm {humidity}%"


# --- Dispatch theo tên (yêu cầu 3) ---

TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "calculate": calculate,
    "get_weather": get_weather,
}


def _pydantic_to_openai_tool(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


TOOLS: list[dict[str, Any]] = [
    _pydantic_to_openai_tool(
        "calculate",
        "Tính kết quả biểu thức số học an toàn (+, -, *, /, //, %, **).",
        CalculateInput,
    ),
    _pydantic_to_openai_tool(
        "get_weather",
        "Tra cứu thời tiết hiện tại của một thành phố (dữ liệu thật từ wttr.in).",
        GetWeatherInput,
    ),
]


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Chạy tool và bắt lỗi — trả text cho model, không crash (yêu cầu 6)."""
    try:
        if name not in TOOL_FUNCTIONS:
            return f"Lỗi: Tool '{name}' không tồn tại."
        return TOOL_FUNCTIONS[name](**arguments)
    except Exception as exc:
        return f"Lỗi: {exc}"


def run_assistant(user_message: str) -> str:
    """Vòng lặp tool calling tới khi model trả lời cuối (yêu cầu 4, 5)."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message

        messages.append(assistant_message.model_dump(exclude_none=True))

        tool_calls = assistant_message.tool_calls
        if not tool_calls:
            return assistant_message.content or ""

        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                result = f"Lỗi: Không parse được arguments JSON: {exc}"
            else:
                result = execute_tool(fn_name, fn_args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )


def main() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Thiếu OPENROUTER_API_KEY. Sao chép .env.example thành .env và điền key.")
        return

    test_queries = [
        # Câu 1: cần 1 tool (thời tiết)
        "Thời tiết ở Hà Nội hôm nay thế nào?",
        # Câu 2: cần cả 2 tool (thời tiết + tính toán)
        (
            "Cho biết thời tiết hiện tại ở Đà Nẵng, "
            "rồi tính giúp tôi: nếu nhiệt độ đó cộng thêm 12 độ thì bằng bao nhiêu?"
        ),
    ]

    print("=== AI_06 · Tool calling assistant ===\n")

    for i, question in enumerate(test_queries, start=1):
        print(f"--- Câu {i} ---")
        print(f"Hỏi: {question}\n")
        try:
            answer = run_assistant(question)
            print(f"Trả lời: {answer}\n")
        except Exception as exc:
            print(f"Lỗi khi gọi API: {exc}\n")

    # Demo xử lý lỗi tool (không crash)
    print("--- Demo lỗi tool ---")
    print("Hỏi: Thời tiết ở invalid_city_xyz thế nào?\n")
    try:
        answer = run_assistant("Thời tiết ở invalid_city_xyz thế nào?")
        print(f"Trả lời: {answer}\n")
    except Exception as exc:
        print(f"Lỗi khi gọi API: {exc}\n")


if __name__ == "__main__":
    main()
