import asyncio
import functools
import time
import sys
from contextlib import contextmanager

# Fix Unicode printing issue on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import Callable, Any, Coroutine, TypeVar

# 1. Schema Summary: Pydantic BaseModel với text: str và word_count: int.
class Summary(BaseModel):
    """
    Định nghĩa cấu trúc dữ liệu trả về với Pydantic.
    Pydantic sẽ tự động validate kiểu dữ liệu (text phải là str, word_count phải là int).
    """
    text: str
    word_count: int

# Định nghĩa TypeVar cho các hàm decorator để giữ nguyên type hint
F = TypeVar('F', bound=Callable[..., Coroutine[Any, Any, Any]])

# 3. @retry(times=3): Decorator tự thử lại khi gặp lỗi tạm thời.
def retry(times: int) -> Callable[[F], F]:
    """
    Decorator có tham số nhận vào số lần thử lại (times).
    Cần 3 tầng hàm lồng nhau:
    - Tầng 1: Nhận tham số của decorator (times)
    - Tầng 2: Nhận hàm cần bọc (func)
    - Tầng 3: Hàm wrapper thực sự thực thi logic
    """
    def decorator(func: F) -> F:
        @functools.wraps(func) # Giữ lại các metadata của hàm ban đầu (tên, docstring...)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(times):
                try:
                    # Vì hàm bị bọc (summarize) là async nên bên trong phải dùng await
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Lỗi ở lần thử {attempt + 1}, thử lại...")
                    await asyncio.sleep(0.5) # Chờ một chút trước khi thử lại
            
            # Nếu hết số lần thử mà vẫn lỗi thì raise lỗi cuối cùng
            print("Đã hết số lần thử lại!")
            raise last_exception if last_exception else Exception("Unknown error")
        return wrapper # type: ignore
    return decorator

# 4. Cache in-memory: Input trùng thì trả kết quả cũ, không tính lại.
# Khởi tạo dict ở mức module để lưu trữ cache.
# Type hint chỉ rõ key là str và value là Summary.
_cache: dict[str, Summary] = {}

# 2. Hàm summarize: async, giả lập LLM bằng asyncio.sleep, trả về Summary.
@retry(times=3)
async def summarize(text: str) -> Summary:
    """
    Giả lập một hàm gọi API LLM tốn thời gian.
    Sử dụng asyncio.sleep để không block luồng thực thi chung.
    """
    # Giả lập thời gian xử lý của LLM là 1 giây
    await asyncio.sleep(1)
    
    # Validate và trả về object Summary. Nếu đưa sai kiểu, Pydantic sẽ báo lỗi.
    return Summary(text=text, word_count=len(text.split()))

# 5. Context manager timer: Đo và in tổng thời gian chạy.
@contextmanager
def timer():
    """
    Sử dụng contextmanager để tạo khối with: tính tổng thời gian chạy.
    Hàm perf_counter() được gọi trước và sau khi yield.
    """
    start_time = time.perf_counter()
    yield # Giao lại quyền điều khiển cho khối code bên trong khối 'with'
    end_time = time.perf_counter()
    print(f"Tổng thời gian chạy: {end_time - start_time:.2f} giây")

async def bound_summarize(text: str, sem: asyncio.Semaphore) -> Summary:
    """
    Hàm trung gian xử lý từng văn bản, bao gồm check cache và giới hạn đồng thời.
    """
    # 4. Cache in-memory: Kiểm tra 'if text in _cache' trước khi gọi API (LLM)
    if text in _cache:
        print(f"Cache hit: '{text[:15]}...' -> Bỏ qua xử lý")
        return _cache[text]
    
    # 6. Giới hạn đồng thời: bọc mỗi call bằng 'async with sem:'
    async with sem:
        print(f"Đang xử lý: '{text[:15]}...'")
        result = await summarize(text)
        
        # Lưu vào cache sau khi xử lý xong để các lời gọi sau dùng lại
        _cache[text] = result
        return result

# 7. Type hint đầy đủ: Hàm ghi rõ kiểu vào/ra (list[str], -> list[Summary])
async def process_batch(texts: list[str]) -> list[Summary]:
    """
    Hàm chạy song song nhiều văn bản với giới hạn số lượng gọi cùng lúc.
    """
    # 6. Giới hạn đồng thời: khởi tạo Semaphore tối đa 3 call đồng thời
    sem = asyncio.Semaphore(3)
    
    # 6. Chạy song song: Dùng asyncio.gather để gom tất cả các task lại và chạy cùng lúc.
    # Sử dụng generator expression để tạo các coroutine bound_summarize
    results = await asyncio.gather(*(bound_summarize(t, sem) for t in texts))
    
    # Trả về list chứa các object Summary đã được xử lý (và validate)
    return list(results)

async def main() -> None:
    # Tạo danh sách ~10 văn bản, có một số văn bản trùng lặp để kiểm tra cache
    texts: list[str] = [
        "Đây là văn bản thứ nhất cần tóm tắt.",
        "Hôm nay trời rất đẹp để học lập trình.",
        "Học AI Engineer với Python rất thực dụng.",
        "Đây là văn bản thứ nhất cần tóm tắt.", # Trùng lặp -> sẽ dùng cache
        "Python idiomatic giúp code ngắn gọn và dễ đọc hơn.",
        "Hôm nay trời rất đẹp để học lập trình.", # Trùng lặp -> sẽ dùng cache
        "Xử lý bất đồng bộ async giúp tăng tốc độ đáng kể.",
        "Pydantic tự động validate dữ liệu đầu vào.",
        "Sử dụng decorator để tái sử dụng logic.",
        "Context manager với yield dùng để quản lý tài nguyên."
    ]
    
    print("Bắt đầu xử lý danh sách văn bản...\n")
    
    # 5. Dùng timer context manager để đo tổng thời gian chạy
    with timer():
        summaries = await process_batch(texts)
    
    print(f"\nĐã xử lý xong {len(summaries)} văn bản.")
    print("Kết quả mẫu (phần tử đầu tiên):")
    print(summaries[0])

if __name__ == "__main__":
    asyncio.run(main())
