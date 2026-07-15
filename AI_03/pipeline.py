import asyncio
import httpx
from pydantic import BaseModel, Field, ConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

class Post(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    user_id: int = Field(alias="userId")
    id: int
    title: str
    body: str

# Chỉ retry lỗi tạm thời
RETRY_EXCEPTIONS = (httpx.TransportError, httpx.TimeoutException)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS)
)
async def fetch_post(client: httpx.AsyncClient, post_id: int) -> dict:
    response = await client.get(f"/posts/{post_id}")
    response.raise_for_status() # Biến 4xx/5xx thành lỗi
    return response.json()

async def fetch_and_validate(client: httpx.AsyncClient, sem: asyncio.Semaphore, post_id: int) -> Post:
    async with sem:
        data = await fetch_post(client, post_id)
        return Post.model_validate(data)

async def main():
    # Một AsyncClient dùng chung
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
    timeout = httpx.Timeout(10.0, connect=5.0)
    
    # Giới hạn đồng thời (ví dụ: 10)
    sem = asyncio.Semaphore(10)
    
    async with httpx.AsyncClient(
        base_url="https://jsonplaceholder.typicode.com",
        limits=limits,
        timeout=timeout
    ) as client:
        # jsonplaceholder có 100 posts, ta fetch đến 105 để thử các post không tồn tại (gây lỗi 404)
        tasks = [fetch_and_validate(client, sem, i) for i in range(1, 106)]
        
        # Chịu lỗi từng phần
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        ok_posts = []
        errors = []
        
        for res in results:
            if isinstance(res, Exception):
                errors.append(res)
            else:
                ok_posts.append(res)
                
        # Kiểm kê
        print(f"OK {len(ok_posts)} / lỗi {len(errors)}")
        
        # Ghi jsonl
        with open("out.jsonl", "w", encoding="utf-8") as f:
            for p in ok_posts:
                f.write(p.model_dump_json() + '\n')

if __name__ == "__main__":
    asyncio.run(main())
