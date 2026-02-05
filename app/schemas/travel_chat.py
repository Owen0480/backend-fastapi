from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    thread_id: str  # 대화 구분을 위한 고유 ID (예: 사용자 ID 또는 세션 ID)

class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    info_complete: bool
    # context는 그래프 내부 상태에서 자동으로 관리되므로 필요한 경우에만 응답에 포함