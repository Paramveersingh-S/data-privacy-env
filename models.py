from pydantic import BaseModel
from typing import Optional, Any, List

class PrivacyAction(BaseModel):
    method: str
    endpoint: str
    payload: Optional[dict] = None

class PrivacyObservation(BaseModel):
    status_code: int
    response_data: Any
    current_task_goal: str
