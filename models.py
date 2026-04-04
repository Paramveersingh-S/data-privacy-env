from typing import Optional, Any, Dict, Union
from pydantic import BaseModel, Field

class PrivacyAction(BaseModel):
    """
    Defines the exact structure of an action the AI agent can take.
    The agent acts like it is making HTTP requests to an internal API.
    """
    method: str = Field(
        description="The HTTP method to use. Must be one of: GET, POST, PATCH, or DELETE."
    )
    endpoint: str = Field(
        description="The API endpoint to interact with, e.g., '/users', '/users/123', or '/logs'."
    )
    payload: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional JSON payload for POST or PATCH requests. Used to update or redact data."
    )

class PrivacyObservation(BaseModel):
    """
    Defines the exact structure of what the AI agent 'sees' after taking an action.
    """
    status_code: int = Field(
        description="The HTTP status code of the response (e.g., 200 for success, 404 for not found)."
    )
    response_data: Union[Dict[str, Any], list, str, None] = Field(
        description="The data returned by the server. Could be JSON objects, a list of users, or plain text logs."
    )
    current_task_goal: str = Field(
        description="The active instruction telling the agent what it needs to achieve."
    )