from fastapi import FastAPI
from typing import Dict, Any

from env import DataPrivacyEnv
from models import PrivacyAction

app = FastAPI(title="DataPrivacyEnv API")
env = DataPrivacyEnv()
from fastapi import FastAPI

app = FastAPI()

# Add this block to handle the root URL and satisfy health checks
@app.get("/")
async def root():
    return {"status": "healthy", "message": "The API is running!"}
@app.post("/reset")
async def reset_env(payload: Dict[str, Any] = {}):
    """
    The pre-validation script pings this endpoint with an empty JSON object {}.
    We default to the easy task if none is provided.
    """
    task_name = payload.get("task_name", "easy_log_redaction")
    obs = env.reset(task_name=task_name)
    
    return {
        "observation": obs.model_dump(),
        "done": False
    }

@app.post("/step")
async def step_env(action: PrivacyAction):
    """
    Executes the agent's action and returns the exact tuple required by OpenEnv.
    """
    obs, reward, done, error = env.step(action)
    
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "error": error
    }

@app.get("/state")
async def get_state():
    """
    Allows the judges to inspect the internal state of the 'database' at any time.
    """
    return {
        "current_step": env.current_step,
        "active_task": env.active_task,
        "db": env.db
    }