import uvicorn
from fastapi import FastAPI, Request
import sys
import os

# 1. Add the root directory to the path so it can find env.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import DataPrivacyEnv

app = FastAPI()

# 2. Initialize your environment
my_env = DataPrivacyEnv()

# 3. Health check for the Hugging Face / Scaler Ping
@app.get("/")
def health_check():
    return {"status": "Environment is Running locally and ready for validation."}

# 4. Manual Reset Endpoint
@app.post("/reset")
async def reset_env(request: Request):
    body = await request.json() if await request.body() else {}
    task_name = body.get("task_name", "easy_log_redaction")
    obs = my_env.reset(task_name=task_name)
    
    if hasattr(obs, "model_dump"): return obs.model_dump()
    elif hasattr(obs, "dict"): return obs.dict()
    return obs

# 5. Manual Step Endpoint
@app.post("/step")
async def step_env(request: Request):
    action = await request.json()
    obs, reward, done, info = my_env.step(action)
    
    if hasattr(obs, "model_dump"): obs_dict = obs.model_dump()
    elif hasattr(obs, "dict"): obs_dict = obs.dict()
    else: obs_dict = obs
    
    return {"observation": obs_dict, "reward": float(reward), "done": bool(done), "info": info}

# 6. Validator Entry Point
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()
