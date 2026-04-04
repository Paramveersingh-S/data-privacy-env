import uvicorn
from fastapi import FastAPI
from openenv_core.server import create_app
from env import DataPrivacyEnv

# 1. Initialize your custom environment
env = DataPrivacyEnv()

# 2. Create the standard OpenEnv FastAPI app
app = create_app(env)

# 3. Add the root health check for Hugging Face Spaces
@app.get("/")
async def root():
    return {"status": "healthy", "message": "DataPrivacyEnv is live!"}

# 4. The 'main' function the validator is looking for
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
