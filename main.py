import uvicorn
from fastapi import FastAPI
from openenv_core.server import create_app
from env import DataPrivacyEnv

# 1. Initialize your custom environment logic
# This pulls in the tasks we defined (Log Redaction, CRM Audit, etc.)
env = DataPrivacyEnv()

# 2. Create the standard OpenEnv FastAPI application
# This automatically sets up the /reset and /step endpoints
app = create_app(env)

# 3. Add a root health check
# This ensures Hugging Face Spaces marks the app as "Running" (Status 200)
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "DataPrivacyEnv is live!",
        "version": "1.0.0"
    }

# 4. The 'main' function required by the pyproject.toml entry point
# The validator calls this to spin up your environment programmatically
def main():
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
