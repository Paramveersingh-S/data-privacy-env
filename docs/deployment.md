# 🚀 Deployment Architecture

The environment is containerized and ready for cloud-scale deployment.

## 🏗️ Stack
- **Runtime**: Python 3.10-slim.
- **Framework**: FastAPI + Uvicorn.
- **Containerization**: Docker.
- **Hosting**: Hugging Face Spaces (Docker SDK).

## 🐳 Docker Configuration
The environment is exposed on port `7860`. The build uses a multi-stage-like approach to keep the image size under 500MB, ensuring fast cold-starts on serverless platforms.

## 🛡️ Security
The environment uses a **read-only** mock database that resets every episode. No actual PII is ever stored; all data is programmatically generated in-memory, making it safe for open-source AI research.
