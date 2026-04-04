#  DataPrivacyEnv: Autonomous PII Remediation Agent

[![OpenEnv](https://img.shields.io/badge/Spec-OpenEnv-blue)](https://github.com/openenv)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Scikitlearn10801/DataPrivacyEnv)

DataPrivacyEnv is a real-world simulation environment for training and evaluating AI agents on **Personally Identifiable Information (PII) remediation**. 

##  Motivation & Real-World Utility
PII leaks in logs and internal CRMs cost companies millions in compliance fines. This environment simulates a Data Privacy Officer's workflow, requiring an agent to audit, redact, and delete sensitive data via a mock REST API.

---

##  Task Descriptions
- **Easy Log Redaction**: Find a leaked API Key in text logs and redact it via PATCH.
- **CRM Audit**: Identify 3 users with exposed passwords and remove the fields.
- **Right to be Forgotten**: Delete a specific user and all associated orphaned log entries.

##  Setup & Usage
1. **Build the image**: `docker build -t data-privacy-env .`
2. **Run the container**: `docker run -p 7860:7860 data-privacy-env`
3. **Run the Baseline Agent**: `python inference.py`

---

##  OpenEnv Spec Compliance
```bash
openenv validate
./validate-submission.sh [https://scikitlearn10801-dataprivacyenv.hf.space](https://scikitlearn10801-dataprivacyenv.hf.space) .

---

## 🧠 Agent Reasoning Examples

#### Hard Task: Right to be Forgotten (GDPR Compliance)
To achieve a 1.0 score, the agent must perform a multi-step sequence:
1. **Discovery**: `GET /users` to identify User 789.
2. **Primary Deletion**: `DELETE /users/789`.
3. **Audit**: `GET /logs` to find references to the deleted user.
4. **Cleanup**: `DELETE /logs/log_4` and `DELETE /logs/log_5`.
*This tests the agent's ability to maintain state and follow-through on complex data-integrity requirements.*

---

### 📜 Example Agent Trace (Gemma 4 31B)
```text
[START] task=easy_log_redaction env=DataPrivacyEnv model=google/gemma-4-31B-it
[STEP] step=1 action={"method": "GET", "endpoint": "/logs"}
[STEP] step=2 action={"method": "PATCH", "endpoint": "/logs/log_3", "payload": {"text": "REDACTED"}}
[END] success=true score=1.000
🛠️ Future Roadmap
[ ] Dynamic PII Generation: Use LLMs to generate randomized logs to prevent agent overfitting.

[ ] Multi-Agent Collaboration: One agent audits while another redacts.

[ ] SQL Database Integration: Move from in-memory dictionaries to a real SQLite/Postgres simulation.
