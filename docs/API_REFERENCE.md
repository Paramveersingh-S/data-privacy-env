# 📡 Mock API Reference

The `DataPrivacyEnv` simulates a RESTful API. Below are the available endpoints for the agent.

| Endpoint | Method | Payload | Description |
| :--- | :--- | :--- | :--- |
| `/users` | `GET` | N/A | Returns list of all user profiles. |
| `/users/{id}` | `PATCH` | `{"remove_fields": ["field"]}` | Removes specific sensitive metadata. |
| `/users/{id}` | `DELETE` | N/A | Completely removes a user record. |
| `/logs` | `GET` | N/A | Returns system logs containing potential PII. |
| `/logs/{id}` | `PATCH` | `{"text": "REDACTED"}` | Overwrites log text with a placeholder. |
| `/logs/{id}` | `DELETE` | N/A | Removes an orphaned log entry. |
