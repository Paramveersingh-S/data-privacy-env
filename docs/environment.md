# 🧩 Environment Specification

`DataPrivacyEnv` follows the Markov Decision Process (MDP) framework where an agent learns to minimize privacy risk through interaction.

## 🟢 State Space ($S$)
The state is represented as an in-memory relational database containing:
- **Users Table**: User metadata (ID, Name, Role, and potentially exposed PII like passwords).
- **Logs Table**: Unstructured text data containing system events and leaked identifiers.

## 👁️ Observation Space ($O$)
The agent does not see the full database state at once. It perceives:
- `status_code`: The result of its last API action.
- `response_data`: The specific JSON fragment returned by the endpoint.
- `current_task_goal`: A natural language embedding of the objective.

## ⚡ Action Space ($A$)
The action space is **discrete-symbolic**. The agent must generate a valid `PrivacyAction` object:
- **Methods**: `GET`, `PATCH`, `DELETE`.
- **Endpoints**: Targeted resource paths.
- **Payload**: Key-value pairs for redaction logic.

## 💰 Reward Function ($R$)
The reward is calculated based on state-delta verification:
$$R_t = \text{Score}_{\text{task}} - \text{Penalty}_{\text{invalid\_action}}$$

- **Success**: $1.0$ (Task completed).
- **Partial**: Scaled based on percentage of PII fixed (e.g., $0.33$ per user record cleaned).
- **Penalty**: $0.0$ for invalid endpoints or malformed JSON.
