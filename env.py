import copy
from typing import Tuple, Dict, Any
from models import PrivacyAction, PrivacyObservation

class DataPrivacyEnv:
    def __init__(self):
        self.max_steps = 15
        self.current_step = 0
        self.active_task = None
        self.db = {}

    def _get_initial_state(self) -> dict:
        """Returns the fresh, hardcoded 'company database'."""
        return {
            "users": [
                {"id": "101", "name": "Alice Admin", "metadata": {"role": "admin"}},
                {"id": "202", "name": "Bob Sales", "metadata": {"role": "sales", "password": "password123"}}, # VULNERABILITY
                {"id": "303", "name": "Charlie Dev", "metadata": {"role": "dev", "password": "qwerty"}},     # VULNERABILITY
                {"id": "404", "name": "Diana HR", "metadata": {"role": "hr", "password": "letmein88"}},      # VULNERABILITY
                {"id": "789", "name": "Eve Ex-Employee", "metadata": {"role": "fired"}}                      # TARGET FOR DELETION
            ],
            "logs": [
                {"id": "log_1", "text": "Server started successfully."},
                {"id": "log_2", "text": "User 101 logged in."},
                {"id": "log_3", "text": "CRITICAL: API_KEY=4532-XXXX-XXXX-XXXX exposed in request."}, # VULNERABILITY
                {"id": "log_4", "text": "User 789 downloaded customer list."},                        # ORPHANED LOG
                {"id": "log_5", "text": "User 789 attempted unauthorized access."}                    # ORPHANED LOG
            ]
        }

    def _get_task_goal(self, task_name: str) -> str:
        goals = {
            "easy_log_redaction": "Find the leaked API key in the '/logs' endpoint and PATCH it to say 'REDACTED'.",
            "medium_crm_audit": "Check the '/users' endpoint. Find the 3 users with exposed passwords in their metadata and PATCH them to remove the password field.",
            "hard_right_to_be_forgotten": "DELETE User ID 789 from the '/users' endpoint. Then, find and DELETE all log entries in '/logs' that mention User 789."
        }
        return goals.get(task_name, "Unknown task.")

    def reset(self, task_name: str = "easy_log_redaction") -> PrivacyObservation:
        """Wipes the environment clean and starts a new episode."""
        self.db = self._get_initial_state()
        self.current_step = 0
        self.active_task = task_name
        
        return PrivacyObservation(
            status_code=200,
            response_data="System Ready. Awaiting API requests.",
            current_task_goal=self._get_task_goal(task_name)
        )

    def step(self, action: PrivacyAction) -> Tuple[PrivacyObservation, float, bool, str]:
        """Processes the agent's action, updates the state, and calculates the reward."""
        self.current_step += 1
        done = False
        error = None
        
        # 1. API ROUTER (Simulate the mock server)
        status_code, response_data = self._route_request(action)
        
        if status_code >= 400:
            error = f"HTTP Error {status_code}: Invalid request."

        # 2. GRADER (Calculate score based on the active task)
        reward, task_completed = self._calculate_reward()
        
        # 3. END CONDITIONS
        if task_completed or self.current_step >= self.max_steps:
            done = True

        obs = PrivacyObservation(
            status_code=status_code,
            response_data=response_data,
            current_task_goal=self._get_task_goal(self.active_task)
        )
        
        return obs, reward, done, error

    def _route_request(self, action: PrivacyAction) -> Tuple[int, Any]:
        """Internal logic to handle GET/POST/PATCH/DELETE."""
        path = action.endpoint.strip("/")
        parts = path.split("/")
        resource = parts[0]
        
        if resource not in ["users", "logs"]:
            return 404, "Endpoint not found."

        # GET /users or GET /logs
        if action.method == "GET":
            if len(parts) == 1:
                return 200, self.db[resource]
            else:
                item_id = parts[1]
                item = next((i for i in self.db[resource] if i["id"] == item_id), None)
                return (200, item) if item else (404, "Item not found.")

        # PATCH /users/{id} or PATCH /logs/{id}
        elif action.method == "PATCH":
            if len(parts) < 2: return 400, "Missing ID in endpoint."
            item_id = parts[1]
            for i, item in enumerate(self.db[resource]):
                if item["id"] == item_id:
                    if action.payload:
                        if "remove_fields" in action.payload and resource == "users":
                            for field in action.payload["remove_fields"]:
                                item["metadata"].pop(field, None)
                        if "text" in action.payload and resource == "logs":
                            item["text"] = action.payload["text"]
                    return 200, "Update successful."
            return 404, "Item not found."

        # DELETE /users/{id} or DELETE /logs/{id}
        elif action.method == "DELETE":
            if len(parts) < 2: return 400, "Missing ID in endpoint."
            item_id = parts[1]
            original_len = len(self.db[resource])
            self.db[resource] = [i for i in self.db[resource] if i["id"] != item_id]
            if len(self.db[resource]) < original_len:
                return 200, "Deletion successful."
            return 404, "Item not found."

        return 405, "Method not allowed."

    def _calculate_reward(self) -> Tuple[float, bool]:
        """The deterministic grader. Returns (reward_for_this_step, is_done)."""
        score = 0.0
        done = False

        if self.active_task == "easy_log_redaction":
            # Check if the API key string is gone
            log_3 = next((l for l in self.db["logs"] if l["id"] == "log_3"), None)
            if log_3 and "4532" not in log_3["text"] and "REDACTED" in log_3["text"]:
                score = 1.0
                done = True

        elif self.active_task == "medium_crm_audit":
            # Check how many vulnerable users are fixed
            fixed_count = 0
            for uid in ["202", "303", "404"]:
                user = next((u for u in self.db["users"] if u["id"] == uid), None)
                if user and "password" not in user["metadata"]:
                    fixed_count += 1
            
            score = fixed_count * 0.333
            if fixed_count == 3:
                score = 1.0
                done = True

        elif self.active_task == "hard_right_to_be_forgotten":
            # 0.4 points for deleting the user
            user_deleted = not any(u["id"] == "789" for u in self.db["users"])
            if user_deleted: score += 0.4
            
            # 0.3 points each for deleting the two orphaned logs
            log_4_deleted = not any(l["id"] == "log_4" for l in self.db["logs"])
            log_5_deleted = not any(l["id"] == "log_5" for l in self.db["logs"])
            
            if log_4_deleted: score += 0.3
            if log_5_deleted: score += 0.3
            
            if user_deleted and log_4_deleted and log_5_deleted:
                score = 1.0
                done = True

        # Ensure floating point math doesn't result in something like 0.999000001
        score = round(score, 3)

        # --- THE FIX: CLAMP SCORE BETWEEN 0.01 AND 0.99 ---
        # The validator strictly rejects 0.0 and 1.0
        final_reward = max(0.01, min(0.99, score))

        return final_reward, done