import os
import sys
import json
import traceback

def log_and_flush(message, stream=sys.stdout):
    stream.write(message + "\n")
    stream.flush()

# -------------------------------------------------------------------
# 1. Try to import real openenv, fall back to mock if missing
# -------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError as e:
    log_and_flush(f"[CRITICAL ERROR] Failed to import OpenAI:\n{traceback.format_exc()}", sys.stderr)
    sys.exit(1)

MOCK_MODE = False
try:
    from openenv.core import make
except ImportError:
    try:
        from openenv import make
    except ImportError:
        log_and_flush("[WARNING] openenv not found. Entering MOCK MODE (simulated environment).", sys.stderr)
        MOCK_MODE = True

        # -------------------------------------------------------------------
        # MOCK ENVIRONMENT for testing without openenv
        # -------------------------------------------------------------------
        class MockDataPrivacyEnv:
            """Minimal mock that simulates a data privacy task."""
            def __init__(self, task_name):
                self.task_name = task_name
                self.step_count = 0
                self._done = False
                self.current_task_goal = f"Mock task: {task_name}. Redact PII from the response."
                self.response_data = '{"user": "john.doe@example.com", "ssn": "123-45-6789"}'

            def reset(self, task_name=None):
                if task_name:
                    self.task_name = task_name
                self.step_count = 0
                self._done = False
                self.current_task_goal = f"Mock task: {self.task_name}. Redact PII from the response."
                self.response_data = '{"user": "john.doe@example.com", "ssn": "123-45-6789"}'
                return self

            def step(self, action):
                self.step_count += 1
                # Simulate a simple success condition: after 2 correct actions
                if self.step_count >= 2:
                    self._done = True
                    reward = 1.0
                else:
                    reward = 0.5
                # Update mock response
                self.response_data = '{"user": "[REDACTED]", "ssn": "[REDACTED]"}'
                # Return (obs, reward, done, error) as in the original unpack logic
                return self, reward, self._done, None

        def make(env_name):
            """Mock make function that returns a MockDataPrivacyEnv instance."""
            log_and_flush(f"[MOCK] Creating simulated environment: {env_name}", sys.stderr)
            return MockDataPrivacyEnv(env_name.split('-')[-1] if '-' in env_name else env_name)

# -------------------------------------------------------------------
# 2. Inference function (unchanged logic, works with real or mock env)
# -------------------------------------------------------------------
def run_inference(task_name):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31B-it")
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not HF_TOKEN:
        log_and_flush("[CRITICAL ERROR] HF_TOKEN environment variable not set.", sys.stderr)
        sys.exit(1)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as e:
        log_and_flush(f"[CRITICAL ERROR] Failed to init OpenAI client:\n{traceback.format_exc()}", sys.stderr)
        sys.exit(1)

    try:
        env = make("data-privacy-env")
        obs = env.reset(task_name=task_name)
    except Exception as e:
        log_and_flush(f"[CRITICAL ERROR] Failed to initialize environment:\n{traceback.format_exc()}", sys.stderr)
        sys.exit(1)

    log_and_flush(f"[START] task={task_name} env=DataPrivacyEnv model={MODEL_NAME} mock={str(MOCK_MODE).lower()}")

    done = False
    step = 0
    total_reward = 0.0
    rewards = []

    while not done and step < 10:
        step += 1

        try:
            task_goal = getattr(obs, 'current_task_goal', str(obs))
            response_data = getattr(obs, 'response_data', str(obs))

            prompt = f"""
            You are a Data Privacy Agent. 
            Task Goal: {task_goal}
            Last Observation: {response_data}
            
            Respond ONLY with a JSON object exactly like this:
            {{"method": "GET/PATCH/DELETE", "endpoint": "/route", "payload": {{...}}}}
            """

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            action_json = response.choices[0].message.content.strip()

            # Clean markdown fences
            if "```json" in action_json:
                action_json = action_json.split("```json")[1].split("```")[0].strip()
            elif "```" in action_json:
                action_json = action_json.split("```")[1].split("```")[0].strip()

            action_dict = json.loads(action_json)

            step_result = env.step(action_dict)

            # Unpack environment return tuple
            if len(step_result) == 4:
                obs, reward, done, error = step_result
            elif len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
                error = info.get('error', None)
            else:
                obs, reward, done = step_result[0], step_result[1], step_result[2]
                error = f"Unexpected tuple size: {len(step_result)}"

        except Exception as e:
            log_and_flush(f"[WARNING] Exception during step {step}:\n{traceback.format_exc()}", sys.stderr)
            action_json = json.dumps({"error": f"Failed: {str(e)}"})
            reward, done, error = 0.0, False, str(e)

        try:
            reward_float = float(reward)
        except (ValueError, TypeError):
            log_and_flush(f"[WARNING] Invalid reward type: '{reward}'. Defaulting to 0.0.", sys.stderr)
            reward_float = 0.0

        total_reward += reward_float
        rewards.append(f"{reward_float:.2f}")

        log_and_flush(f"[STEP] step={step} action={action_json} reward={reward_float:.2f} done={str(done).lower()} error={str(error).lower()}")

    success = total_reward >= 0.99
    log_and_flush(f"[END] success={str(success).lower()} steps={step} score={total_reward:.3f} rewards={','.join(rewards)}")

# -------------------------------------------------------------------
# 3. Main entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        task = os.getenv("TASK_NAME", "easy_log_redaction")
        run_inference(task)
    except Exception as e:
        log_and_flush(f"\n[FATAL UNHANDLED EXCEPTION]\n{traceback.format_exc()}", sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        log_and_flush("\n[WARNING] Process interrupted by user (Ctrl+C).", sys.stderr)
        sys.exit(130)