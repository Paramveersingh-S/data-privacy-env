import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

from env import DataPrivacyEnv
from models import PrivacyAction

# MANDATORY ENVIRONMENT VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy_key"

# INFERENCE CONFIGURATION
TASK_NAME = os.getenv("TASK_NAME", "easy_log_redaction")
BENCHMARK = "DataPrivacyEnv"
MAX_STEPS = 15
TEMPERATURE = 0.1  # Very low temperature for highly deterministic JSON output

# SYSTEM PROMPT
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous Data Privacy Compliance Agent.
    Your goal is to interact with a company's internal JSON API to find and remediate leaked PII.
    
    You must reply with exactly ONE valid JSON object representing your next API request.
    Do not include markdown blocks, text formatting, or conversational text. ONLY JSON.
    
    Format:
    {
        "method": "GET|POST|PATCH|DELETE",
        "endpoint": "/users OR /users/{id} OR /logs OR /logs/{id}",
        "payload": { "key": "value" } // Optional, use null if not needed
    }
    """
).strip()

# --- STRICT LOGGING FORMATTERS ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = str(error) if error else "null"
    done_val = str(done).lower()
    # Flatten the action string so it doesn't break the single-line regex parser
    action_oneline = action.replace("\n", "").replace("\r", "")
    print(f"[STEP] step={step} action={action_oneline} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- LLM INTERACTION ---
def get_model_action(client: OpenAI, obs_data: dict, history: List[str]) -> tuple[PrivacyAction, str]:
    user_prompt = textwrap.dedent(f"""
    Current Observation:
    {json.dumps(obs_data, indent=2)}
    
    Past steps history:
    {chr(10).join(history[-3:]) if history else "None"}
    
    Generate your next JSON action.
    """).strip()
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=250,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        
        # Clean markdown if the model hallucinated code blocks
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
        action_dict = json.loads(response_text)
        return PrivacyAction(**action_dict), response_text
        
    except Exception as e:
        # If the LLM hallucinates bad JSON, we fallback to a safe 'dummy' action.
        # This prevents the script from crashing and allows the env to return a 404 error cleanly.
        fallback_action = PrivacyAction(method="GET", endpoint="/invalid", payload=None)
        return fallback_action, f'{{"error": "Failed to parse JSON: {e}"}}'

# --- MAIN LOOP ---
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DataPrivacyEnv()
    
    history = []
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Start the environment
        obs = env.reset(task_name=TASK_NAME)
        
        for step in range(1, MAX_STEPS + 1):
            steps_taken = step
            
            # 1. Ask the LLM what to do
            action_obj, raw_action_str = get_model_action(client, obs.model_dump(), history)
            
            # 2. Execute the action in our environment
            obs, reward, done, error = env.step(action_obj)
            
            # 3. Record metrics
            rewards.append(reward)
            score = reward  # In our env, the step reward is cumulative/final
            
            # 4. Strictly log the step
            log_step(step=step, action=raw_action_str, reward=reward, done=done, error=error)
            
            # 5. Update history for the LLM's context
            history.append(f"Action: {raw_action_str} | Status: {obs.status_code}")
            
            if done:
                break
                
        # The OpenEnv requirement states success is when the score threshold is met (1.0 for perfect)
        success = score >= 0.99
        
    except Exception as e:
        # We silently catch fatal errors so we ALWAYS print the [END] block per instructions
        pass
        
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()