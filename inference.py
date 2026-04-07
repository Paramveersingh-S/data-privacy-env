import os
import sys
import json
from openai import OpenAI

try:
    from openenv.core import make
except ImportError:
    try:
        from openenv import make
    except ImportError:
        print("Error: Could not import 'make' from openenv or openenv.core")
        sys.exit(1)


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31B-it")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable not set. Grader must provide this.")
    sys.exit(1)

# Initialize the standard OpenAI Client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def run_inference(task_name):
    # Safely initialize the environment
    try:
        env = make("data-privacy-env")
        obs = env.reset(task_name=task_name)
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        sys.exit(1)
    
    # 1. MANDATORY START LOG
    print(f"[START] task={task_name} env=DataPrivacyEnv model={MODEL_NAME}")
    
    done = False
    step = 0
    total_reward = 0.0
    rewards = []

    # Run for a maximum of 10 steps to prevent infinite loops
    while not done and step < 10:
        step += 1
        
        prompt = f"""
        You are a Data Privacy Agent. 
        Task Goal: {getattr(obs, 'current_task_goal', str(obs))}
        Last Observation: {getattr(obs, 'response_data', str(obs))}
        
        Respond ONLY with a JSON object exactly like this:
        {{"method": "GET/PATCH/DELETE", "endpoint": "/route", "payload": {{...}}}}
        """
        
    
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            action_json = response.choices[0].message.content.strip()
            
            # Clean potential markdown backticks from the LLM
            if "```json" in action_json:
                action_json = action_json.split("```json")[1].split("```")[0].strip()
            elif "```" in action_json:
                action_json = action_json.split("```")[1].split("```")[0].strip()
            
            action_dict = json.loads(action_json)
            
            # Take step in the environment
            obs, reward, done, error = env.step(action_dict)
            
        except Exception as e:
            # If the LLM returns bad JSON or the network fails, DO NOT CRASH.
            # Log the error and give 0 reward for this step.
            action_json = json.dumps({"error": f"Failed: {str(e)}"})
            reward, done, error = 0.0, False, str(e)

        total_reward += float(reward)
        rewards.append(f"{reward:.2f}")
        
        # 2. MANDATORY STEP LOG
        print(f"[STEP] step={step} action={action_json} reward={reward:.2f} done={str(done).lower()} error={str(error).lower()}")

    # Determine success based on the sum of rewards
    success = total_reward >= 0.99
    
    # 3. MANDATORY END LOG
    print(f"[END] success={str(success).lower()} steps={step} score={total_reward:.3f} rewards={','.join(rewards)}")

if __name__ == "__main__":
    # If the grader doesn't specify a task, default to the first one
    task = os.getenv("TASK_NAME", "easy_log_redaction")
    run_inference(task)