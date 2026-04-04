import os
import json
from openai import OpenAI
from openenv_core import make

# Match the exact format required by the hackathon checklist
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31B-it")
HF_TOKEN = os.getenv("HF_TOKEN") # NO DEFAULT HERE

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def run_inference(task_name):
    env = make("data-privacy-env")
    obs = env.reset(task_name=task_name)
    
    print(f"[START] task={task_name} env=DataPrivacyEnv model={MODEL_NAME}")
    
    done = False
    step = 0
    total_reward = 0
    rewards = []

    while not done and step < 10:
        step += 1
        
        prompt = f"""
        You are a Data Privacy Agent. 
        Task Goal: {obs.current_task_goal}
        Last Observation: {obs.response_data}
        
        Respond ONLY with a JSON object:
        {{"method": "GET/PATCH/DELETE", "endpoint": "/route", "payload": {{...}}}}
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            action_json = response.choices[0].message.content.strip()
            # Clean potential markdown backticks
            if "```json" in action_json:
                action_json = action_json.split("```json")[1].split("```")[0].strip()
            
            action_dict = json.loads(action_json)
            obs, reward, done, error = env.step(action_dict)
        except Exception as e:
            action_json = json.dumps({"error": f"Failed to parse JSON: {str(e)}"})
            reward, done, error = 0.0, False, str(e)

        total_reward += reward
        rewards.append(f"{reward:.2f}")
        print(f"[STEP] step={step} action={action_json} reward={reward:.2f} done={str(done).lower()} error={str(error).lower()}")

    success = total_reward >= 0.99
    print(f"[END] success={str(success).lower()} steps={step} score={total_reward:.3f} rewards={','.join(rewards)}")

if __name__ == "__main__":
    task = os.getenv("TASK_NAME", "easy_log_redaction")
    run_inference(task)
