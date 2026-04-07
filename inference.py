import os
import sys
import json
from openai import OpenAI
from openenv.core import make

def run_inference():
    # 1. Load Mandatory Environment Variables
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    if not all([api_base_url, model_name, hf_token]):
        print("Error: Missing required environment variables (API_BASE_URL, MODEL_NAME, HF_TOKEN)")
        sys.exit(1)

    # 2. Initialize the OpenAI Client
    # We pass the HF Token as the API key to authenticate with the remote model
    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token,
    )

    # 3. Initialize OpenEnv (With Phase 2 Try/Except Safety Net)
    try:
        # Connects to the local Dockerized instance running on port 7860
        env = make("http://localhost:7860")
    except ImportError as ie:
        print(f"Import Error during initialization: {ie}")
        sys.exit(1)
    except Exception as e:
        print(f"Unhandled exception during environment initialization: {e}")
        sys.exit(1)

    # 4. Agent Interaction Loop
    try:
        # Reset environment to get the initial observation
        obs = env.reset()
        
        # --- [START] MANDATORY LOG ---
        print(f"[START] {json.dumps({'observation': obs})}")

        done = False
        step_count = 0
        max_steps = 15  # Prevents infinite loops and ensures we stay under the 20 min limit
        total_reward = 0.0

        # System prompt instructing the LLM on how to behave in your specific environment
        system_prompt = (
            "You are a Data Privacy Remediation Agent. Your goal is to solve compliance tasks based on observations. "
            "You must output ONLY valid JSON representing your next action. "
            "Your output must contain exactly these keys: 'method', 'endpoint', and 'payload'."
        )

        # Message history to feed the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Initial Observation: {json.dumps(obs)}"}
        ]

        while not done and step_count < max_steps:
            # Ask the LLM for the next action
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"} # Forces the LLM to return valid JSON
            )
            
            # Extract and parse the action
            try:
                action_str = response.choices[0].message.content
                action = json.loads(action_str)
            except Exception as e:
                # If parsing fails, create a dummy action so the script doesn't crash
                action = {"method": "ERROR", "endpoint": "/parse_fail", "payload": {"error": str(e)}}

            # Execute the action in the environment
            next_obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            
            # --- [STEP] MANDATORY LOG ---
            step_log = {
                "step": step_count,
                "action": action,
                "observation": next_obs,
                "reward": float(reward),
                "done": bool(done),
                "info": info
            }
            print(f"[STEP] {json.dumps(step_log)}")

            # Append the interaction to the message history so the LLM remembers what it did
            messages.append({"role": "assistant", "content": json.dumps(action)})
            messages.append({
                "role": "user", 
                "content": f"Observation: {json.dumps(next_obs)}\nReward: {reward}\nDone: {done}"
            })
            
            step_count += 1

        # --- [END] MANDATORY LOG ---
        end_log = {
            "total_steps": step_count,
            "total_reward": total_reward,
            "success": done
        }
        print(f"[END] {json.dumps(end_log)}")

    except Exception as e:
        # Catch any other runtime errors so the script fails gracefully instead of crashing the grader
        print(f"Unhandled exception during inference loop: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_inference()