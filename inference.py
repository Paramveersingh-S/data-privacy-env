import os
import sys
import json
import traceback


def log_and_flush(message, stream=sys.stdout):
    stream.write(message + "\n")
    stream.flush()

# Safely import dependencies
try:
    from openai import OpenAI
except ImportError as e:
    log_and_flush(f"[CRITICAL ERROR] Failed to import OpenAI:\n{traceback.format_exc()}", sys.stderr)
    sys.exit(1)

try:
    from openenv.core import make
except ImportError:
    try:
        from openenv import make
    except ImportError as e:
        log_and_flush(f"[CRITICAL ERROR] Could not import 'make' from openenv:\n{traceback.format_exc()}", sys.stderr)
        sys.exit(1)


def run_inference(task_name):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31B-it")
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not HF_TOKEN:
        log_and_flush("[CRITICAL ERROR] HF_TOKEN environment variable not set.", sys.stderr)
        sys.exit(1)

    # Initialize client safely
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as e:
        log_and_flush(f"[CRITICAL ERROR] Failed to init OpenAI client:\n{traceback.format_exc()}", sys.stderr)
        sys.exit(1)

    # Initialize environment safely
    try:
        env = make("data-privacy-env")
        obs = env.reset(task_name=task_name)
    except Exception as e:
        log_and_flush(f"[CRITICAL ERROR] Failed to initialize environment:\n{traceback.format_exc()}", sys.stderr)
        sys.exit(1)
    
    # 1. MANDATORY START LOG
    log_and_flush(f"[START] task={task_name} env=DataPrivacyEnv model={MODEL_NAME}")
    
    done = False
    step = 0
    total_reward = 0.0
    rewards = []

    # Run for a maximum of 10 steps to prevent infinite loops
    while not done and step < 10:
        step += 1
        
        try:
            # Safely extract observation properties
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
            
            # Clean potential markdown backticks from the LLM
            if "```json" in action_json:
                action_json = action_json.split("```json")[1].split("```")[0].strip()
            elif "```" in action_json:
                action_json = action_json.split("```")[1].split("```")[0].strip()
            
            action_dict = json.loads(action_json)
            
            # Take step in the environment
            step_result = env.step(action_dict)
            
            # SAFELY UNPACK ENV TUPLE: Environments frequently change their return sizes
            if len(step_result) == 4:
                obs, reward, done, error = step_result
            elif len(step_result) == 5: # Handles modern Gym API: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
                error = info.get('error', None)
            else:
                obs, reward, done = step_result[0], step_result[1], step_result[2]
                error = f"Unexpected tuple size: {len(step_result)}"

        except Exception as e:
            # Log the full traceback without crashing the loop entirely
            log_and_flush(f"[WARNING] Exception during step {step}:\n{traceback.format_exc()}", sys.stderr)
            action_json = json.dumps({"error": f"Failed: {str(e)}"})
            reward, done, error = 0.0, False, str(e)

        # Safely convert reward to float
        try:
            reward_float = float(reward)
        except (ValueError, TypeError):
            log_and_flush(f"[WARNING] Invalid reward type returned: '{reward}'. Defaulting to 0.0.", sys.stderr)
            reward_float = 0.0

        total_reward += reward_float
        rewards.append(f"{reward_float:.2f}")
        
        # 2. MANDATORY STEP LOG
        log_and_flush(f"[STEP] step={step} action={action_json} reward={reward_float:.2f} done={str(done).lower()} error={str(error).lower()}")

    # Determine success based on the sum of rewards
    success = total_reward >= 0.99
    
    # 3. MANDATORY END LOG
    log_and_flush(f"[END] success={str(success).lower()} steps={step} score={total_reward:.3f} rewards={','.join(rewards)}")

if __name__ == "__main__":
    # The Global Catch-All: Nothing escapes this block without logging
    try:
        task = os.getenv("TASK_NAME", "easy_log_redaction")
        run_inference(task)
    except Exception as e:
        log_and_flush(f"\n[FATAL UNHANDLED EXCEPTION]\n{traceback.format_exc()}", sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        log_and_flush("\n[WARNING] Process interrupted by user (Ctrl+C).", sys.stderr)
        sys.exit(130)