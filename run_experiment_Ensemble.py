# run_experiment.py
# PURPOSE: Runs the "Reliable Democratizer" ensemble on a benchmark.
# This is the ONLY script needed to generate the raw data for the paper.

import os
import sys
import time
import logging
import pathlib
import concurrent.futures
from openai import OpenAI
from datasets import load_dataset
import re
import csv
from datetime import datetime, UTC

# --- CONFIGURATION ---

# 1. Worker Models & Synthesizer (The "Reliable Democratizer" Roster)
WORKER_MODELS = [
    'cohere/command-r7b-12-2024',
    'meta-llama/llama-3.3-70b-instruct',
    'qwen/qwen3-32b',
    'qwen/qwen3-8b',
    'google/gemma-3-27b-it',
    'meta-llama/llama-4-scout',
    'google/gemini-2.5-flash-lite-preview-09-2025',
    'deepseek/deepseek-v3.2-exp',
]
SYNTHESIZER_MODEL = 'google/gemini-2.5-flash-preview-09-2025'

# 2. Benchmark Configuration (Primary benchmark is GSM8K for reliability analysis)
BENCHMARK_NAME = 'gsm8k' # <-- Primary benchmark for reliability
BENCHMARK_CONFIG = 'main'    # <-- Config for gsm8k is 'main'
# To switch to MMLU, change to: BENCHMARK_NAME = 'cais/mmlu', BENCHMARK_CONFIG = 'all'
BENCHMARK_SPLIT = 'test'
NUM_QUESTIONS_TO_TEST = 500 # <-- Set to a statistically significant number for the paper

# 3. Model-Specific Temperature Configuration
MODEL_TEMPERATURE_CONFIG = { 'default': 0 } # Setting all to 0 for deterministic output

# 4. Execution and API Settings
MAX_SIMULTANEOUS_WORKERS = 3
REQUEST_TIMEOUT = 180
OPENROUTER_API_KEY = "API_KEY_HERE" # IMPORTANT: Replace with your key

# 5. Output Configuration
OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_CSV_PATH = OUTPUT_DIR / f"results_ReliableDemocratizer_{BENCHMARK_NAME.split('/')[-1]}.csv"
COST_LOG_PATH = OUTPUT_DIR / "benchmark_cost_log.csv"

# 6. Cost Tracking Price List (per 1 Million Tokens) - Abbreviated for brevity
PRICE_LIST = {
    'google/gemini-2.5-flash-preview-09-2025': {"input": 0.3, "output": 2.5}, 'google/gemini-2.5-flash-lite-preview-09-2025': {"input": 0.1, "output": 0.4},
    'google/gemma-3-27b-it': {"input": 0.07, "output": 0.26}, 'deepseek/deepseek-v3.2-exp': {"input": 0.27, "output": 0.41},
    'qwen/qwen3-32b': {"input": 0.03, "output": 0.13}, 'qwen/qwen3-8b': {"input": 0.035, "output": 0.138},
    'meta-llama/llama-4-scout': {"input": 0.08, "output": 0.30}, 'meta-llama/llama-3.3-70b-instruct': {"input": 0.04, "output": 0.12},
    'cohere/command-r7b-12-2024': {"input": 0.038, "output": 0.15}
}

# --- Setup and Helper Functions ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def setup_output_files():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Setup cost log header
    if not (COST_LOG_PATH.is_file() and COST_LOG_PATH.stat().st_size > 0):
        with open(COST_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["timestamp_utc", "benchmark", "question_id", "model", "prompt_tokens", "completion_tokens", "total_cost_usd"])
    
    # Setup results CSV with NEW reliability headers
    if not (OUTPUT_CSV_PATH.is_file() and OUTPUT_CSV_PATH.stat().st_size > 0):
        with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            header = [
                "Question_ID", "Question_Text", "Correct_Answer", "Synthesizer_Answer", "Is_Correct",
                # --- NEW RELIABILITY COLUMNS ---
                "Num_Correct_Workers", "Worker_Majority_Status", "Synthesizer_Correction_Status"
            ] + [f"{model}_Answer" for model in WORKER_MODELS] + \
                [f"{model}_Full_Response" for model in WORKER_MODELS]
            writer.writerow(header)

def log_api_call_details(benchmark_name, question_id, model, usage_info):
    if not usage_info: return
    try:
        p_tok, c_tok = usage_info.prompt_tokens, usage_info.completion_tokens
        prices = PRICE_LIST.get(model, {"input": 0, "output": 0})
        cost = ((p_tok / 1e6) * prices["input"]) + ((c_tok / 1e6) * prices["output"])
        row = [datetime.now(UTC).isoformat(), benchmark_name, question_id, model, p_tok, c_tok, f"{cost:.8f}"]
        with open(COST_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)
    except Exception as e:
        logging.error(f"Error logging cost for {model}: {e}")

def call_llm(model_name: str, prompt: str, client: OpenAI) -> tuple[str, object]:
    """Calls the LLM with improved error handling and retries."""
    temp = MODEL_TEMPERATURE_CONFIG.get(model_name, MODEL_TEMPERATURE_CONFIG['default'])
    last_exception = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                seed=42
            )
            # Check for a valid, complete response before returning
            if response and response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip(), response.usage
            else:
                logging.warning(f"Attempt {attempt+1} for {model_name} returned an incomplete response.")
                last_exception = "IncompleteResponse"
        except Exception as e:
            last_exception = e
            logging.error(f"Attempt {attempt+1} failed for {model_name}: {e}")
        
        if attempt < 4:
            time.sleep(5 * (attempt + 1)) # Exponential backoff
    
    return f"[ERROR: {type(last_exception).__name__} after 5 retries]", None

def format_prompt_for_worker(q_data: dict) -> str:
    # MMLU format
    if 'choices' in q_data:
        question = q_data['question']
        choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(q_data['choices'])])
    # GSM8K format
    else:
        question = q_data['question']
        choices = "Provide the final numerical answer." # GSM8K has no choices
    
    return f"Please answer the following question.\n\n**Instructions:**\n1. First, provide your detailed, step-by-step reasoning.\n2. After your reasoning, on a new line, state your final answer in the format: `Final Answer: [Your Answer]`\n\n**Question:**\n{question}\n\n**Options:**\n{choices}"

def format_prompt_for_synthesizer(q_data: dict, worker_responses: dict) -> str:
    question = q_data.get('question', '')
    choices = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(q_data.get('choices', []))])
    responses = "".join([f"--- Analysis from Model: {m} ---\n{r}\n" for m, r in worker_responses.items()])
    return f"You are an impartial judge. Your task is to determine the correct answer by evaluating the reasoning of 8 AI assistants. Do not simply count votes. The quality of the reasoning is what matters. Discard analyses with clear errors.\n\n**Original Question:**\n{question}\n\n**Options:**\n{choices}\n\n**Here are the 8 AI analyses to evaluate:**\n{responses}\n\n**Your Final Verdict (single capital letter, or final numerical answer if no options):**"

def extract_final_answer(full_response: str) -> str:
    """A more robust function to extract the final answer from a messy LLM response."""
    # 1. Clean the response of common artifacts like backticks and commas in numbers
    cleaned_response = full_response.strip().replace('`', '').replace(',', '')

    # 2. Strongest pattern: Look for "Final Answer: [answer]"
    match = re.search(r"Final Answer:\s*(.*)", cleaned_response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. Next best: Look for a standalone letter (for MMLU)
    match = re.search(r"^\s*([A-Z])\s*$", cleaned_response, re.MULTILINE)
    if match:
        return match.group(1).upper()

    # 4. Robust fallback for numbers (GSM8K): Find the LAST number in the string.
    #    This is effective because LLMs often conclude with the final answer.
    numerical_matches = re.findall(r"[-+]?\d*\.?\d+", cleaned_response)
    if numerical_matches:
        return numerical_matches[-1]

    # 5. If all else fails, declare a parsing error.
    return "[ERROR: Could not parse answer]"

def get_correct_answer(q_data: dict) -> str:
    """Gets the correct answer, handling MMLU and GSM8K formats."""
    answer_field = q_data.get('answer')
    if isinstance(answer_field, int): # Handles MMLU
        return chr(65 + answer_field)
    
    if isinstance(answer_field, str): # Handles GSM8K
        # The ground truth is often "#### 123", this extracts just "123"
        match = re.findall(r"[-+]?\d*\.\d+|\d+", answer_field)
        if match:
            return match[-1]
            
    return "[ERROR: No ground truth]"

def run_ensemble_and_save(q_data: dict, q_index: int, client: OpenAI):
    q_id = f"Q{q_index + 1}"
    logging.info(f"--- Processing {q_id}/{NUM_QUESTIONS_TO_TEST} ---")

    worker_prompt = format_prompt_for_worker(q_data)
    worker_responses = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_SIMULTANEOUS_WORKERS) as executor:
        future_to_model = {executor.submit(call_llm, m, worker_prompt, client): m for m in WORKER_MODELS}
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            response_text, usage = future.result()
            worker_responses[model] = response_text
            log_api_call_details(BENCHMARK_NAME, q_id, model, usage)

    synth_prompt = format_prompt_for_synthesizer(q_data, worker_responses)
    synth_response, synth_usage = call_llm(SYNTHESIZER_MODEL, synth_prompt, client)
    log_api_call_details(BENCHMARK_NAME, q_id, SYNTHESIZER_MODEL, synth_usage)
    
    synth_answer = extract_final_answer(synth_response)
    correct_answer = get_correct_answer(q_data)
    
    # --- START: NEW Reliability Metrics Calculation ---
    worker_answers = [extract_final_answer(resp) for resp in worker_responses.values()]
    valid_workers = [ans for ans in worker_answers if not ans.startswith('[ERROR')]
    num_correct_workers = sum(1 for ans in valid_workers if ans == correct_answer)
    
    if not valid_workers:
        worker_majority_status = "ALL_FAILED"
    elif num_correct_workers > len(valid_workers) / 2:
        worker_majority_status = "MAJORITY_CORRECT"
    else:
        worker_majority_status = "MAJORITY_WRONG"

    synthesizer_is_correct = (synth_answer == correct_answer)
    correction_status = "NOT_APPLICABLE"
    if worker_majority_status == "MAJORITY_WRONG":
        correction_status = "SUCCESS" if synthesizer_is_correct else "FAIL"
    # --- END: NEW Reliability Metrics Calculation ---
    
    # Build CSV row
    row = [
        q_id, q_data['question'], correct_answer, synth_answer, "YES" if synthesizer_is_correct else "NO",
        num_correct_workers, worker_majority_status, correction_status
    ]
    row.extend([extract_final_answer(worker_responses.get(m, "")) for m in WORKER_MODELS])
    row.extend([worker_responses.get(m, "").strip() for m in WORKER_MODELS])
    
    with open(OUTPUT_CSV_PATH, 'a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(row)
    logging.info(f"Saved results for {q_id}. Ensemble correct: {'YES' if synthesizer_is_correct else 'NO'}")

def main():
    start_time = time.time()
    setup_output_files()
    logging.info(f"--- Starting Reliability Experiment: {os.path.basename(__file__)} ---")
    if "YOUR_OPENROUTER_API_KEY_HERE" in OPENROUTER_API_KEY:
        logging.error("FATAL: Please replace 'YOUR_OPENROUTER_API_KEY_HERE' in the script.")
        sys.exit(1)

    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        dataset = load_dataset(BENCHMARK_NAME, BENCHMARK_CONFIG, split=BENCHMARK_SPLIT)
        subset = dataset.shuffle(seed=42).select(range(NUM_QUESTIONS_TO_TEST))
    except Exception as e:
        logging.error(f"Fatal error during setup: {e}"); sys.exit(1)

    for i, q_data in enumerate(subset):
        run_ensemble_and_save(q_data, i, client)
        
    logging.info(f"\n--- Experiment Finished in {time.time() - start_time:.2f}s ---")
    logging.info(f"Results saved to '{OUTPUT_CSV_PATH}'")

if __name__ == "__main__":
    main()