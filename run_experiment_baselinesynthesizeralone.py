# run_synthesizer_alone.py
# PURPOSE: Runs ONLY the synthesizer model on the benchmark to get its baseline performance.

import os
import sys
import time
import logging
import pathlib
from openai import OpenAI
from datasets import load_dataset
import re
import csv
from datetime import datetime, UTC

# --- CONFIGURATION ---

# 1. Model to Test (Using the Synthesizer model from the main experiment)
MODEL_TO_TEST = 'google/gemini-2.5-flash-preview-09-2025'

# 2. Benchmark Configuration
BENCHMARK_NAME = 'gsm8k'
BENCHMARK_CONFIG = 'main'
BENCHMARK_SPLIT = 'test'
NUM_QUESTIONS_TO_TEST = 500 # Should be the same as your main experiment

# 3. API Settings and Cost List (Copied from the main script)
OPENROUTER_API_KEY = ""API_KEY_HERE"
MODEL_TEMPERATURE_CONFIG = { 'default': 0 }
PRICE_LIST = { 'google/gemini-2.5-flash-preview-09-2025': {"input": 0.3, "output": 2.5} } # Only need the price for the model being tested

# 4. Output Configuration
OUTPUT_DIR = pathlib.Path("outputs")
# A new, unique filename for this baseline experiment
OUTPUT_CSV_PATH = OUTPUT_DIR / f"results_SynthesizerAlone_{BENCHMARK_NAME.split('/')[-1]}.csv"
COST_LOG_PATH = OUTPUT_DIR / "benchmark_cost_log.csv"

# --- Setup and Helper Functions ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def setup_output_files():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (OUTPUT_CSV_PATH.is_file() and OUTPUT_CSV_PATH.stat().st_size > 0):
        with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            # Simplified header for a single-model run
            header = ["Question_ID", "Question_Text", "Correct_Answer", "Model_Answer", "Is_Correct", "Full_Response"]
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
    temp = MODEL_TEMPERATURE_CONFIG.get(model_name, MODEL_TEMPERATURE_CONFIG['default'])
    last_exception = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}], temperature=temp, seed=42)
            if response and response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip(), response.usage
            else:
                last_exception = "IncompleteResponse"
        except Exception as e:
            last_exception = e
        logging.error(f"Attempt {attempt+1} failed for {model_name}: {last_exception}")
        if attempt < 4: time.sleep(5 * (attempt + 1))
    return f"[ERROR: {type(last_exception).__name__} after 5 retries]", None

def format_prompt_for_model(q_data: dict) -> str:
    question = q_data.get('question', '')
    choices_text = "Provide the final numerical answer." # GSM8K default
    if 'choices' in q_data:
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(q_data['choices'])])
    return f"Please answer the following question.\n\n**Instructions:**\n1. First, provide your detailed, step-by-step reasoning.\n2. After your reasoning, on a new line, state your final answer in the format: `Final Answer: [Your Answer]`\n\n**Question:**\n{question}\n\n**Options:**\n{choices_text}"

def extract_final_answer(full_response: str) -> str:
    cleaned_response = full_response.strip().replace('`', '').replace(',', '')
    match = re.search(r"Final Answer:\s*(.*)", cleaned_response, re.IGNORECASE | re.DOTALL)
    if match: return match.group(1).strip()
    match = re.search(r"^\s*([A-Z])\s*$", cleaned_response, re.MULTILINE)
    if match: return match.group(1).upper()
    numerical_matches = re.findall(r"[-+]?\d*\.?\d+", cleaned_response)
    if numerical_matches: return numerical_matches[-1]
    return "[ERROR: Could not parse answer]"

def get_correct_answer(q_data: dict) -> str:
    answer_field = q_data.get('answer')
    if isinstance(answer_field, int): return chr(65 + answer_field)
    if isinstance(answer_field, str):
        match = re.findall(r"[-+]?\d*\.\d+|\d+", answer_field)
        if match: return match[-1]
    return "[ERROR: No ground truth]"

def run_single_model_and_save(q_data: dict, q_index: int, client: OpenAI):
    q_id = f"Q{q_index + 1}"
    logging.info(f"--- Processing {q_id}/{NUM_QUESTIONS_TO_TEST} ---")
    
    prompt = format_prompt_for_model(q_data)
    
    # --- MINIMAL CHANGE: Call only one model, not the worker ensemble ---
    full_response, usage_info = call_llm(MODEL_TO_TEST, prompt, client)
    log_api_call_details(BENCHMARK_NAME, q_id, MODEL_TO_TEST, usage_info)
    
    model_answer = extract_final_answer(full_response)
    correct_answer = get_correct_answer(q_data)
    is_correct = "YES" if model_answer == correct_answer else "NO"
    
    # Simplified row for a single-model run
    row = [q_id, q_data['question'], correct_answer, model_answer, is_correct, full_response]
    
    with open(OUTPUT_CSV_PATH, 'a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(row)
    logging.info(f"Saved results for {q_id}. Model correct: {is_correct}")

def main():
    start_time = time.time()
    setup_output_files()
    logging.info(f"--- Starting Synthesizer-Alone Baseline Run ---")

    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        dataset = load_dataset(BENCHMARK_NAME, BENCHMARK_CONFIG, split=BENCHMARK_SPLIT)
        subset = dataset.shuffle(seed=42).select(range(NUM_QUESTIONS_TO_TEST))
    except Exception as e:
        logging.error(f"Fatal error during setup: {e}"); sys.exit(1)

    for i, q_data in enumerate(subset):
        run_single_model_and_save(q_data, i, client)
        
    logging.info(f"\n--- Experiment Finished in {time.time() - start_time:.2f}s ---")
    logging.info(f"Results saved to '{OUTPUT_CSV_PATH}'")

if __name__ == "__main__":
    main()