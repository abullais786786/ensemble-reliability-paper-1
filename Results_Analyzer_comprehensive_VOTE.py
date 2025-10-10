# analyze_results_final.py
# PURPOSE: Generates the final, comprehensive analysis including the critical "Majority Vote" baseline.

import pandas as pd
import pathlib
import sys
from datetime import datetime
from collections import Counter

# --- CONFIGURATION ---
RESULTS_DIR = pathlib.Path("outputs")
BENCHMARK_NAME = 'gsm8k'

# Define the exact filenames for all input files
ENSEMBLE_RESULTS_PATH = RESULTS_DIR / f"results_ReliableDemocratizer_{BENCHMARK_NAME}.csv"
SYNTH_ALONE_RESULTS_PATH = RESULTS_DIR / f"results_SynthesizerAlone_gsm8k.csv"
COST_LOG_PATH = RESULTS_DIR / "benchmark_cost_log.csv"

# The models list must exactly match the list in the experiment script
WORKER_MODELS = [
    'cohere/command-r7b-12-2024', 'meta-llama/llama-3.3-70b-instruct', 'qwen/qwen3-32b', 'qwen/qwen3-8b',
    'google/gemma-3-27b-it', 'meta-llama/llama-4-scout', 'google/gemini-2.5-flash-lite-preview-09-2025',
    'deepseek/deepseek-v3.2-exp',
]
SYNTHESIZER_MODEL = 'google/gemini-2.5-flash-preview-09-2025'


# --- CORE ANALYSIS FUNCTIONS ---

def load_csv(file_path, file_description):
    """Robustly loads a CSV file and returns a DataFrame."""
    if not file_path.exists():
        print(f"FATAL ERROR: {file_description} file not found at '{file_path}'")
        sys.exit(1)
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"✓ Successfully loaded {len(df)} rows from '{file_path.name}'")
        return df
    except Exception as e:
        print(f"FATAL ERROR: Could not read {file_description} file: {e}")
        sys.exit(1)

def calculate_accuracy(df, correct_col='Is_Correct'):
    """Calculates accuracy from a results DataFrame."""
    correct = (df[correct_col] == 'YES').sum()
    total = len(df)
    accuracy = (correct / total) * 100 if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}

def find_best_single_worker(df):
    """Finds the best performing individual worker model from the ensemble results."""
    best_worker = {"name": "N/A", "accuracy": 0.0, "correct": 0, "total": len(df)}
    for model in WORKER_MODELS:
        answer_col = f"{model}_Answer"
        if answer_col in df.columns:
            df[answer_col] = df[answer_col].astype(str)
            df['Correct_Answer'] = df['Correct_Answer'].astype(str)
            correct_count = (df[answer_col] == df['Correct_Answer']).sum()
            accuracy = (correct_count / len(df)) * 100 if len(df) > 0 else 0
            if accuracy > best_worker["accuracy"]:
                best_worker.update({"name": model, "accuracy": accuracy, "correct": correct_count})
    return best_worker

def calculate_reliability_metrics(df):
    """Calculates the error correction rate from the ensemble results."""
    opportunities = (df['Worker_Majority_Status'] == 'MAJORITY_WRONG').sum()
    successes = (df['Synthesizer_Correction_Status'] == 'SUCCESS').sum()
    rate = (successes / opportunities) * 100 if opportunities > 0 else 0
    return {"opportunities": opportunities, "successes": successes, "error_correction_rate": rate}

def calculate_majority_vote_performance(df):
    """
    NEW FUNCTION: Simulates a majority vote baseline and calculates its accuracy.
    """
    correct_votes = 0
    total_questions = len(df)
    
    answer_cols = [f"{model}_Answer" for model in WORKER_MODELS]
    
    for index, row in df.iterrows():
        worker_answers = [str(row[col]) for col in answer_cols if not pd.isna(row[col]) and not str(row[col]).startswith('[ERROR')]
        
        if not worker_answers:
            continue # No valid answers to vote on for this question

        # Find the most common answer (the mode)
        vote_counts = Counter(worker_answers)
        # In case of a tie, this picks one of the winners. This is a standard way to handle ties.
        voted_answer = vote_counts.most_common(1)[0][0]
        
        if str(voted_answer) == str(row['Correct_Answer']):
            correct_votes += 1
            
    accuracy = (correct_votes / total_questions) * 100 if total_questions > 0 else 0
    return {"accuracy": accuracy, "correct": correct_votes, "total": total_questions}


def calculate_total_cost(cost_df):
    """Calculates total cost from a cost log DataFrame."""
    if cost_df is None or 'total_cost_usd' not in cost_df.columns: return 0.0
    return cost_df['total_cost_usd'].astype(float).sum()


# --- REPORTING FUNCTIONS ---

def generate_final_report(ensemble_stats, synth_alone_stats, best_worker_stats, vote_stats, reliability_stats):
    """Generates the final, comprehensive text report including the vote baseline."""
    
    print("\n" + "="*80)
    print("       FINAL COMPREHENSIVE ANALYSIS REPORT")
    print("="*80)
    
    # --- Publication-Ready Performance Table ---
    print("\n" + "="*80)
    print("       TABLE 1: Comprehensive Performance Analysis (GSM8K, n=500)")
    print("="*80)
    
    table1_data = {
        "Metric": [
            "Final Accuracy (%)",
            "Correct / Total Questions",
            "Uplift vs. Synthesizer Alone (pts)",
            "Uplift vs. Best Single Worker (pts)",
            "Uplift vs. Majority Vote (pts)",
            "Error Correction Rate (%)*"
        ],
        "Reliable Democratizer (Ensemble)": [
            f"{ensemble_stats['accuracy']:.2f}",
            f"{ensemble_stats['correct']} / {ensemble_stats['total']}",
            f"+{ensemble_stats['accuracy'] - synth_alone_stats['accuracy']:.2f}",
            f"+{ensemble_stats['accuracy'] - best_worker_stats['accuracy']:.2f}",
            f"+{ensemble_stats['accuracy'] - vote_stats['accuracy']:.2f}",
            f"{reliability_stats['error_correction_rate']:.2f}"
        ],
        "Majority Vote (Baseline 1)": [
            f"{vote_stats['accuracy']:.2f}",
            f"{vote_stats['correct']} / {vote_stats['total']}",
            "N/A", "N/A", "Baseline", "N/A"
        ],
        "Synthesizer Alone (Baseline 2)": [
            f"{synth_alone_stats['accuracy']:.2f}",
            f"{synth_alone_stats['correct']} / {synth_alone_stats['total']}",
            "Baseline", "N/A", "N/A", "N/A"
        ],
        "Best Single Worker (Baseline 3)": [
            f"{best_worker_stats['accuracy']:.2f}",
            f"{best_worker_stats['correct']} / {best_worker_stats['total']}",
            "N/A", "Baseline", "N/A", "N/A"
        ]
    }
    df_table1 = pd.DataFrame(table1_data)
    print(df_table1.to_string(index=False))
    print("-" * 80)
    print(f"* Error Correction Rate is the percentage of times the synthesizer was correct when the majority of workers were incorrect ({reliability_stats['successes']}/{reliability_stats['opportunities']} instances).")
    print("="*80)


# --- MAIN EXECUTION ---
def main():
    print("\n" + "="*80)
    print("AUTOMATED COMPREHENSIVE ANALYSIS (w/ VOTE BASELINE)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    # 1. Load Data Files
    ensemble_df = load_csv(ENSEMBLE_RESULTS_PATH, "Ensemble results")
    synth_alone_df = load_csv(SYNTH_ALONE_RESULTS_PATH, "Synthesizer-Alone results")
    
    # 2. Perform All Calculations
    print("\nCalculating performance metrics...")
    ensemble_stats = calculate_accuracy(ensemble_df)
    synth_alone_stats = calculate_accuracy(synth_alone_df)
    best_worker_stats = find_best_single_worker(ensemble_df.copy())
    reliability_stats = calculate_reliability_metrics(ensemble_df)
    
    # NEW: Calculate the majority vote baseline
    vote_stats = calculate_majority_vote_performance(ensemble_df)
    
    print("✓ Calculations complete.")
    
    # 3. Generate Final Comprehensive Report
    generate_final_report(ensemble_stats, synth_alone_stats, best_worker_stats, vote_stats, reliability_stats)

if __name__ == "__main__":
    main()