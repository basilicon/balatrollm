import subprocess
import os
import glob
import json
from collections import defaultdict
import datetime
import argparse

SEEDS = ["AAAAAAA", "BBBBBBB", "CCCCCCC", "DDDDDDD", "EEEEEEE"]
RUNS_PER_SEED = 3

def run_tests():
    # Build the seed arguments
    seed_args = []
    for seed in SEEDS:
        for _ in range(RUNS_PER_SEED):
            seed_args.append(seed)
            
    print(f"Starting {len(seed_args)} runs across {len(SEEDS)} seeds...")
    
    # Run the balatrollm CLI command
    cmd = ["uv", "run", "balatrollm", "--model", "vertex_ai/openai/gpt-oss-20b-maas", "--seed"] + seed_args
    
    # We use subprocess.run so the user can see the live output
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: balatrollm exited with code {e.returncode}")
        # We continue to generate the report even if some runs failed

def generate_report():
    # Find the most recent runs in the runs directory
    # Format is runs/v1.1.1/default/<provider>/<model>/<timestamp>_<seed>/stats.json
    
    # Using a glob to find all stats.json files
    stats_files = glob.glob("runs/**/stats.json", recursive=True)
    
    if not stats_files:
        print("No stats.json files found. Did the runs complete?")
        return
        
    # Sort files by modification time (newest first)
    stats_files.sort(key=os.path.getmtime, reverse=True)
    
    # Take the top N files (where N is the number of runs we just executed)
    recent_stats = stats_files[:len(SEEDS) * RUNS_PER_SEED]
    
    report_data = []
    for filepath in recent_stats:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Extract seed from folder name (format: YYYYMMDD_HHMMSS_XXX_RED_WHITE_SEED)
            folder_name = os.path.basename(os.path.dirname(filepath))
            parts = folder_name.split("_")
            seed = parts[-1] if len(parts) >= 6 else "UNKNOWN"
            
            report_data.append({
                "seed": seed,
                "time_total_ms": data.get("time_total_ms", 0),
                "tokens_in": data.get("tokens_in_total", 0),
                "tokens_out": data.get("tokens_out_total", 0),
                "highest_round": data.get("final_round", 0)
            })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # Aggregate stats
    total_time = sum(run["time_total_ms"] for run in report_data) / 1000.0
    total_tokens_in = sum(run["tokens_in"] for run in report_data)
    total_tokens_out = sum(run["tokens_out"] for run in report_data)
    
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(f"Total Runs: {len(report_data)}")
    print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Total Tokens In: {total_tokens_in:,}")
    print(f"Total Tokens Out: {total_tokens_out:,}")
    print("-" * 50)
    
    # Round distribution
    round_dist = defaultdict(int)
    seed_highest = defaultdict(list)
    
    for run in report_data:
        round_dist[run["highest_round"]] += 1
        seed_highest[run["seed"]].append(run["highest_round"])
        
    print("Round Distribution (All Runs):")
    for r in sorted(round_dist.keys(), reverse=True):
        print(f"  Round {r}: {round_dist[r]} runs")
        
    print("-" * 50)
    print("Highest Round by Seed:")
    for seed in SEEDS:
        rounds = seed_highest.get(seed, [])
        if rounds:
            print(f"  {seed}: {rounds} (Avg: {sum(rounds)/len(rounds):.1f})")
        else:
            print(f"  {seed}: No data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BalatroLLM across multiple seeds")
    parser.add_argument("--report-only", action="store_true", help="Only generate report from recent runs, do not execute new runs")
    args = parser.parse_args()
    
    if not args.report_only:
        run_tests()
        
    generate_report()



# ==================================================
# EVALUATION REPORT
# ==================================================
# Total Runs: 15
# Total Time: 1323.63 seconds (22.06 minutes)
# Total Tokens In: 3,936,360
# Total Tokens Out: 49,658
# --------------------------------------------------
# Round Distribution (All Runs):
#   Round 13: 1 runs
#   Round 12: 1 runs
#   Round 9: 2 runs
#   Round 8: 2 runs
#   Round 7: 2 runs
#   Round 6: 3 runs
#   Round 5: 1 runs
#   Round 4: 2 runs
#   Round 3: 1 runs
# --------------------------------------------------
# Highest Round by Seed:
#   AAAAAAA: [3, 12, 7, 6] (Avg: 7.0)
#   BBBBBBB: [13, 6, 4] (Avg: 7.7)
#   CCCCCCC: [5, 8] (Avg: 6.5)
#   DDDDDDD: [6, 8, 4] (Avg: 6.0)
#   EEEEEEE: [9, 7, 9] (Avg: 8.3)