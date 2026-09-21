import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

from langsmith import Client
import pandas as pd

client = Client()

# ===== CONFIG =====
PROJECT_NAME = "hopebot-3-0-phase2"

# Two participants excluded, for different reasons:
# - P05: technical failure during conversation, session never reached PHQ-9 completion / tool calls
# - P08: completed full bot session but did not complete the feedback survey (reason not captured)
KEEP_IDS = ["P04", "P06", "P07", "P010", "P011", "P012", "P013", "P014", "P016"]

FAILURE_INDICATORS = ["An error occurred", "No relevant", "No psychoeducational content", "No relevant session preparation"]

ALL_KNOWN_TOOLS = ["send_email", "psychoeducation", "session_prep", "calendar_input"]

P50_THRESHOLD = 1.5
P99_THRESHOLD = 5.0
COMPLETION_THRESHOLD = 0.90


# ===== FETCH RUNS =====
tool_runs = client.list_runs(project_name=PROJECT_NAME, run_type="tool")
llm_runs = client.list_runs(project_name=PROJECT_NAME, run_type="llm")


# ===== BUILD TOOL (TASK COMPLETION) DATAFRAME =====
tool_results = []

for run in tool_runs:
    tool_name = run.name
    output_dict = run.outputs.get("output", {}) if run.outputs else {}

    if isinstance(output_dict, dict):
        content = output_dict.get("content", "")
        status = output_dict.get("status", "unknown")
    else:
        content = str(output_dict)
        status = "unknown"

    participant_id = run.extra.get("metadata", {}).get("participant_id") if run.extra else None

    success = (status == "success") and not any(content.startswith(ind) for ind in FAILURE_INDICATORS)

    tool_results.append({
        "run_id": run.id,
        "participant_id": participant_id,
        "tool_name": tool_name,
        "output": content,
        "status": status,
        "success": success
    })

tool_df = pd.DataFrame(tool_results)
print(f"tool_df participants (unfiltered): {sorted(tool_df['participant_id'].dropna().unique())}")

tool_df = tool_df[tool_df["participant_id"].isin(KEEP_IDS)]


# ===== BUILD LLM (LATENCY / TOKEN / EMERGENCY PATHWAY) DATAFRAME =====
llm_results = []

for run in llm_runs:
    participant_id = run.extra.get("metadata", {}).get("participant_id") if run.extra else None
    latency = (run.end_time - run.start_time).total_seconds() if run.end_time and run.start_time else None
    total_tokens = run.total_tokens if hasattr(run, "total_tokens") else None

    inputs_str = str(run.inputs) if run.inputs else ""
    is_emergency_flag = "Pathway: emergency" in inputs_str

    llm_results.append({
        "run_id": run.id,
        "participant_id": participant_id,
        "latency_seconds": latency,
        "total_tokens": total_tokens,
        "is_emergency_pathway_run": is_emergency_flag
    })

llm_df = pd.DataFrame(llm_results)
print(f"llm_df participants (unfiltered): {sorted(llm_df['participant_id'].dropna().unique())}")

llm_df = llm_df[llm_df["participant_id"].isin(KEEP_IDS)]


# ===== TASK COMPLETION =====
overall_completion = tool_df["success"].mean()
print(f"\nOverall task completion: {overall_completion:.1%} "
      f"(threshold: ≥{COMPLETION_THRESHOLD:.0%}) — "
      f"{'PASS' if overall_completion >= COMPLETION_THRESHOLD else 'FAIL'}")

print("\nTask completion rate by participant:")
print(tool_df.groupby("participant_id")["success"].mean())

# Exact call counts per tool (x/x successful)
tool_call_counts = tool_df.groupby("tool_name")["success"].agg(
    n_successful="sum",
    n_total="count"
)
tool_call_counts["success_str"] = tool_call_counts.apply(
    lambda row: f"{int(row['n_successful'])}/{int(row['n_total'])}", axis=1
)
tool_call_counts["success_rate"] = (tool_call_counts["n_successful"] / tool_call_counts["n_total"]).round(3)

print("\nTask completion rate by feature (tool):")
print(tool_call_counts[["success_str", "success_rate"]])

# Tools never invoked by any participant in this dataset
called_tools = set(tool_df["tool_name"].unique())
never_called = [t for t in ALL_KNOWN_TOOLS if t not in called_tools]
if never_called:
    print(f"\nTools never invoked in this dataset: {never_called}")
else:
    print("\nAll known tools were invoked at least once.")


# ===== FAILURE ANALYSIS =====
failures = tool_df[~tool_df["success"]]

print(f"\nTotal failed tool calls: {len(failures)} / {len(tool_df)} ({len(failures)/len(tool_df):.1%})")

if not failures.empty:
    print("\nFailures by tool:")
    print(failures.groupby("tool_name").size().sort_values(ascending=False))

    print("\nFailures by participant:")
    print(failures.groupby("participant_id").size().sort_values(ascending=False))

    def classify_failure(row):
        if row["status"] != "success":
            return f"status={row['status']}"
        output_str = str(row["output"])
        for indicator in FAILURE_INDICATORS:
            if output_str.startswith(indicator):
                return indicator
        return "other/unclassified"

    failures = failures.copy()
    failures["failure_reason"] = failures.apply(classify_failure, axis=1)

    print("\nFailure reason breakdown:")
    print(failures["failure_reason"].value_counts())

    print("\nFailure reason by tool:")
    print(failures.groupby(["tool_name", "failure_reason"]).size())
else:
    print("\nNo failed tool calls found in this dataset.")


# ===== LATENCY (P50 / P99) =====
n_invocations = len(llm_df)
p50 = llm_df["latency_seconds"].quantile(0.50)
p99 = llm_df["latency_seconds"].quantile(0.99)
meets_p50 = p50 < P50_THRESHOLD
meets_p99 = p99 < P99_THRESHOLD

print(f"\nP50 latency: {p50:.3f}s (threshold: <{P50_THRESHOLD}s) — {'PASS' if meets_p50 else 'FAIL'}")
print(f"P99 latency: {p99:.3f}s (threshold: <{P99_THRESHOLD}s) — {'PASS' if meets_p99 else 'FAIL'}")

overall_status = "meeting" if (meets_p50 and meets_p99) else "not meeting"
print(f"\nSentence: Across {n_invocations} agent invocations captured via LangSmith, "
      f"the median (P50) latency was {p50:.3f}s and the 99th percentile (P99) was {p99:.3f}s "
      f"– {overall_status} the pre-specified performance thresholds.")


# ===== TOKEN CONSUMPTION =====
mean_tokens = llm_df["total_tokens"].mean()
sd_tokens = llm_df["total_tokens"].std()
print(f"\nMean token consumption per invocation: {mean_tokens:.1f} (SD = {sd_tokens:.1f})")

print("\nAverage latency and tokens by participant:")
print(llm_df.groupby("participant_id")[["latency_seconds", "total_tokens"]].mean())


# ===== EMERGENCY PATHWAY DETECTION =====
emergency_participants = sorted(
    llm_df.loc[llm_df["is_emergency_pathway_run"], "participant_id"].unique()
)
print(f"\nParticipants who triggered the emergency pathway: {emergency_participants}")
print(f"n = {len(emergency_participants)} / {len(KEEP_IDS)}")


# ===== SAVE RESULTS =====
tool_df.to_csv("phase2_dev_task_completion.csv", index=False)
llm_df.to_csv("phase2_latency_tokens.csv", index=False)