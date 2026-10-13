"""Анализ журнала agent_log.jsonl с помощью pandas: расчёт ключевых метрик ReAct-агента."""

import json
from pathlib import Path
import pandas as pd


def analyze(log_file: str = "agent_log.jsonl") -> pd.DataFrame:
  path = Path(log_file)
  if not path.exists():
    print(f"Файл журнала '{log_file}' не найден.")
    return pd.DataFrame()

  df = pd.read_json(path, lines=True)
  if df.empty:
    print("Журнал пуст.")
    return df

  print("=" * 60)
  print("АНАЛИЗ МЕТРИК РАБОТЫ REACT-АГЕНТА")
  print("=" * 60)

  # 1. Запуски и шаги
  run_ids = df["run_id"].unique()
  total_runs = len(run_ids)
  step_rows = df[df["event"].isin(["step", "final"])]
  steps_per_run = step_rows.groupby("run_id").size()

  avg_steps = steps_per_run.mean() if not steps_per_run.empty else 0.0
  print(f"Всего запусков (run_id): {total_runs}")
  print(f"Среднее число шагов на вопрос: {avg_steps:.2f}")

  # 2. Задержка LLM
  avg_latency = df["latency_s"].dropna().mean()
  print(f"Средняя задержка одного шага LLM: {avg_latency:.3f} с")

  # 3. Доля успешных вызовов инструментов
  action_rows = df[df["action"].notna() & (df["action"] != "")]
  if not action_rows.empty:
    tool_ok_count = (action_rows["tool_ok"] == True).sum()
    tool_total = len(action_rows)
    tool_ok_rate = (tool_ok_count / tool_total) * 100
    print(f"Вызовов инструментов: {tool_total}")
    print(f"Успешных вызовов инструментов: {tool_ok_count} ({tool_ok_rate:.1f}%)")
  else:
    tool_ok_rate = 100.0
    print("Инструменты не вызывались.")

  # 4. Токены
  def extract_token_field(usage, key):
    if isinstance(usage, dict):
      return usage.get(key, 0)
    return 0

  prompt_tokens = df["usage"].apply(lambda u: extract_token_field(u, "prompt_tokens")).sum()
  comp_tokens = df["usage"].apply(lambda u: extract_token_field(u, "completion_tokens")).sum()
  total_tokens = df["usage"].apply(lambda u: extract_token_field(u, "total_tokens")).sum()

  print(f"Суммарное число токенов:")
  print(f"  - Prompt tokens:     {prompt_tokens}")
  print(f"  - Completion tokens: {comp_tokens}")
  print(f"  - Total tokens:      {total_tokens}")

  # 5. Сводная таблица по запускам
  print("\n" + "=" * 60)
  print("СВОДКА ПО КАЖДОМУ ЗАПУСКУ:")
  print("=" * 60)

  summary_rows = []
  for rid in run_ids:
    sub = df[df["run_id"] == rid]
    start_row = sub[sub["event"] == "start"]
    question = start_row["question"].values[0] if not start_row.empty else "N/A"

    final_row = sub[sub["event"] == "final"]
    if not final_row.empty:
      final_answer = final_row["final_answer"].values[0]
      status = "SUCCESS"
    elif not sub[sub["event"] == "max_steps_reached"].empty:
      final_answer = "Лимит шагов исчерпан"
      status = "MAX_STEPS"
    else:
      final_answer = "Прервано / Ошибка"
      status = "ERROR"

    run_steps = len(sub[sub["event"].isin(["step", "final"])])
    run_latency = sub["latency_s"].sum()
    run_tokens = sub["usage"].apply(lambda u: extract_token_field(u, "total_tokens")).sum()

    summary_rows.append({
        "run_id": rid,
        "question": question,
        "status": status,
        "steps": run_steps,
        "latency_s": round(run_latency, 2),
        "total_tokens": run_tokens,
        "final_answer": final_answer,
    })

  summary_df = pd.DataFrame(summary_rows)
  print(summary_df.to_string(index=False))
  return summary_df


if __name__ == "__main__":
  analyze()
