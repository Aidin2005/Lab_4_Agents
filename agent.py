"""ReAct-агент на чистом Python: цикл «мысль -> действие -> наблюдение».

Реализация без внешних фреймворков (LangChain, LangGraph).
Поддерживает Groq, Ollama, любой OpenAI-совместимый API и режим Mock LLM.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools import calculator, text_search

SYSTEM_PROMPT = """Ты — ассистент, который решает задачи пошагово в формате ReAct.
Доступные инструменты:
{tools}

Строго соблюдай формат. На каждом шаге пиши:
Thought: рассуждение о том, что делать дальше
Action: имя_инструмента
Action Input: входные данные инструмента одной строкой

После этого ОСТАНОВИСЬ и жди строку Observation с результатом.
Когда информации достаточно, напиши:
Thought: итоговое рассуждение
Final Answer: ответ пользователю

Правила: не придумывай Observation сам; используй только перечисленные инструменты;
все вычисления делай через calculator; факты о компании бери только из text_search."""

ACTION_RE = re.compile(r"Action:\s*([\w\-]+)\s*\n\s*Action Input:\s*(.+)")
FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.S)

TOOLS: Dict[str, Callable[[str], str]] = {
    "calculator": calculator,
    "text_search": text_search,
}

TOOLS_DESC = (
    "- calculator: вычисляет арифметическое выражение\n"
    "- text_search: ищет фрагменты во внутренних документах"
)

DEFAULT_LOG_PATH = Path("agent_log.jsonl")


def parse(text: str) -> Tuple[str, ...]:
  """Парсинг вывода модели.

  Возвращает:
  - ('final', answer)
  - ('action', name, arg)
  - ('error', message)
  """
  # Отрезаем всё после возможного добавленного моделью Observation:
  text = text.split("Observation:")[0].strip()

  m_act = ACTION_RE.search(text)
  m_fin = FINAL_RE.search(text)

  # Проверяем, что появилось раньше — Final Answer или Action
  if m_fin and (not m_act or m_fin.start() < m_act.start()):
    return ("final", m_fin.group(1).strip())
  if m_act:
    return ("action", m_act.group(1).strip(), m_act.group(2).strip())
  return (
      "error",
      "Нет 'Action:'+'Action Input:' и нет 'Final Answer:'. Соблюдай формат.",
  )


def log_event(record: Dict[str, Any], log_file: Path = DEFAULT_LOG_PATH) -> None:
  """Запись одного события в формате JSON Lines."""
  with open(log_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_mock_llm(script: List[str]) -> Callable[[List[Dict[str, str]]], Tuple[str, Dict[str, int]]]:
  """Создание подставной модели для детерминированного тестирования без сети."""
  replies = iter(script)

  def _mock(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
    reply = next(replies)
    usage = {
        "prompt_tokens": sum(len(m["content"].split()) for m in messages) * 2,
        "completion_tokens": len(reply.split()) * 2,
        "total_tokens": (sum(len(m["content"].split()) for m in messages) + len(reply.split())) * 2,
    }
    return reply, usage

  return _mock


def get_default_llm() -> Callable[[List[Dict[str, str]]], Tuple[str, Dict[str, int]]]:
  """Создание функции вызова LLM через OpenAI-совместимый клиент (Groq, Ollama, etc.)."""
  from openai import OpenAI

  base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
  api_key = os.getenv("GROQ_API_KEY", "")
  model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

  client = OpenAI(base_url=base_url, api_key=api_key or "no-key")

  def _call(messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        stop=["Observation:"],
    )
    content = (resp.choices[0].message.content or "").strip()
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
        "total_tokens": getattr(resp.usage, "total_tokens", 0) if resp.usage else 0,
    }
    return content, usage

  return _call


def run_agent(
    question: str,
    max_steps: int = 8,
    llm: Optional[Callable[[List[Dict[str, str]]], Any]] = None,
    log_file: Path = DEFAULT_LOG_PATH,
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> str:
  """Основной цикл ReAct-агента.

  Args:
      question: вопрос пользователя
      max_steps: максимальное число шагов рассуждения
      llm: функция вызова LLM (по умолчанию Groq/OpenAI client)
      log_file: путь к файлу журнала JSON Lines
      run_id: уникальный идентификатор запуска
      verbose: выводить ли шаги в консоль
  """
  if run_id is None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

  if llm is None:
    llm = get_default_llm()

  system_prompt = SYSTEM_PROMPT.format(tools=TOOLS_DESC)
  messages: List[Dict[str, str]] = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": f"Question: {question}"},
  ]

  # Запись старта в журнал
  start_record = {
      "ts": datetime.now(timezone.utc).isoformat(),
      "run_id": run_id,
      "step": 0,
      "event": "start",
      "question": question,
  }
  log_event(start_record, log_file)

  if verbose:
    print(f"\n{'='*60}\nВопрос: {question}\nRun ID: {run_id}\n{'='*60}")

  for step in range(1, max_steps + 1):
    t0 = time.perf_counter()
    try:
      llm_res = llm(messages)
      if isinstance(llm_res, tuple):
        text, usage = llm_res
      else:
        text, usage = llm_res, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    except Exception as e:
      latency = round(time.perf_counter() - t0, 3)
      err_record = {
          "ts": datetime.now(timezone.utc).isoformat(),
          "run_id": run_id,
          "step": step,
          "event": "api_error",
          "error": str(e),
          "latency_s": latency,
      }
      log_event(err_record, log_file)
      if verbose:
        print(f"[Шаг {step}] Ошибка вызова LLM: {e}")
      return f"Ошибка обращения к LLM: {e}"

    latency = round(time.perf_counter() - t0, 3)
    text = (text or "").strip()

    if verbose:
      print(f"\n--- Шаг {step} (задержка: {latency}c) ---")
      print(text)

    parsed = parse(text)

    # 1. Финальный ответ
    if parsed[0] == "final":
      final_answer = parsed[1]
      step_record = {
          "ts": datetime.now(timezone.utc).isoformat(),
          "run_id": run_id,
          "step": step,
          "event": "final",
          "llm_output": text,
          "final_answer": final_answer,
          "latency_s": latency,
          "usage": usage,
      }
      log_event(step_record, log_file)
      if verbose:
        print(f"\n[Ответ агента]: {final_answer}")
      return final_answer

    # 2. Действие (Action)
    elif parsed[0] == "action":
      action_name, action_input = parsed[1], parsed[2]
      tool_fn = TOOLS.get(action_name)

      if tool_fn is not None:
        try:
          observation = tool_fn(action_input)
          tool_ok = not str(observation).startswith("Ошибка:")
        except Exception as e:
          observation = f"Ошибка выполнения инструмента: {e}"
          tool_ok = False
      else:
        avail = ", ".join(TOOLS.keys())
        observation = f"Ошибка: неизвестный инструмент '{action_name}'. Доступные инструменты: {avail}."
        tool_ok = False

      step_record = {
          "ts": datetime.now(timezone.utc).isoformat(),
          "run_id": run_id,
          "step": step,
          "event": "step",
          "llm_output": text,
          "action": action_name,
          "action_input": action_input,
          "observation": observation,
          "tool_ok": tool_ok,
          "latency_s": latency,
          "usage": usage,
      }
      log_event(step_record, log_file)

      if verbose:
        print(f"Observation: {observation}")

      messages.append({"role": "assistant", "content": text})
      messages.append({"role": "user", "content": f"Observation: {observation}"})

    # 3. Ошибка формата ответа
    elif parsed[0] == "error":
      error_msg = parsed[1]
      observation = f"Ошибка формата: {error_msg}"
      tool_ok = False

      step_record = {
          "ts": datetime.now(timezone.utc).isoformat(),
          "run_id": run_id,
          "step": step,
          "event": "step",
          "llm_output": text,
          "action": None,
          "action_input": None,
          "observation": observation,
          "tool_ok": tool_ok,
          "latency_s": latency,
          "usage": usage,
      }
      log_event(step_record, log_file)

      if verbose:
        print(f"Observation: {observation}")

      messages.append({"role": "assistant", "content": text})
      messages.append({"role": "user", "content": f"Observation: {observation}"})

  # Превышение лимита шагов
  max_record = {
      "ts": datetime.now(timezone.utc).isoformat(),
      "run_id": run_id,
      "step": max_steps,
      "event": "max_steps_reached",
  }
  log_event(max_record, log_file)

  msg = "Не удалось получить ответ за отведённое число шагов."
  if verbose:
    print(f"\n[Завершение]: {msg}")
  return msg


if __name__ == "__main__":
  import sys
  query = sys.argv[1] if len(sys.argv) > 1 else "Сколько дней ежегодного отпуска полагается сотруднику?"
  run_agent(query)
