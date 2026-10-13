"""Продвинутая часть: Реализация паттернов Reflexion и Human-in-the-Loop.

1. Паттерн Reflexion (Shinn et al., NeurIPS 2023):
   - Actor: выполняет решение задачи через ReAct-цикл.
   - Evaluator: проверяет качество ответа (например, наличие вычисленного числа).
   - Self-Reflection: формулирует вербальную критику при ошибке.
   - Память: передача рефлексии в контекст для повторной попытки.

2. Паттерн Human-in-the-Loop:
   - Добавление инструмента `ask_human` для уточнения неоднозначных данных у человека.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Tuple

from agent import DEFAULT_LOG_PATH, log_event, make_mock_llm, parse, run_agent, TOOLS, TOOLS_DESC
from tools import calculator, text_search

# =====================================================================
# 1. Паттерн Reflexion (Вариант C)
# =====================================================================

def evaluate_answer(question: str, answer: str, expected_type: str = "number") -> Tuple[bool, str]:
  """Оценщик (Evaluator) качества ответа агента.

  Проверяет, соответствует ли ответ критериям (например, наличие конкретного числа).
  """
  if "не удалось получить ответ" in answer.lower():
    return False, "Ответ не был сформирован за отведённые шаги."

  if expected_type == "number":
    # Проверка наличия хотя бы одного числа в ответе
    numbers = re.findall(r"\d+", answer)
    if not numbers:
      return False, "Ответ не содержит конкретного вычисленного числового значения, хотя вопрос требовал расчёта."

  return True, "Ответ успешно прошёл валидацию."


def run_reflexion_agent(
    question: str,
    expected_type: str = "number",
    max_trials: int = 2,
    log_file: Path = DEFAULT_LOG_PATH,
    verbose: bool = True,
) -> str:
  """Запуск агента с рефлексией (Reflexion Pattern).

  Если ответ не проходит валидатор Evaluator, агент генерирует рефлексию
  и делает повторную попытку, учитывая её в контексте.
  """
  memory_reflections: List[str] = []

  # Симуляция сценария для демонстрации:
  # Попытка 1: агент отвечает без вызова калькулятора (не называет сумму)
  attempt_1_script = [
      (
          "Thought: Ищу суточные по стране.\n"
          "Action: text_search\n"
          "Action Input: суточные командировка\n"
      ),
      (
          "Thought: Я нашёл ставку, напишу пользователю общими словами.\n"
          "Final Answer: Суточные выплачиваются по установленной ставке за каждый день поездки.\n"
      ),
  ]

  # Попытка 2: с учётом рефлексии агент использует калькулятор и возвращает точное число
  attempt_2_script = [
      (
          "Thought: С учётом предыдущей ошибки и рефлексии мне обязательно нужно найти ставку и посчитать число через calculator.\n"
          "Action: text_search\n"
          "Action Input: суточные командировка по стране\n"
      ),
      (
          "Thought: Ставка 1500 сом. За 5 дней посчитаю через calculator.\n"
          "Action: calculator\n"
          "Action Input: 1500 * 5\n"
      ),
      (
          "Thought: Получено число 7500. Включаю точное число в ответ.\n"
          "Final Answer: За 5 дней командировки сотруднику полагается 7500 сом суточных.\n"
      ),
  ]

  scripts = [attempt_1_script, attempt_2_script]

  for trial in range(1, max_trials + 1):
    trial_run_id = f"reflexion_trial_{trial}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    if verbose:
      print(f"\n{'#'*65}")
      print(f"ПОПЫТКА {trial} (TRIAL {trial}/{max_trials})")
      if memory_reflections:
        print(f"ПАМЯТЬ РЕФЛЕКСИЙ: {memory_reflections[-1]}")
      print(f"{'#'*65}")

    # Формируем уточнённый вопрос с рефлексией, если она есть
    augmented_question = question
    if memory_reflections:
      augmented_question += f"\n[Предыдущая рефлексия об ошибке: {memory_reflections[-1]}]"

    mock_llm = make_mock_llm(scripts[trial - 1])
    answer = run_agent(
        augmented_question,
        max_steps=6,
        llm=mock_llm,
        log_file=log_file,
        run_id=trial_run_id,
        verbose=verbose,
    )

    # Evaluator: проверка ответа
    is_valid, eval_msg = evaluate_answer(question, answer, expected_type=expected_type)

    eval_record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": trial_run_id,
        "event": "reflexion_eval",
        "trial": trial,
        "is_valid": is_valid,
        "eval_feedback": eval_msg,
    }
    log_event(eval_record, log_file)

    if is_valid:
      if verbose:
        print(f"\n[Evaluator]: Валидация пройдена успешно на попытке {trial}! ({eval_msg})")
      return answer
    else:
      # Генератор саморефлексии
      reflection = (
          f"В попытке {trial} была допущена ошибка: {eval_msg}. "
          f"В следующей попытке необходимо обязательно использовать инструмент calculator "
          f"и указать точный числовой результат в Final Answer."
      )
      memory_reflections.append(reflection)
      if verbose:
        print(f"\n[Evaluator]: Ошибка валидации! {eval_msg}")
        print(f"[Self-Reflection]: {reflection}")

  return answer


# =====================================================================
# 2. Паттерн Human-in-the-Loop (Вариант D)
# =====================================================================

def make_human_in_the_loop_agent(human_input_fn: Optional[Callable[[str], str]] = None):
  """Создание агента с инструментом взаимодействия с человеком."""
  hitl_tools = dict(TOOLS)

  def ask_human(prompt: str) -> str:
    """Запрос уточнения у человека-пользователя."""
    if human_input_fn is not None:
      return human_input_fn(prompt)
    return input(f"\n[Агент просит уточнение]: {prompt}\n[Ваш ответ]: ")

  hitl_tools["ask_human"] = ask_human
  return hitl_tools


def run_hitl_demonstration():
  """Демонстрация работы Human-in-the-loop."""
  print("\n" + "=" * 65)
  print("ДЕМОНСТРАЦИЯ ПАТТЕРНА HUMAN-IN-THE-LOOP")
  print("=" * 65)

  # Имитация ответа человека на вопрос агента
  def simulated_human(prompt: str) -> str:
    print(f"[Агент обращается к человеку]: {prompt}")
    user_reply = "Командировка внутри страны, продолжительность 5 дней."
    print(f"[Человек отвечает]: {user_reply}")
    return user_reply

  hitl_tools = make_human_in_the_loop_agent(simulated_human)

  # Скрипт: агент получает неполный вопрос, спрашивает человека и решает задачу
  hitl_script = [
      (
          "Thought: Вопрос звучит 'Сколько суточных положено?', но не указана страна и число дней. Нужно спросить пользователя.\n"
          "Action: ask_human\n"
          "Action Input: Уточните, пожалуйста: командировка по стране или за рубеж, и сколько дней она продлится?\n"
      ),
      (
          "Thought: Человек ответил, что поездка по стране на 5 дней. Ищу ставку суточных по стране.\n"
          "Action: text_search\n"
          "Action Input: суточные командировка по стране\n"
      ),
      (
          "Thought: Ставка 1500 сом в день. Считаю 1500 * 5 через calculator.\n"
          "Action: calculator\n"
          "Action Input: 1500 * 5\n"
      ),
      (
          "Thought: Итого 7500 сом. Готов дать ответ.\n"
          "Final Answer: С учётом вашего уточнения (командировка по стране на 5 дней) общая сумма суточных составит 7500 сом (из расчёта 1500 сом в день).\n"
      ),
  ]

  mock_llm = make_mock_llm(hitl_script)

  # Временно подменяем TOOLS на hitl_tools
  old_tools = dict(TOOLS)
  try:
    TOOLS.clear()
    TOOLS.update(hitl_tools)
    answer = run_agent(
        "Сколько суточных мне положено?",
        llm=mock_llm,
        run_id="hitl_demo_run",
        verbose=True,
    )
  finally:
    TOOLS.clear()
    TOOLS.update(old_tools)

  print(f"\n[HITL Результат]: {answer}")


if __name__ == "__main__":
  import sys
  if len(sys.argv) > 1 and sys.argv[1] == "--hitl":
    run_hitl_demonstration()
  else:
    print("Запуск Reflexion-агента:")
    run_reflexion_agent(
        "Сколько суточных положено сотруднику за 5 дней командировки по стране?",
        expected_type="number",
    )
