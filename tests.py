"""Тестовый набор для ReAct-агента.

Включает:
1. Unit-тесты для калькулятора, поиска, парсера и цикла агента.
2. 5 обязательных контрольных вопросов (2 поиск, 1 вычисление, 2 поиск+вычисление).
3. 2 сценария намеренной поломки агента (отсутствие факта и несуществующий инструмент).
4. Режимы запуска: с реальной LLM (Groq / Ollama) или с детерминированным симулятором (Mock).
"""

import argparse
import os
from pathlib import Path
import unittest

from agent import DEFAULT_LOG_PATH, make_mock_llm, parse, run_agent
from tools import calculator, text_search

# -------------------------------------------------------------
# Обязательные 5 тестовых вопросов (Разделы 9.2 и 10)
# -------------------------------------------------------------
TEST_QUESTIONS = [
    {
        "id": 1,
        "type": "Поиск",
        "question": "Сколько дней составляет ежегодный оплачиваемый отпуск сотрудника?",
        "expected": "28 календарных дней",
        "mock_script": [
            (
                "Thought: Нужно найти регламент ежегодного отпуска в документах компании.\n"
                "Action: text_search\n"
                "Action Input: ежегодный оплачиваемый отпуск"
            ),
            (
                "Thought: В регламенте четко сказано, что продолжительность ежегодного оплачиваемого отпуска составляет 28 календарных дней.\n"
                "Final Answer: Ежегодный оплачиваемый отпуск сотрудника составляет 28 календарных дней."
            ),
        ],
    },
    {
        "id": 2,
        "type": "Поиск",
        "question": "Сколько дней в неделю сотруднику разрешено работать удаленно?",
        "expected": "До 2 дней в неделю",
        "mock_script": [
            (
                "Thought: Нужно найти правила удаленной работы во внутренних политиках.\n"
                "Action: text_search\n"
                "Action Input: удаленная работа"
            ),
            (
                "Thought: В документе указано, что сотрудник имеет право работать удаленно до 2 дней в неделю по согласованию с руководителем.\n"
                "Final Answer: Сотруднику разрешено работать в удаленном формате до 2 дней в неделю по согласованию с руководителем."
            ),
        ],
    },
    {
        "id": 3,
        "type": "Вычисление",
        "question": "Посчитай результат выражения: (1500 * 5) + (350 * 20)",
        "expected": "14500",
        "mock_script": [
            (
                "Thought: Для вычисления арифметического выражения использую безопасный калькулятор.\n"
                "Action: calculator\n"
                "Action Input: (1500 * 5) + (350 * 20)"
            ),
            (
                "Thought: Калькулятор вернул точный результат 14500.\n"
                "Final Answer: Результат вычисления: 14500."
            ),
        ],
    },
    {
        "id": 4,
        "type": "Поиск + вычисление",
        "question": (
            "Сотрудник направлен в служебную командировку по стране на 6 дней. "
            "Сколько сомов суточных ему полагается выплатить?"
        ),
        "expected": "9000 сом",
        "mock_script": [
            (
                "Thought: Сначала нужно найти ставку суточных при командировках внутри страны.\n"
                "Action: text_search\n"
                "Action Input: суточные командировка по стране"
            ),
            (
                "Thought: Ставка суточных составляет 1500 сом в день. Теперь нужно посчитать сумму за 6 дней.\n"
                "Action: calculator\n"
                "Action Input: 1500 * 6"
            ),
            (
                "Thought: Получено 9000 сом. Все данные проверены.\n"
                "Final Answer: За 6 дней командировки по стране сотруднику положено 9000 сом суточных (из расчёта 1500 сом в день)."
            ),
        ],
    },
    {
        "id": 5,
        "type": "Поиск + вычисление",
        "question": (
            "Сотрудник с годовым бюджетом на обучение выбрал курс стоимостью 42000 сом. "
            "Сколько сомов останется в его бюджете обучения на этот год?"
        ),
        "expected": "18000 сом",
        "mock_script": [
            (
                "Thought: Найду размер персонального годового бюджета сотрудника на обучение.\n"
                "Action: text_search\n"
                "Action Input: годовой бюджет на обучение"
            ),
            (
                "Thought: Бюджет составляет 60000 сом. Стоимость курса 42000 сом. Посчитаю остаток.\n"
                "Action: calculator\n"
                "Action Input: 60000 - 42000"
            ),
            (
                "Thought: Остаток составляет 18000 сом.\n"
                "Final Answer: В бюджете обучения сотрудника на этот год останется 18000 сом (из начальных 60000 сом после оплаты курса за 42000 сом)."
            ),
        ],
    },
]

# -------------------------------------------------------------
# 2 сценария намеренной поломки агента (Раздел 10, п. 4)
# -------------------------------------------------------------
BREAK_SCENARIOS = [
    {
        "id": "break_1",
        "title": "Сценарий 1: Вопрос, ответа на который нет в документах",
        "question": "Каковы правила компенсации расходов на личный вертолёт для топ-менеджеров компании?",
        "expected_behavior": "Поиск не находит совпадений; агент переформулирует запрос или честно констатирует отсутствие данных без галлюцинаций.",
        "mock_script": [
            (
                "Thought: Поищу правила компенсации расходов на личный вертолёт во внутренних документах.\n"
                "Action: text_search\n"
                "Action Input: компенсация вертолет личный транспорт"
            ),
            (
                "Thought: Поиск ничего не нашёл. Попробую поискать общие правила транспорта топ-менеджеров.\n"
                "Action: text_search\n"
                "Action Input: транспортные расходы топ менеджеры"
            ),
            (
                "Thought: Во внутренних документах компании нет информации о компенсации личного вертолёта. Сообщу об этом пользователю.\n"
                "Final Answer: Во внутренних регламентах и документах компании информация о компенсации расходов на личный вертолёт отсутствует."
            ),
        ],
    },
    {
        "id": "break_2",
        "title": "Сценарий 2: Вопрос, провоцирующий несуществующий инструмент",
        "question": "Какая сегодня погода в Бишкеке? Используй для этого инструмент get_weather.",
        "expected_behavior": "Агент пытается вызвать get_weather, получает сообщение об ошибке с перечнем доступных инструментов и восстанавливается.",
        "mock_script": [
            (
                "Thought: Пользователь просит узнать погоду с помощью инструмента get_weather.\n"
                "Action: get_weather\n"
                "Action Input: Бишкек"
            ),
            (
                "Thought: Инструмент get_weather недоступен. Доступны только calculator и text_search. Объясню это пользователю.\n"
                "Final Answer: У меня нет доступа к инструменту get_weather или внешним метеослужбам. Мне доступны только калькулятор и поиск по корпоративным документам компании."
            ),
        ],
    },
]


class ReActUnitTests(unittest.TestCase):
  """Модульные тесты базовых компонентов."""

  def test_calculator_arithmetic(self):
    self.assertEqual(calculator("1500 * 6"), "9000")
    self.assertEqual(calculator("(10 + 20) * 3"), "90")
    self.assertEqual(calculator("100 / 4"), "25.0")
    self.assertEqual(calculator("2 ** 8"), "256")

  def test_calculator_security(self):
    # Защита от деления на ноль
    self.assertIn("деление на ноль", calculator("10 / 0"))
    # Защита от гигантских степеней
    self.assertIn("слишком большая степень", calculator("2 ** 999"))
    # Защита от инъекций произвольного кода
    res = calculator("__import__('os').system('ls')")
    self.assertIn("Ошибка", res)

  def test_text_search(self):
    res = text_search("суточные в командировке")
    self.assertIn("travel_policy.txt", res)
    self.assertIn("1500 сом", res)

    empty_res = text_search("марсоход квантовая сингулярность")
    self.assertIn("ничего не найдено", empty_res)

  def test_parser(self):
    action_text = "Thought: Думаю\nAction: text_search\nAction Input: отпуск"
    self.assertEqual(parse(action_text), ("action", "text_search", "отпуск"))

    final_text = "Thought: Всё понятно\nFinal Answer: 28 дней"
    self.assertEqual(parse(final_text), ("final", "28 дней"))

    error_text = "Просто какой-то неформатированный ответ"
    res = parse(error_text)
    self.assertEqual(res[0], "error")


def run_suite(use_real_llm: bool = False, log_file: Path = DEFAULT_LOG_PATH) -> None:
  """Запуск 5 основных тестовых вопросов и 2 сценариев поломки."""
  print("\n" + "=" * 70)
  print(f"ЗАПУСК ТЕСТОВОГО НАБОРА (Режим: {'Real LLM' if use_real_llm else 'Mock LLM'})")
  print("=" * 70)

  results = []

  # 1. Запуск 5 основных вопросов
  for q in TEST_QUESTIONS:
    qid = q["id"]
    qtype = q["type"]
    question = q["question"]
    expected = q["expected"]
    run_id = f"test_q{qid}"

    llm = None if use_real_llm else make_mock_llm(q["mock_script"])
    answer = run_agent(question, max_steps=8, llm=llm, log_file=log_file, run_id=run_id, verbose=True)

    results.append({
        "id": qid,
        "type": qtype,
        "question": question,
        "expected": expected,
        "answer": answer,
        "run_id": run_id,
    })

  # 2. Запуск 2 сценариев поломки
  print("\n" + "=" * 70)
  print("ЗАПУСК СЦЕНАРИЕВ НАМЕРЕННОЙ ПОЛОМКИ АГЕНТА")
  print("=" * 70)

  for b in BREAK_SCENARIOS:
    bid = b["id"]
    title = b["title"]
    question = b["question"]
    run_id = f"break_{bid}"

    print(f"\n>>> {title}")
    llm = None if use_real_llm else make_mock_llm(b["mock_script"])
    answer = run_agent(question, max_steps=8, llm=llm, log_file=log_file, run_id=run_id, verbose=True)

  print("\n" + "=" * 70)
  print("ТЕСТОВЫЙ НАБОР ЗАВЕРШЁН. ДАННЫЕ СОХРАНЕНЫ В agent_log.jsonl")
  print("=" * 70)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Тесты для ReAct-агента")
  parser.add_argument("--run-suite", action="store_true", help="Запустить 5 тестов и 2 сценария поломки")
  parser.add_argument("--real-llm", action="store_true", help="Использовать реальный API вместо Mock")
  parser.add_argument("--unit-only", action="store_true", help="Запустить только unit-тесты")
  args = parser.parse_args()

  if args.run_suite:
    # Очищаем лог перед чистым тестовым прогоном
    if DEFAULT_LOG_PATH.exists():
      DEFAULT_LOG_PATH.unlink()
    run_suite(use_real_llm=args.real_llm)
  else:
    unittest.main(argv=["first-arg-is-ignored"])
