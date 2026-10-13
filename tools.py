"""Инструменты для ReAct-агента: безопасный калькулятор и поиск по документам."""

import ast
import operator
from pathlib import Path
import re
from typing import List, Set, Tuple

DOCS_DIR = Path("docs")

# Разрешённые математические операции
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculator(expression: str) -> str:
  """Безопасный калькулятор: вычисляет арифметические выражения без eval().

  Поддерживает числа и операторы +, -, *, /, %, **.
  Возвращает результат строкой или сообщение об ошибке.
  """

  def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
      return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
      left, right = _eval(node.left), _eval(node.right)
      if isinstance(node.op, ast.Pow) and abs(right) > 100:
        raise ValueError("слишком большая степень")
      return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
      return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("допустимы только числа и операторы + - * / % **")

  try:
    expr = str(expression).strip().strip("`'\"").replace("^", "**")
    result = _eval(ast.parse(expr, mode="eval").body)
    return str(round(result, 6)) if isinstance(result, float) else str(result)
  except ZeroDivisionError:
    return "Ошибка: деление на ноль"
  except Exception as e:
    return f"Ошибка: {e}"


def _stem(word: str) -> str:
  """Нормализация слова для сопоставления падежей и окончаний:
  Для слов от 6 символов берём основу первых 6 букв (команд-ировка, суточн-ые, отпуск-ные).
  Для более коротких слов отсекаем гласные окончания при длине корня >= 3.
  """
  word = word.lower()
  if len(word) >= 6:
    return word[:6]
  for ending in ("а", "я", "е", "и", "ы", "у", "ю", "о", "ь"):
    if word.endswith(ending) and len(word) > 4:
      return word[:-1]
  return word


def _tokenize(text: str) -> List[str]:
  """Извлечение информативных слов длиннее 2 символов."""
  return [w.lower() for w in re.findall(r"\w+", text) if len(w) > 2]


def text_search(query: str, top_k: int = 3) -> str:
  """Поиск фрагментов во внутренних документах папки docs/.

  Оценивает каждый абзац по совпадению токенов и стемов запроса.
  Возвращает top_k наиболее релевантных абзацев с указанием источника.
  """
  q_words = list(dict.fromkeys(_tokenize(query)))
  if not q_words:
    return "Ошибка: пустой поисковый запрос."

  if not DOCS_DIR.exists():
    return "Ошибка: папка docs не найдена."

  hits: List[Tuple[int, str, int, str]] = []

  for path in sorted(DOCS_DIR.glob("*.txt")):
    try:
      text = path.read_text(encoding="utf-8")
    except Exception as e:
      continue

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for i, para in enumerate(paragraphs):
      para_words = _tokenize(para)
      if not para_words:
        continue

      para_exact = set(para_words)
      para_stems = {_stem(w) for w in para_words}

      score = 0
      for qw in q_words:
        if qw in para_exact or _stem(qw) in para_stems:
          score += 1

      if score > 0:
        hits.append((score, path.name, i + 1, para))

  if not hits:
    return (
        "По вашему запросу ничего не найдено во внутренних документах. "
        "Попробуйте переформулировать ключевые слова."
    )

  # Сортировка по убыванию score, затем по имени файла
  hits.sort(key=lambda x: (-x[0], x[1], x[2]))

  results = []
  for score, filename, para_idx, para in hits[:top_k]:
    clean_text = " ".join(para.split())
    if len(clean_text) > 400:
      clean_text = clean_text[:397] + "..."
    results.append(f"[{filename}, абзац {para_idx}] {clean_text}")

  return "\n\n".join(results)
