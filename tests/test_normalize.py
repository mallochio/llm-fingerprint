"""Unit tests for answer normalization and validity taxonomy."""

import json
from pathlib import Path
from fingerprint.normalize import basic_clean, zh_number, normalize_answer


def test_basic_clean():
    raw = '  «"Hello, World!"»  '
    cleaned = basic_clean(raw)
    assert cleaned == "Hello World"


def test_zh_number():
    assert zh_number("七") == 7
    assert zh_number("十") == 10
    assert zh_number("十七") == 17
    assert zh_number("四十二") == 42
    assert zh_number("九十九") == 99


def test_normalize_integer():
    task = {"normalize_as": "integer", "answer_space": "1-100", "category": "number"}

    res1 = normalize_answer("42", lang="en", task=task)
    assert res1.normalized == "42"
    assert res1.answer_class == "valid"

    res_ar = normalize_answer("٧", lang="ar", task=task)
    assert res_ar.normalized == "7"
    assert res_ar.answer_class == "valid"

    res_zh = normalize_answer("四十二", lang="zh", task=task)
    assert res_zh.normalized == "42"
    assert res_zh.answer_class == "valid"

    res_out = normalize_answer("150", lang="en", task=task)
    assert res_out.normalized == "150"
    assert res_out.answer_class == "invalid"


def test_normalize_binary():
    task = {"normalize_as": "binary", "answer_space": "heads|tails", "category": "binary"}

    res_en = normalize_answer("heads", lang="en", task=task)
    assert res_en.normalized == "h"
    assert res_en.answer_class == "valid"

    res_ru = normalize_answer("орёл", lang="ru", task=task)
    assert res_ru.normalized == "h"
    assert res_ru.answer_class == "valid"

    res_zh = normalize_answer("反面", lang="zh", task=task)
    assert res_zh.normalized == "t"
    assert res_zh.answer_class == "valid"


def test_normalize_refusal():
    res = normalize_answer("I'm sorry, as an AI model I cannot fulfill this.", lang="en")
    assert res.answer_class == "refusal"
    assert res.normalized is None


def test_color_lexicon_mapping():
    color_file = Path("batteries/v1/color-lexicon.json")
    with open(color_file, "r", encoding="utf-8") as f:
        color_lex = json.load(f)

    task = {"normalize_as": "word", "category": "color"}

    res_ru = normalize_answer("красный", lang="ru", task=task, color_lexicon=color_lex)
    assert res_ru.normalized == "красный"
    assert res_ru.color_canon == "red"

    res_zh = normalize_answer("天蓝色", lang="zh", task=task, color_lexicon=color_lex)
    assert res_zh.normalized == "天蓝色"
    assert res_zh.color_canon == "azure"
