#!/usr/bin/env python3
"""
Просмотр результатов валидации бенчмарка
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

def view_results(results_file: Path):
    """Показать результаты валидации"""
    if not results_file.exists():
        print(f"❌ Файл не найден: {results_file}")
        return
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = data.get('stats', {})
    results = data.get('results', [])
    
    print("\n" + "="*80)
    print("📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ ПАЙПЛАЙНА")
    print("="*80)
    print(f"\n📁 Файл: {results_file.name}")
    print(f"🕐 Время: {data.get('timestamp', 'unknown')}")
    print(f"📋 Бенчмарк: {data.get('benchmark_file', 'unknown')}")
    
    print("\n" + "-"*80)
    print("📈 ОБЩАЯ СТАТИСТИКА")
    print("-"*80)
    print(f"Всего примеров: {stats.get('total', 0)}")
    print(f"\n✅ Сработало (пайплайн успешен):")
    print(f"   {stats.get('worked', 0)} / {stats.get('total', 0)} ({stats.get('success_rate', {}).get('worked', 0):.1f}%)")
    print(f"\n💾 Выполнилось (SQL выполнен):")
    print(f"   {stats.get('executed', 0)} / {stats.get('total', 0)} ({stats.get('success_rate', {}).get('executed', 0):.1f}%)")
    print(f"\n🎯 Похожий результат:")
    print(f"   {stats.get('similar_result', 0)} / {stats.get('total', 0)} ({stats.get('success_rate', {}).get('similar_result', 0):.1f}%)")
    print(f"\n❌ Ошибки: {stats.get('errors', 0)}")
    
    print("\n" + "-"*80)
    print("⏱️  ПРОИЗВОДИТЕЛЬНОСТЬ")
    print("-"*80)
    print(f"Общее время: {stats.get('total_time', 0)/60:.1f} мин ({stats.get('total_time', 0):.1f} сек)")
    print(f"Среднее время на пример: {stats.get('avg_time_per_example', 0):.2f} сек")
    
    # Детальная статистика по ошибкам
    errors = [r for r in results if r.get('error')]
    if errors:
        print("\n" + "-"*80)
        print("❌ ОШИБКИ (первые 10)")
        print("-"*80)
        for i, err in enumerate(errors[:10], 1):
            print(f"\n{i}. Пример #{err.get('index', '?')}")
            print(f"   Вопрос: {err.get('question', '')[:80]}...")
            print(f"   Ошибка: {err.get('error', 'Unknown')[:100]}")
    
    # Примеры с похожими результатами
    similar = [r for r in results if r.get('similar_result')]
    if similar:
        print("\n" + "-"*80)
        print("✅ ПРИМЕРЫ С ПОХОЖИМИ РЕЗУЛЬТАТАМИ (первые 5)")
        print("-"*80)
        for i, sim in enumerate(similar[:5], 1):
            print(f"\n{i}. Пример #{sim.get('index', '?')}")
            print(f"   Вопрос: {sim.get('question', '')[:80]}...")
            comp = sim.get('comparison', {})
            print(f"   Схожесть: {comp.get('similarity_score', 0):.2%}")
            print(f"   Строки: {comp.get('generated_rows', 0)} (ожидалось: {comp.get('expected_rows', 0)})")
    
    # Примеры с непохожими результатами
    not_similar = [r for r in results if r.get('executed') and not r.get('similar_result')]
    if not_similar:
        print("\n" + "-"*80)
        print("⚠️  ПРИМЕРЫ С НЕПОХОЖИМИ РЕЗУЛЬТАТАМИ (первые 5)")
        print("-"*80)
        for i, ns in enumerate(not_similar[:5], 1):
            print(f"\n{i}. Пример #{ns.get('index', '?')}")
            print(f"   Вопрос: {ns.get('question', '')[:80]}...")
            comp = ns.get('comparison', {})
            print(f"   Схожесть: {comp.get('similarity_score', 0):.2%}")
            print(f"   Строки совпадают: {comp.get('rows_match', False)}")
            print(f"   Колонки совпадают: {comp.get('columns_match', False)}")
    
    print("\n" + "="*80)
    print(f"💾 Полные результаты в: {results_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Просмотр результатов валидации')
    parser.add_argument('file', nargs='?', type=str, 
                       help='Путь к файлу результатов (по умолчанию: последний файл)')
    
    args = parser.parse_args()
    
    results_dir = Path(__file__).parent / "results"
    
    if args.file:
        results_file = Path(args.file)
        if not results_file.is_absolute():
            results_file = results_dir / args.file
    else:
        # Найти последний файл результатов
        result_files = sorted(results_dir.glob("benchmark_validation_*.json"), reverse=True)
        if result_files:
            results_file = result_files[0]
            print(f"📂 Используется последний файл: {results_file.name}\n")
        else:
            print("❌ Файлы результатов не найдены в tests/results/")
            sys.exit(1)
    
    view_results(results_file)

