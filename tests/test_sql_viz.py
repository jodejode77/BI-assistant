"""
Тестирование только SQL и типов визуализации
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
import sys

# Добавить родительскую директорию в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_accuracy import SQLAccuracyTester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_benchmark(agent, benchmark_path: str, max_tests: int = None):
    """
    Тестирование на бенчмарке
    
    Args:
        agent: Экземпляр EnhancedSQLAgent
        benchmark_path: Путь к бенчмарку
        max_tests: Максимальное количество тестов
    """
    print(f"\n📋 Загрузка бенчмарка из {benchmark_path}...")
    
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmark = json.load(f)
    
    if max_tests:
        benchmark = benchmark[:max_tests]
    
    print(f"✅ Загружено {len(benchmark)} тестов\n")
    
    tester = SQLAccuracyTester(agent)
    
    results = []
    start_time = time.time()
    
    # Статистика по типам визуализации
    viz_stats = {}
    
    for i, test_case in enumerate(benchmark):
        if (i + 1) % 50 == 0:
            print(f"Прогресс: {i + 1}/{len(benchmark)} тестов...")
        
        result = tester.test_single_query(
            question=test_case['question'],
            expected_sql=test_case['sql'],
            test_id=i + 1
        )
        
        # Добавить информацию о визуализации
        result['expected_visualization'] = test_case.get('visualization_type', 'unknown')
        result['source'] = test_case.get('source', 'unknown')
        
        # Статистика по визуализации
        viz_type = test_case.get('visualization_type', 'unknown')
        if viz_type not in viz_stats:
            viz_stats[viz_type] = {'total': 0, 'success': 0, 'failed': 0}
        
        viz_stats[viz_type]['total'] += 1
        if result.get('success'):
            viz_stats[viz_type]['success'] += 1
        else:
            viz_stats[viz_type]['failed'] += 1
        
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Вычисление статистики
    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    failed = total - successful
    
    # Метрики для успешных тестов
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        exact_matches = sum(
            1 for r in successful_results
            if r.get('comparison', {}).get('exact_match', False)
        )
        
        similarities = [
            r.get('comparison', {}).get('similarity', 0)
            for r in successful_results
            if r.get('comparison')
        ]
        
        tables_matches = sum(
            1 for r in successful_results
            if r.get('comparison', {}).get('tables_match', False)
        )
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        avg_time = sum(r.get('generation_time', 0) for r in successful_results) / len(successful_results)
        avg_retries = sum(r.get('retry_count', 0) for r in successful_results) / len(successful_results)
        
        confidences = [
            r.get('confidence', 0) for r in successful_results
            if r.get('confidence') is not None
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    else:
        exact_matches = 0
        avg_similarity = 0
        tables_matches = 0
        avg_time = 0
        avg_retries = 0
        avg_confidence = 0
    
    stats = {
        'total_tests': total,
        'successful': successful,
        'failed': failed,
        'success_rate': successful / total if total > 0 else 0,
        'exact_matches': exact_matches,
        'exact_match_rate': exact_matches / successful if successful > 0 else 0,
        'avg_similarity': avg_similarity,
        'tables_match_rate': tables_matches / successful if successful > 0 else 0,
        'avg_generation_time': avg_time,
        'avg_retries': avg_retries,
        'avg_confidence': avg_confidence,
        'total_time': total_time,
        'throughput': total / total_time if total_time > 0 else 0,
        'visualization_stats': viz_stats,
        'results': results
    }
    
    return stats


def print_statistics(stats: Dict[str, Any]):
    """Вывод статистики"""
    print("\n" + "="*80)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ БЕНЧМАРКА")
    print("="*80)
    
    print(f"\n📈 Общая статистика:")
    print(f"   Всего тестов: {stats['total_tests']}")
    print(f"   Успешных: {stats['successful']} ({stats['success_rate']*100:.1f}%)")
    print(f"   Неудачных: {stats['failed']} ({(1-stats['success_rate'])*100:.1f}%)")
    
    print(f"\n🎯 Точность SQL:")
    print(f"   Точное совпадение: {stats['exact_matches']} ({stats['exact_match_rate']*100:.1f}%)")
    print(f"   Средняя схожесть: {stats['avg_similarity']*100:.1f}%")
    print(f"   Совпадение таблиц: {stats['tables_match_rate']*100:.1f}%")
    
    print(f"\n⚡ Производительность:")
    print(f"   Среднее время генерации: {stats['avg_generation_time']:.2f}s")
    print(f"   Среднее количество retry: {stats['avg_retries']:.2f}")
    print(f"   Средняя уверенность: {stats['avg_confidence']*100:.1f}%")
    print(f"   Общее время: {stats['total_time']:.1f}s")
    print(f"   Пропускная способность: {stats['throughput']:.2f} запросов/сек")
    
    if 'visualization_stats' in stats:
        print(f"\n📊 Статистика по типам визуализации:")
        for viz_type, viz_stat in sorted(stats['visualization_stats'].items()):
            success_rate = (viz_stat['success'] / viz_stat['total'] * 100) if viz_stat['total'] > 0 else 0
            print(f"   {viz_type}:")
            print(f"      Всего: {viz_stat['total']}")
            print(f"      Успешных: {viz_stat['success']} ({success_rate:.1f}%)")
            print(f"      Неудачных: {viz_stat['failed']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    from agents.enhanced_sql_agent import create_universal_sql_agent
    from core.llm_manager import LLMManager
    from config.config import settings
    
    print("🚀 Тестирование SQL и визуализации на бенчмарке\n")
    
    # Создание агента
    print("📡 Инициализация агента...")
    try:
        # Получить правильный ключ для провайдера
        provider_key_map = {
            "gemini": "gemini_api_key",
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key"
        }
        api_key_name = provider_key_map.get(settings.llm_provider, "gemini_api_key")
        api_key = getattr(settings, api_key_name, None)
        
        llm = LLMManager(
            provider=settings.llm_provider,
            model=settings.llm_model,
            gemini_api_key=api_key if settings.llm_provider == "gemini" else None,
            openai_api_key=api_key if settings.llm_provider == "openai" else None,
            anthropic_api_key=api_key if settings.llm_provider == "anthropic" else None
        )
        
        agent = create_universal_sql_agent(
            connection_url=settings.database_url,
            llm_manager=llm,
            enable_analysis=True,
            max_retries=2
        )
        
        print("✅ Агент инициализирован\n")
        
        # Тестирование на бенчмарке
        benchmark_path = project_root / "tests" / "benchmark_3000.json"
        
        if not benchmark_path.exists():
            print(f"⚠️  Бенчмарк не найден. Создаю...")
            from tests.create_benchmark import create_benchmark
            
            csv_path = project_root / "home_credit_qa_11000_with_hard_joins.csv"
            json_path = project_root / "result10000.json"
            
            create_benchmark(
                csv_path=str(csv_path),
                json_path=str(json_path),
                output_path=str(benchmark_path),
                target_size=3000
            )
        
        # Тестирование (можно ограничить для быстрого теста)
        max_tests = 100  # Для начала 100, потом можно увеличить
        
        stats = test_benchmark(agent, str(benchmark_path), max_tests=max_tests)
        print_statistics(stats)
        
        # Сохранение результатов
        results_file = project_root / "tests" / "results" / "benchmark_test_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Результаты сохранены: {results_file}")
        print("\n✅ Тестирование завершено!")
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}", exc_info=True)
        print(f"\n❌ Ошибка: {e}")

