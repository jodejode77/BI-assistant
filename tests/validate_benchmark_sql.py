"""
Проверка SQL запросов из бенчмарка без использования LLM API
Просто валидация и проверка возможности выполнения
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

from core.database_adapter import create_database_adapter
from config.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_sql_query(db_adapter, sql: str) -> Dict[str, Any]:
    """
    Валидация SQL запроса
    
    Args:
        db_adapter: Database adapter
        sql: SQL запрос для проверки
        
    Returns:
        Результат валидации
    """
    # Очистить SQL
    sql = sql.replace(';', '').strip()
    
    result = {
        'sql': sql,
        'valid': False,
        'executable': False,
        'error': None,
        'execution_time': 0.0,
        'rows_returned': 0
    }
    
    # 1. Синтаксическая валидация через EXPLAIN
    try:
        is_valid, error = db_adapter.validate_query(sql)
        result['valid'] = is_valid
        if not is_valid:
            result['error'] = error
            return result
    except Exception as e:
        result['error'] = f"Validation error: {str(e)}"
        return result
    
    # 2. Попытка выполнения (с ограничением)
    try:
        start_time = time.time()
        data = db_adapter.execute_query(sql)
        execution_time = time.time() - start_time
        
        result['executable'] = True
        result['execution_time'] = execution_time
        result['rows_returned'] = len(data)
        result['columns'] = list(data.columns) if not data.empty else []
        
    except Exception as e:
        result['error'] = f"Execution error: {str(e)}"
        result['executable'] = False
    
    return result


def validate_benchmark(
    benchmark_path: str,
    max_tests: int = None
) -> Dict[str, Any]:
    """
    Валидация всех SQL запросов из бенчмарка
    
    Args:
        benchmark_path: Путь к бенчмарку
        max_tests: Максимальное количество тестов (None = все)
        
    Returns:
        Статистика валидации
    """
    print(f"\n📋 Загрузка бенчмарка из {benchmark_path}...")
    
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmark = json.load(f)
    
    if max_tests:
        benchmark = benchmark[:max_tests]
        print(f"⚠️  Ограничено до {max_tests} тестов")
    
    print(f"✅ Загружено {len(benchmark)} тестов\n")
    
    # Подключение к БД
    print("📡 Подключение к базе данных...")
    db_adapter = create_database_adapter(settings.database_url)
    print(f"✅ Подключено к {db_adapter.dialect.value}\n")
    
    results = []
    start_time = time.time()
    
    # Статистика по типам визуализации
    viz_stats = {}
    
    print("🔍 Начало валидации SQL запросов...\n")
    
    for i, test_case in enumerate(benchmark):
        if (i + 1) % 100 == 0:
            print(f"Прогресс: {i + 1}/{len(benchmark)} тестов...")
        
        sql = test_case['sql']
        viz_type = test_case.get('visualization_type', 'unknown')
        
        # Валидация SQL
        validation_result = validate_sql_query(db_adapter, sql)
        
        result = {
            'test_id': i + 1,
            'question': test_case['question'],
            'sql': sql,
            'visualization_type': viz_type,
            'source': test_case.get('source', 'unknown'),
            'valid': validation_result['valid'],
            'executable': validation_result['executable'],
            'error': validation_result.get('error'),
            'execution_time': validation_result.get('execution_time', 0),
            'rows_returned': validation_result.get('rows_returned', 0),
            'columns': validation_result.get('columns', [])
        }
        
        # Статистика по визуализации
        if viz_type not in viz_stats:
            viz_stats[viz_type] = {
                'total': 0,
                'valid': 0,
                'executable': 0,
                'invalid': 0,
                'errors': []
            }
        
        viz_stats[viz_type]['total'] += 1
        if validation_result['valid']:
            viz_stats[viz_type]['valid'] += 1
        else:
            viz_stats[viz_type]['invalid'] += 1
            if validation_result.get('error'):
                viz_stats[viz_type]['errors'].append(validation_result['error'])
        
        if validation_result['executable']:
            viz_stats[viz_type]['executable'] += 1
        
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Вычисление общей статистики
    total = len(results)
    valid_count = sum(1 for r in results if r['valid'])
    executable_count = sum(1 for r in results if r['executable'])
    
    # Статистика по ошибкам
    error_types = {}
    for r in results:
        if r.get('error'):
            error_msg = str(r['error'])
            # Упростить ошибку для группировки
            if 'does not exist' in error_msg:
                error_type = 'table_or_column_not_found'
            elif 'syntax error' in error_msg.lower():
                error_type = 'syntax_error'
            elif 'ambiguous' in error_msg.lower():
                error_type = 'ambiguous_column'
            else:
                error_type = 'other'
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    # Средние значения
    avg_execution_time = sum(r.get('execution_time', 0) for r in results if r['executable']) / executable_count if executable_count > 0 else 0
    avg_rows = sum(r.get('rows_returned', 0) for r in results if r['executable']) / executable_count if executable_count > 0 else 0
    
    stats = {
        'total_tests': total,
        'valid_sql': valid_count,
        'executable_sql': executable_count,
        'invalid_sql': total - valid_count,
        'valid_rate': valid_count / total if total > 0 else 0,
        'executable_rate': executable_count / total if total > 0 else 0,
        'avg_execution_time': avg_execution_time,
        'avg_rows_returned': avg_rows,
        'total_time': total_time,
        'throughput': total / total_time if total_time > 0 else 0,
        'error_types': error_types,
        'visualization_stats': viz_stats,
        'results': results
    }
    
    return stats


def print_statistics(stats: Dict[str, Any]):
    """Вывод статистики"""
    print("\n" + "="*80)
    print("📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ SQL ИЗ БЕНЧМАРКА")
    print("="*80)
    
    print(f"\n📈 Общая статистика:")
    print(f"   Всего тестов: {stats['total_tests']}")
    print(f"   Валидных SQL: {stats['valid_sql']} ({stats['valid_rate']*100:.1f}%)")
    print(f"   Выполняемых SQL: {stats['executable_sql']} ({stats['executable_rate']*100:.1f}%)")
    print(f"   Невалидных SQL: {stats['invalid_sql']} ({(1-stats['valid_rate'])*100:.1f}%)")
    
    print(f"\n⚡ Производительность:")
    print(f"   Среднее время выполнения: {stats['avg_execution_time']:.3f}s")
    print(f"   Среднее количество строк: {stats['avg_rows_returned']:.1f}")
    print(f"   Общее время валидации: {stats['total_time']:.1f}s")
    print(f"   Пропускная способность: {stats['throughput']:.2f} запросов/сек")
    
    if stats.get('error_types'):
        print(f"\n❌ Типы ошибок:")
        for error_type, count in sorted(stats['error_types'].items(), key=lambda x: -x[1]):
            print(f"   {error_type}: {count}")
    
    if 'visualization_stats' in stats:
        print(f"\n📊 Статистика по типам визуализации:")
        for viz_type, viz_stat in sorted(stats['visualization_stats'].items()):
            valid_rate = (viz_stat['valid'] / viz_stat['total'] * 100) if viz_stat['total'] > 0 else 0
            exec_rate = (viz_stat['executable'] / viz_stat['total'] * 100) if viz_stat['total'] > 0 else 0
            print(f"   {viz_type}:")
            print(f"      Всего: {viz_stat['total']}")
            print(f"      Валидных: {viz_stat['valid']} ({valid_rate:.1f}%)")
            print(f"      Выполняемых: {viz_stat['executable']} ({exec_rate:.1f}%)")
            print(f"      Невалидных: {viz_stat['invalid']}")
            if viz_stat.get('errors'):
                unique_errors = len(set(viz_stat['errors'][:5]))  # Первые 5 уникальных
                print(f"      Уникальных ошибок: {unique_errors}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    benchmark_path = project_root / "tests" / "benchmark_3000.json"
    
    if not benchmark_path.exists():
        print(f"❌ Бенчмарк не найден: {benchmark_path}")
        print("Запустите сначала: python tests/create_benchmark.py")
        sys.exit(1)
    
    print("🚀 Валидация SQL запросов из бенчмарка (без LLM API)\n")
    
    # Можно ограничить количество для быстрого теста
    max_tests = 100  # Быстрый тест на 100 примерах
    
    try:
        stats = validate_benchmark(str(benchmark_path), max_tests=max_tests)
        print_statistics(stats)
        
        # Сохранение результатов
        results_file = project_root / "tests" / "results" / "benchmark_validation.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохранить только статистику (без всех результатов для экономии места)
        stats_to_save = {k: v for k, v in stats.items() if k != 'results'}
        stats_to_save['sample_results'] = stats['results'][:10]  # Первые 10 как пример
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Статистика сохранена: {results_file}")
        
        # Сохранить полные результаты в отдельный файл (опционально)
        full_results_file = project_root / "tests" / "results" / "benchmark_validation_full.json"
        with open(full_results_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Полные результаты: {full_results_file}")
        print("\n✅ Валидация завершена!")
        
    except Exception as e:
        logger.error(f"Ошибка при валидации: {e}", exc_info=True)
        print(f"\n❌ Ошибка: {e}")

