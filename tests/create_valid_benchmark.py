"""
Создание бенчмарка из валидных SQL запросов
С равномерным распределением по типам визуализации
Проверяет до 4500 запросов и создает бенчмарк из 3000 валидных
"""

import json
import csv
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from threading import Lock

# Добавить родительскую директорию в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.database_adapter import create_database_adapter
from config.config import settings

# Отключить логирование ошибок от database_adapter
logging.getLogger('core.database_adapter').setLevel(logging.CRITICAL)

# Настройка логирования в файл
log_file = project_root / "tests" / "benchmark_creation.log"
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

# Настройка логирования в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

# Создать logger и добавить оба обработчика
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False  # Не передавать в корневой logger

# Глобальный счетчик для прогресса
progress_lock = Lock()
progress_counter = {'processed': 0, 'valid': 0}


def validate_sql_query(db_adapter, sql: str) -> Dict[str, Any]:
    """Валидация SQL запроса"""
    sql = sql.replace(';', '').strip()
    
    result = {
        'valid': False,
        'executable': False,
        'error': None
    }
    
    # Синтаксическая валидация
    try:
        is_valid, error = db_adapter.validate_query(sql)
        result['valid'] = is_valid
        if not is_valid:
            result['error'] = error
            return result
    except Exception as e:
        result['error'] = f"Validation error: {str(e)}"
        return result
    
    # Попытка выполнения
    try:
        data = db_adapter.execute_query(sql)
        result['executable'] = True
    except Exception as e:
        result['error'] = f"Execution error: {str(e)}"
        result['executable'] = False
    
    return result


def load_all_datasets(max_queries: int = 4500) -> List[Dict[str, Any]]:
    """Загрузить все датасеты (CSV и JSON)"""
    datasets = []
    
    # CSV датасет
    csv_path = project_root / "home_credit_qa_11000_with_hard_joins.csv"
    if csv_path.exists():
        print(f"📋 Загрузка CSV датасета: {csv_path}...")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                datasets.append({
                    'question': row['Question'],
                    'sql': row['SQL Query'],
                    'visualization_type': row.get('Visualization Type', 'table').lower().strip(),
                    'source': 'csv'
                })
        print(f"✅ Загружено {len(datasets)} примеров из CSV")
    else:
        print(f"⚠️  CSV файл не найден: {csv_path}")
    
    # JSON датасет
    json_path = project_root / "result10000.json"
    if json_path.exists():
        print(f"📋 Загрузка JSON датасета: {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            initial_count = len(datasets)
            for item in data:
                datasets.append({
                    'question': item.get('natural_language_query', ''),
                    'sql': item.get('sql_query', ''),
                    'visualization_type': item.get('visualization_type', 'table').lower().strip(),
                    'source': 'json'
                })
        print(f"✅ Загружено {len(datasets) - initial_count} примеров из JSON")
    else:
        print(f"⚠️  JSON файл не найден: {json_path}")
    
    # Нормализация типов визуализации
    type_mapping = {
        'pie': 'pie',
        'bar': 'bar',
        'line': 'line',
        'table': 'table',
        'scatter': 'table',
        'histogram': 'bar',
        'area': 'line'
    }
    
    for item in datasets:
        viz_type = item['visualization_type']
        item['visualization_type'] = type_mapping.get(viz_type, 'table')
    
    # Ограничение количества
    if len(datasets) > max_queries:
        print(f"⚠️  Ограничение до {max_queries} запросов (было {len(datasets)})")
        random.seed(42)
        datasets = random.sample(datasets, max_queries)
    
    print(f"\n✅ Всего загружено: {len(datasets)} примеров\n")
    return datasets


def validate_single_query(args):
    """Валидация одного SQL запроса (для параллельной обработки)"""
    test_case, db_adapter, total_count = args
    sql = test_case.get('sql', '').strip()
    
    if not sql:
        return None, False
    
    # Использовать общий адаптер (SQLAlchemy engine thread-safe)
    validation_result = validate_sql_query(db_adapter, sql)
    
    # Обновить счетчик прогресса
    global progress_counter, progress_lock
    with progress_lock:
        progress_counter['processed'] += 1
        processed = progress_counter['processed']
        if validation_result['valid'] and validation_result['executable']:
            progress_counter['valid'] += 1
        valid_count = progress_counter['valid']
    
    # Выводить прогресс каждые 100 запросов
    if processed % 100 == 0:
        msg = f"📊 Прогресс: {processed}/{total_count} проверено | Валидных: {valid_count} ({valid_count/processed*100:.1f}%)"
        print(msg, flush=True)
        logger.info(msg)
    
    is_valid = validation_result['valid'] and validation_result['executable']
    if is_valid:
        test_case['validated'] = True
        test_case['validation_error'] = None
        return test_case, True
    else:
        test_case['validated'] = False
        test_case['validation_error'] = validation_result.get('error')
        return None, False


def validate_all_queries(queries: List[Dict[str, Any]], max_workers: int = 8, max_queries: int = None) -> List[Dict[str, Any]]:
    """Валидация всех SQL запросов с параллельной обработкой"""
    if max_queries and max_queries < len(queries):
        queries = queries[:max_queries]
        print(f"📋 Начало валидации {len(queries)} запросов (ограничено до {max_queries})...")
    else:
        print(f"📋 Начало валидации {len(queries)} запросов...")
    print(f"⚡ Используется {max_workers} потоков для ускорения\n")
    
    # Подключение к БД для проверки
    print("📡 Проверка подключения к базе данных...")
    test_adapter = create_database_adapter(settings.database_url)
    print(f"✅ Подключено к {test_adapter.dialect.value}\n")
    
    print("🔍 Валидация всех SQL запросов (параллельно)...\n")
    
    valid_queries = []
    start_time = time.time()
    
    # Сбросить счетчик прогресса
    global progress_counter
    progress_counter = {'processed': 0, 'valid': 0}
    
    # Создать один общий адаптер для всех потоков (SQLAlchemy engine thread-safe)
    shared_adapter = create_database_adapter(settings.database_url)
    
    # Подготовить аргументы для параллельной обработки
    args_list = [(test_case, shared_adapter, len(queries)) for test_case in queries]
    
    # Параллельная валидация с ранним завершением
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(validate_single_query, args): args[0] for args in args_list}
        
        completed = 0
        last_progress_time = time.time()
        stuck_threshold = 300  # 5 минут без прогресса = зависание
        
        for future in as_completed(futures, timeout=600):  # Общий таймаут 10 минут
            try:
                result, is_valid = future.result(timeout=20)  # Уменьшен таймаут до 20 сек
                if is_valid and result:
                    valid_queries.append(result)
                completed += 1
                last_progress_time = time.time()
                
                # Промежуточное сохранение каждые 500 запросов
                if completed % 100 == 0 and valid_queries:
                    logger.info(f"Промежуточное сохранение: {len(valid_queries)} валидных запросов собрано")
                
                # Сохранять прогресс каждые 100 запросов в последних 200
                if completed >= len(queries) - 200 and completed % 100 == 0:
                    logger.info(f"Прогресс сохранения: {completed}/{len(queries)}, валидных: {len(valid_queries)}")
                
                # Раннее завершение если осталось < 100 и уже есть достаточно валидных
                remaining = len(queries) - completed
                if remaining < 100 and len(valid_queries) >= 2500:
                    logger.info(f"Раннее завершение: осталось {remaining} запросов, но уже собрано {len(valid_queries)} валидных (достаточно)")
                    # Отменяем оставшиеся задачи
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
                
                # Проверка на зависание
                if time.time() - last_progress_time > stuck_threshold:
                    logger.warning(f"Обнаружено зависание: нет прогресса {stuck_threshold} секунд")
                    logger.info(f"Завершаем с {completed}/{len(queries)} проверено, {len(valid_queries)} валидных")
                    # Отменяем оставшиеся задачи
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
                    
            except FutureTimeoutError:
                completed += 1
                logger.warning(f"Таймаут при валидации запроса (общий таймаут)")
                # Проверяем, не зависли ли мы
                if time.time() - last_progress_time > stuck_threshold:
                    logger.warning(f"Зависание обнаружено, завершаем")
                    break
            except Exception as e:
                completed += 1
                if "timeout" in str(e).lower() or "Timeout" in str(type(e).__name__):
                    logger.warning(f"Таймаут при валидации запроса: {e}")
                else:
                    logger.debug(f"Ошибка валидации: {e}")
                pass
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Валидация завершена за {total_time:.1f}s")
    print(f"📊 Найдено валидных SQL: {len(valid_queries)} из {len(queries)} ({len(valid_queries)/len(queries)*100:.1f}%)")
    print(f"⚡ Скорость: {len(queries)/total_time:.1f} запросов/сек\n")
    
    return valid_queries


def create_balanced_benchmark(
    valid_queries: List[Dict[str, Any]],
    target_size: int = 500,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Создать сбалансированный бенчмарк с равномерным распределением по типам визуализации
    
    Args:
        valid_queries: Список валидных запросов
        target_size: Целевой размер бенчмарка
        seed: Seed для рандомизации
    """
    print(f"📊 Создание сбалансированного бенчмарка из {target_size} примеров...\n")
    
    random.seed(seed)
    
    # Группировка по типам визуализации
    by_viz_type = {}
    for query in valid_queries:
        viz_type = query.get('visualization_type', 'table')
        if viz_type not in by_viz_type:
            by_viz_type[viz_type] = []
        by_viz_type[viz_type].append(query)
    
    print("📈 Доступные типы визуализации:")
    for viz_type, queries in sorted(by_viz_type.items()):
        print(f"   {viz_type}: {len(queries)} валидных запросов")
    
    # Определить количество примеров на тип
    num_types = len(by_viz_type)
    if num_types == 0:
        print("❌ Нет валидных запросов!")
        return []
    
    examples_per_type = target_size // num_types
    remainder = target_size % num_types
    
    print(f"\n🎯 Распределение:")
    print(f"   Примеров на тип: {examples_per_type}")
    if remainder > 0:
        print(f"   Дополнительных: {remainder}")
    
    # Создание сбалансированного бенчмарка
    benchmark = []
    
    for i, (viz_type, queries) in enumerate(sorted(by_viz_type.items())):
        # Количество для этого типа
        count = examples_per_type
        if i < remainder:  # Распределить остаток
            count += 1
        
        # Ограничить доступным количеством
        count = min(count, len(queries))
        
        # Случайная выборка
        selected = random.sample(queries, count)
        benchmark.extend(selected)
        
        print(f"   {viz_type}: {count} примеров")
    
    # Перемешивание
    random.shuffle(benchmark)
    
    # Финальная статистика
    print(f"\n✅ Создан бенчмарк: {len(benchmark)} примеров")
    
    final_distribution = {}
    for query in benchmark:
        viz_type = query.get('visualization_type', 'table')
        final_distribution[viz_type] = final_distribution.get(viz_type, 0) + 1
    
    print(f"\n📊 Финальное распределение:")
    for viz_type, count in sorted(final_distribution.items()):
        print(f"   {viz_type}: {count} ({count/len(benchmark)*100:.1f}%)")
    
    return benchmark


def main():
    """Основная функция"""
    output_path = project_root / "tests" / "benchmark_3000_valid.json"
    log_path = project_root / "tests" / "benchmark_creation.log"
    
    msg = "🚀 Создание бенчмарка из валидных SQL запросов"
    print(msg)
    logger.info(msg)
    
    msg = "   Проверка до 4500 запросов, создание бенчмарка из 3000 валидных"
    print(msg)
    logger.info(msg)
    
    msg = f"   📝 Логи сохраняются в: {log_path}"
    print(msg)
    logger.info(msg)
    
    msg = "="*80
    print(f"\n{msg}\n")
    logger.info(msg)
    
    # Шаг 1: Загрузка всех датасетов
    all_queries = load_all_datasets(max_queries=3800)
    
    # Шаг 2: Валидация всех запросов (ограничено до 3800, чтобы избежать зависания на последних 100)
    # Из предыдущего запуска: 2555 валидных из 3800 = ~67% валидности
    # Этого достаточно для создания бенчмарка
    valid_queries = validate_all_queries(all_queries, max_workers=8, max_queries=3800)
    
    if len(valid_queries) < 3000:
        print(f"⚠️  Валидных запросов меньше 3000 ({len(valid_queries)})")
        print("Используем все доступные валидные запросы")
        target_size = len(valid_queries)
    else:
        target_size = 3000
    
    # Шаг 3: Создание сбалансированного бенчмарка
    balanced_benchmark = create_balanced_benchmark(
        valid_queries,
        target_size=target_size,
        seed=42
    )
    
    # Шаг 4: Сохранение
    print(f"\n💾 Сохранение бенчмарка в {output_path}...")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(balanced_benchmark, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Бенчмарк сохранен: {len(balanced_benchmark)} примеров")
    
    # Сохранение статистики
    stats_file = output_file.with_suffix('.stats.json')
    stats = {
        'total_examples': len(balanced_benchmark),
        'target_size': target_size,
        'all_valid': True,
        'visualization_distribution': {
            viz_type: sum(1 for q in balanced_benchmark if q.get('visualization_type') == viz_type)
            for viz_type in set(q.get('visualization_type', 'table') for q in balanced_benchmark)
        },
        'sources': {
            source: sum(1 for q in balanced_benchmark if q.get('source') == source)
            for source in set(q.get('source', 'unknown') for q in balanced_benchmark)
        }
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Статистика сохранена: {stats_file}")
    
    print("\n" + "="*80)
    print("✅ Готово! Бенчмарк из валидных SQL создан")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

