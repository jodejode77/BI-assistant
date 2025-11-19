"""
Валидация пайплайна на бенчмарке benchmark_3000.json
Проверяет: сработало / выполнилось / похожий результат
Без визуализации
"""

import json
import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Добавить родительскую директорию в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.orchestrator import AgentOrchestrator
from core.database_adapter import create_database_adapter
from core.llm_manager import LLMManager
from config.config import settings

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / "tests" / "benchmark_validation.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def compare_results(generated_df: pd.DataFrame, expected_sql: str, db_adapter) -> Dict[str, Any]:
    """
    Сравнить результаты сгенерированного SQL с ожидаемым SQL
    
    Args:
        generated_df: DataFrame с результатами сгенерированного SQL
        expected_sql: Ожидаемый SQL запрос
        db_adapter: Database adapter для выполнения ожидаемого SQL
        
    Returns:
        Dict с результатами сравнения
    """
    result = {
        'similar': False,
        'rows_match': False,
        'columns_match': False,
        'values_similar': False,
        'expected_rows': 0,
        'generated_rows': len(generated_df),
        'similarity_score': 0.0
    }
    
    try:
        # Очистить ожидаемый SQL (убрать точку с запятой, лишние пробелы)
        expected_sql_clean = expected_sql.replace(';', '').strip()
        
        # Выполнить ожидаемый SQL
        expected_df = db_adapter.execute_query(expected_sql_clean)
        result['expected_rows'] = len(expected_df)
        
        # Проверка количества строк
        result['rows_match'] = (len(generated_df) == len(expected_df))
        
        # Проверка колонок (игнорируя порядок)
        generated_cols = set(generated_df.columns)
        expected_cols = set(expected_df.columns)
        result['columns_match'] = (generated_cols == expected_cols)
        
        if result['rows_match'] and result['columns_match'] and len(generated_df) > 0:
            # Сравнение значений (для числовых колонок)
            numeric_cols = generated_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                # Сортировка для сравнения
                generated_sorted = generated_df.sort_values(by=list(generated_df.columns)).reset_index(drop=True)
                expected_sorted = expected_df.sort_values(by=list(expected_df.columns)).reset_index(drop=True)
                
                # Вычисление схожести для числовых колонок
                similarities = []
                for col in numeric_cols:
                    if col in expected_sorted.columns:
                        gen_values = generated_sorted[col].fillna(0)
                        exp_values = expected_sorted[col].fillna(0)
                        
                        # Проверка на близость значений (в пределах 1%)
                        if len(gen_values) > 0 and len(exp_values) > 0:
                            # Для агрегированных значений - проверка близости
                            if len(gen_values) == 1 and len(exp_values) == 1:
                                diff = abs(gen_values.iloc[0] - exp_values.iloc[0])
                                max_val = max(abs(gen_values.iloc[0]), abs(exp_values.iloc[0]), 1)
                                similarity = 1 - min(diff / max_val, 1.0)
                                similarities.append(similarity)
                            else:
                                # Для множественных строк - проверка корреляции
                                if len(gen_values) == len(exp_values):
                                    correlation = gen_values.corr(exp_values)
                                    if pd.notna(correlation):
                                        similarities.append(max(0, correlation))
                
                if similarities:
                    result['similarity_score'] = np.mean(similarities)
                    result['values_similar'] = result['similarity_score'] > 0.8
                else:
                    # Если нет числовых колонок, проверяем точное совпадение
                    result['values_similar'] = generated_sorted.equals(expected_sorted)
                    result['similarity_score'] = 1.0 if result['values_similar'] else 0.0
            else:
                # Для нечисловых колонок - точное совпадение
                generated_sorted = generated_df.sort_values(by=list(generated_df.columns)).reset_index(drop=True)
                expected_sorted = expected_df.sort_values(by=list(expected_df.columns)).reset_index(drop=True)
                result['values_similar'] = generated_sorted.equals(expected_sorted)
                result['similarity_score'] = 1.0 if result['values_similar'] else 0.0
        
        # Общая оценка схожести
        result['similar'] = (
            result['rows_match'] and 
            result['columns_match'] and 
            (result['values_similar'] or result['similarity_score'] > 0.7)
        )
        
    except Exception as e:
        logger.warning(f"Ошибка при сравнении результатов: {e}")
        result['error'] = str(e)
    
    return result


async def validate_single_example(
    orchestrator: AgentOrchestrator,
    db_adapter,
    example: Dict[str, Any],
    index: int
) -> Dict[str, Any]:
    """
    Валидация одного примера из бенчмарка
    
    Args:
        orchestrator: AgentOrchestrator instance
        db_adapter: Database adapter
        example: Пример из бенчмарка
        index: Индекс примера
        
    Returns:
        Dict с результатами валидации
    """
    question = example.get('question', '')
    expected_sql = example.get('sql', '')
    
    result = {
        'index': index,
        'question': question,
        'expected_sql': expected_sql,
        'worked': False,  # Сработало ли (успешно выполнился пайплайн)
        'executed': False,  # Выполнилось ли (SQL выполнился без ошибок)
        'similar_result': False,  # Похожий результат
        'generated_sql': None,
        'error': None,
        'execution_time': 0.0,
        'rows_returned': 0,
        'comparison': {}
    }
    
    try:
        start_time = time.time()
        
        # Запуск пайплайна (без визуализации - она будет пропущена автоматически)
        # Используем timeout для предотвращения зависаний
        try:
            task_result = await asyncio.wait_for(
                orchestrator.process_request(
                    user_input=question,
                    context={}
                ),
                timeout=120.0  # 2 минуты на запрос
            )
        except asyncio.TimeoutError:
            result['error'] = "Timeout: запрос превысил 2 минуты"
            result['execution_time'] = time.time() - start_time
            return result
        except BrokenPipeError:
            result['error'] = "Broken pipe: процесс был прерван"
            result['execution_time'] = time.time() - start_time
            return result
        
        result['execution_time'] = time.time() - start_time
        result['worked'] = task_result.success
        
        if task_result.success and task_result.sql_result:
            sql_result = task_result.sql_result
            result['executed'] = sql_result.success
            result['generated_sql'] = sql_result.query
            result['rows_returned'] = sql_result.rows_returned if sql_result.data is not None else 0
            
            # Сравнение результатов
            if sql_result.data is not None and not sql_result.data.empty:
                comparison = compare_results(sql_result.data, expected_sql, db_adapter)
                result['comparison'] = comparison
                result['similar_result'] = comparison.get('similar', False)
            else:
                result['comparison'] = {'error': 'No data returned'}
        else:
            result['error'] = task_result.error or "Unknown error"
            
    except BrokenPipeError as e:
        logger.warning(f"Broken pipe при валидации примера {index}: {e}")
        result['error'] = f"Broken pipe: {str(e)}"
        result['execution_time'] = time.time() - start_time if 'start_time' in locals() else 0.0
    except Exception as e:
        logger.error(f"Ошибка при валидации примера {index}: {e}", exc_info=True)
        result['error'] = str(e)
        result['execution_time'] = time.time() - start_time if 'start_time' in locals() else 0.0
    
    return result


async def validate_benchmark(
    benchmark_path: Path,
    max_examples: Optional[int] = None,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Валидация всего бенчмарка
    
    Args:
        benchmark_path: Путь к файлу бенчмарка
        max_examples: Максимальное количество примеров для проверки
        sample_size: Размер выборки для случайной проверки (если указан)
        
    Returns:
        Dict с результатами валидации
    """
    logger.info(f"📊 Загрузка бенчмарка: {benchmark_path}")
    
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmark = json.load(f)
    
    total_examples = len(benchmark)
    logger.info(f"✅ Загружено {total_examples} примеров")
    
    # Выборка для проверки
    if sample_size and sample_size < total_examples:
        import random
        random.seed(42)
        examples_to_check = random.sample(benchmark, sample_size)
        logger.info(f"🎲 Выбрана случайная выборка: {sample_size} примеров")
    elif max_examples:
        examples_to_check = benchmark[:max_examples]
        logger.info(f"📝 Проверка первых {max_examples} примеров")
    else:
        examples_to_check = benchmark
        logger.info(f"🔍 Проверка всех {total_examples} примеров")
    
    # Инициализация компонентов
    logger.info("🔧 Инициализация компонентов...")
    db_adapter = create_database_adapter(settings.database_url)
    llm = LLMManager(
        provider=settings.llm_provider,
        model=settings.llm_model,
        gemini_api_key=settings.gemini_api_key,
        openai_api_key=getattr(settings, 'openai_api_key', None),
        anthropic_api_key=getattr(settings, 'anthropic_api_key', None)
    )
    orchestrator = AgentOrchestrator(db_adapter, llm)
    
    logger.info("🚀 Начало валидации...")
    
    results = []
    stats = {
        'total': len(examples_to_check),
        'worked': 0,  # Сработало
        'executed': 0,  # Выполнилось
        'similar_result': 0,  # Похожий результат
        'errors': 0
    }
    
    start_time = time.time()
    
    # Валидация примеров последовательно (чтобы не перегружать LLM)
    for i, example in enumerate(examples_to_check, 1):
        logger.info(f"\n[{i}/{len(examples_to_check)}] Валидация: {example.get('question', '')[:80]}...")
        
        try:
            result = await asyncio.wait_for(
                validate_single_example(orchestrator, db_adapter, example, i),
                timeout=180.0  # 3 минуты на пример (включая LLM вызовы)
            )
            results.append(result)
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Timeout для примера {i}")
            results.append({
                'index': i,
                'question': example.get('question', ''),
                'error': 'Timeout: превышено 3 минуты',
                'worked': False,
                'executed': False,
                'similar_result': False
            })
            stats['errors'] += 1
        except KeyboardInterrupt:
            logger.info("\n⚠️ Прервано пользователем (Ctrl+C)")
            break
        except Exception as e:
            logger.error(f"Критическая ошибка при обработке примера {i}: {e}")
            results.append({
                'index': i,
                'question': example.get('question', ''),
                'error': f'Critical error: {str(e)}',
                'worked': False,
                'executed': False,
                'similar_result': False
            })
            stats['errors'] += 1
        
        # Обновление статистики
        if result['worked']:
            stats['worked'] += 1
        if result['executed']:
            stats['executed'] += 1
        if result['similar_result']:
            stats['similar_result'] += 1
        if result['error']:
            stats['errors'] += 1
        
        # Прогресс каждые 10 примеров
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(examples_to_check) - i) * avg_time
            logger.info(f"📊 Прогресс: {i}/{len(examples_to_check)} | "
                       f"Сработало: {stats['worked']} ({stats['worked']/i*100:.1f}%) | "
                       f"Выполнилось: {stats['executed']} ({stats['executed']/i*100:.1f}%) | "
                       f"Похожий результат: {stats['similar_result']} ({stats['similar_result']/i*100:.1f}%) | "
                       f"Осталось: ~{remaining/60:.1f} мин")
    
    total_time = time.time() - start_time
    
    # Финальная статистика
    final_stats = {
        **stats,
        'total_time': total_time,
        'avg_time_per_example': total_time / len(examples_to_check),
        'success_rate': {
            'worked': stats['worked'] / stats['total'] * 100,
            'executed': stats['executed'] / stats['total'] * 100,
            'similar_result': stats['similar_result'] / stats['total'] * 100
        }
    }
    
    # Сохранение результатов
    output_file = project_root / "tests" / "results" / f"benchmark_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    output_data = {
        'stats': final_stats,
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'benchmark_file': str(benchmark_path)
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("📊 ФИНАЛЬНАЯ СТАТИСТИКА")
    logger.info(f"{'='*80}")
    logger.info(f"Всего примеров: {final_stats['total']}")
    logger.info(f"Сработало (пайплайн успешен): {final_stats['worked']} ({final_stats['success_rate']['worked']:.1f}%)")
    logger.info(f"Выполнилось (SQL выполнен): {final_stats['executed']} ({final_stats['success_rate']['executed']:.1f}%)")
    logger.info(f"Похожий результат: {final_stats['similar_result']} ({final_stats['success_rate']['similar_result']:.1f}%)")
    logger.info(f"Ошибки: {final_stats['errors']}")
    logger.info(f"Общее время: {final_stats['total_time']/60:.1f} мин")
    logger.info(f"Среднее время на пример: {final_stats['avg_time_per_example']:.2f} сек")
    logger.info(f"\n💾 Результаты сохранены: {output_file}")
    
    return output_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Валидация пайплайна на бенчмарке')
    parser.add_argument('--benchmark', type=str, default='tests/benchmark_3000.json',
                       help='Путь к файлу бенчмарка')
    parser.add_argument('--max', type=int, default=None,
                       help='Максимальное количество примеров для проверки')
    parser.add_argument('--sample', type=int, default=None,
                       help='Размер случайной выборки для проверки')
    
    args = parser.parse_args()
    
    benchmark_path = project_root / args.benchmark
    
    if not benchmark_path.exists():
        logger.error(f"❌ Файл бенчмарка не найден: {benchmark_path}")
        sys.exit(1)
    
    # Запуск валидации
    results = asyncio.run(validate_benchmark(
        benchmark_path=benchmark_path,
        max_examples=args.max,
        sample_size=args.sample
    ))

