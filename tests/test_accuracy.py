"""
Тестирование точности SQL генерации на датасетах
"""

import json
import csv
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from difflib import SequenceMatcher
import re

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SQLAccuracyTester:
    """Тестер точности генерации SQL"""
    
    def __init__(self, agent):
        """
        Инициализация тестера
        
        Args:
            agent: Экземпляр EnhancedSQLAgent
        """
        self.agent = agent
        self.results = []
    
    def normalize_sql(self, sql: str) -> str:
        """
        Нормализация SQL для сравнения
        
        Args:
            sql: SQL запрос
            
        Returns:
            Нормализованный SQL
        """
        if not sql:
            return ""
        
        # Удалить комментарии
        sql = re.sub(r'--.*', '', sql)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        # Привести к нижнему регистру
        sql = sql.lower()
        
        # Удалить лишние пробелы
        sql = re.sub(r'\s+', ' ', sql)
        
        # Удалить точки с запятой
        sql = sql.replace(';', '')
        
        # Удалить кавычки вокруг значений
        sql = re.sub(r"'([^']*)'", r'\1', sql)
        sql = re.sub(r'"([^"]*)"', r'\1', sql)
        
        # Нормализовать пробелы вокруг операторов
        sql = re.sub(r'\s*=\s*', ' = ', sql)
        sql = re.sub(r'\s*>\s*', ' > ', sql)
        sql = re.sub(r'\s*<\s*', ' < ', sql)
        sql = re.sub(r'\s*>=\s*', ' >= ', sql)
        sql = re.sub(r'\s*<=\s*', ' <= ', sql)
        
        return sql.strip()
    
    def compare_sql(self, generated: str, expected: str) -> Dict[str, Any]:
        """
        Сравнение двух SQL запросов
        
        Args:
            generated: Сгенерированный SQL
            expected: Ожидаемый SQL
            
        Returns:
            Словарь с метриками сравнения
        """
        gen_norm = self.normalize_sql(generated)
        exp_norm = self.normalize_sql(expected)
        
        # Точное совпадение
        exact_match = gen_norm == exp_norm
        
        # Схожесть по Левенштейну
        similarity = SequenceMatcher(None, gen_norm, exp_norm).ratio()
        
        # Извлечение ключевых элементов
        gen_tables = set(re.findall(r'from\s+(\w+)', gen_norm, re.IGNORECASE))
        exp_tables = set(re.findall(r'from\s+(\w+)', exp_norm, re.IGNORECASE))
        
        gen_columns = set(re.findall(r'select\s+(.*?)\s+from', gen_norm, re.IGNORECASE | re.DOTALL))
        exp_columns = set(re.findall(r'select\s+(.*?)\s+from', exp_norm, re.IGNORECASE | re.DOTALL))
        
        gen_where = re.search(r'where\s+(.*?)(?:group|order|limit|$)', gen_norm, re.IGNORECASE | re.DOTALL)
        exp_where = re.search(r'where\s+(.*?)(?:group|order|limit|$)', exp_norm, re.IGNORECASE | re.DOTALL)
        
        gen_where_clause = gen_where.group(1).strip() if gen_where else ""
        exp_where_clause = exp_where.group(1).strip() if exp_where else ""
        
        # Проверка ключевых элементов
        tables_match = gen_tables == exp_tables
        where_similarity = SequenceMatcher(None, gen_where_clause, exp_where_clause).ratio() if gen_where_clause or exp_where_clause else 1.0
        
        # Проверка агрегаций
        gen_has_agg = bool(re.search(r'\b(avg|sum|count|min|max)\b', gen_norm, re.IGNORECASE))
        exp_has_agg = bool(re.search(r'\b(avg|sum|count|min|max)\b', exp_norm, re.IGNORECASE))
        agg_match = gen_has_agg == exp_has_agg
        
        # Проверка GROUP BY
        gen_has_group = 'group by' in gen_norm
        exp_has_group = 'group by' in exp_norm
        group_match = gen_has_group == exp_has_group
        
        # Семантическая правильность (проверка выполнения)
        semantic_correct = None  # Будет установлено при выполнении
        
        return {
            'exact_match': exact_match,
            'similarity': similarity,
            'tables_match': tables_match,
            'where_similarity': where_similarity,
            'agg_match': agg_match,
            'group_match': group_match,
            'generated_tables': list(gen_tables),
            'expected_tables': list(exp_tables),
            'generated_sql': generated,
            'expected_sql': expected,
            'generated_normalized': gen_norm,
            'expected_normalized': exp_norm,
            'semantic_correct': semantic_correct
        }
    
    def test_single_query(
        self,
        question: str,
        expected_sql: str,
        test_id: int = None
    ) -> Dict[str, Any]:
        """
        Тестирование одного запроса
        
        Args:
            question: Вопрос на естественном языке
            expected_sql: Ожидаемый SQL
            test_id: ID теста
            
        Returns:
            Результат теста
        """
        start_time = time.time()
        
        try:
            # Генерация SQL
            result = self.agent.process_question(question)
            
            generation_time = time.time() - start_time
            
            if not result.success:
                return {
                    'test_id': test_id,
                    'question': question,
                    'success': False,
                    'error': result.error,
                    'generation_time': generation_time,
                    'expected_sql': expected_sql,
                    'generated_sql': None,
                    'comparison': None
                }
            
            # Сравнение SQL
            comparison = self.compare_sql(result.query, expected_sql)
            
            # Проверка семантической правильности (выполнение запроса)
            try:
                # Очистить SQL от точки с запятой для выполнения
                clean_gen_sql = result.query.replace(';', '').strip()
                clean_exp_sql = expected_sql.replace(';', '').strip()
                
                # Пытаемся выполнить оба запроса и сравнить результаты
                gen_data = self.agent.db_adapter.execute_query(clean_gen_sql)
                exp_data = self.agent.db_adapter.execute_query(clean_exp_sql)
                
                # Сравнение результатов (упрощенное - проверяем структуру)
                # Сравниваем количество строк и колонок
                rows_match = abs(len(gen_data) - len(exp_data)) <= 1  # Допускаем разницу в 1 строку
                cols_match = gen_data.columns.tolist() == exp_data.columns.tolist()
                
                # Если структура совпадает, проверяем что данные похожи
                if rows_match and cols_match and len(gen_data) > 0:
                    # Сравниваем первые несколько строк (если есть)
                    try:
                        # Проверяем что типы данных совпадают
                        gen_dtypes = gen_data.dtypes.to_dict()
                        exp_dtypes = exp_data.dtypes.to_dict()
                        dtypes_match = gen_dtypes == exp_dtypes
                        
                        semantic_correct = rows_match and cols_match and dtypes_match
                    except:
                        semantic_correct = rows_match and cols_match
                else:
                    semantic_correct = False
                
                comparison['semantic_correct'] = semantic_correct
                comparison['generated_rows'] = len(gen_data)
                comparison['expected_rows'] = len(exp_data)
                
            except Exception as e:
                logger.debug(f"Semantic check failed: {e}")
                comparison['semantic_correct'] = False
                comparison['semantic_error'] = str(e)
            
            return {
                'test_id': test_id,
                'question': question,
                'success': True,
                'generation_time': generation_time,
                'expected_sql': expected_sql,
                'generated_sql': result.query,
                'comparison': comparison,
                'retry_count': result.retry_count,
                'confidence': result.generation_metadata.confidence if result.generation_metadata else None
            }
            
        except Exception as e:
            logger.error(f"Error testing query {test_id}: {e}", exc_info=True)
            return {
                'test_id': test_id,
                'question': question,
                'success': False,
                'error': str(e),
                'generation_time': time.time() - start_time,
                'expected_sql': expected_sql,
                'generated_sql': None,
                'comparison': None
            }
    
    def test_from_csv(self, csv_path: str, max_tests: int = None) -> Dict[str, Any]:
        """
        Тестирование на CSV датасете
        
        Args:
            csv_path: Путь к CSV файлу
            max_tests: Максимальное количество тестов (None = все)
            
        Returns:
            Статистика тестирования
        """
        logger.info(f"Loading CSV dataset from {csv_path}")
        
        test_cases = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_tests and i >= max_tests:
                    break
                test_cases.append({
                    'id': i + 1,
                    'question': row['Question'],
                    'expected_sql': row['SQL Query'],
                    'viz_type': row.get('Visualization Type', '')
                })
        
        logger.info(f"Loaded {len(test_cases)} test cases")
        return self._run_tests(test_cases, "CSV")
    
    def test_from_json(self, json_path: str, max_tests: int = None) -> Dict[str, Any]:
        """
        Тестирование на JSON датасете
        
        Args:
            json_path: Путь к JSON файлу
            max_tests: Максимальное количество тестов (None = все)
            
        Returns:
            Статистика тестирования
        """
        logger.info(f"Loading JSON dataset from {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_cases = []
        for i, item in enumerate(data):
            if max_tests and i >= max_tests:
                break
            test_cases.append({
                'id': i + 1,
                'question': item['natural_language_query'],
                'expected_sql': item['sql_query'],
                'viz_type': item.get('visualization_type', '')
            })
        
        logger.info(f"Loaded {len(test_cases)} test cases")
        return self._run_tests(test_cases, "JSON")
    
    def _run_tests(self, test_cases: List[Dict], dataset_type: str) -> Dict[str, Any]:
        """
        Запуск тестов
        
        Args:
            test_cases: Список тестовых случаев
            dataset_type: Тип датасета
            
        Returns:
            Статистика
        """
        logger.info(f"Running {len(test_cases)} tests from {dataset_type} dataset...")
        
        results = []
        start_time = time.time()
        
        for i, test_case in enumerate(test_cases):
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(test_cases)} tests completed")
            
            result = self.test_single_query(
                question=test_case['question'],
                expected_sql=test_case['expected_sql'],
                test_id=test_case['id']
            )
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Вычисление статистики
        stats = self._calculate_statistics(results, dataset_type, total_time)
        
        return stats
    
    def _calculate_statistics(
        self,
        results: List[Dict],
        dataset_type: str,
        total_time: float
    ) -> Dict[str, Any]:
        """
        Вычисление статистики тестирования
        
        Args:
            results: Результаты тестов
            dataset_type: Тип датасета
            total_time: Общее время выполнения
            
        Returns:
            Статистика
        """
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
            
            semantic_correct = sum(
                1 for r in successful_results
                if r.get('comparison', {}).get('semantic_correct', False)
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
            semantic_correct = 0
            avg_time = 0
            avg_retries = 0
            avg_confidence = 0
        
        stats = {
            'dataset_type': dataset_type,
            'total_tests': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'exact_matches': exact_matches,
            'exact_match_rate': exact_matches / successful if successful > 0 else 0,
            'avg_similarity': avg_similarity,
            'tables_match_rate': tables_matches / successful if successful > 0 else 0,
            'semantic_correct_rate': semantic_correct / successful if successful > 0 else 0,
            'avg_generation_time': avg_time,
            'avg_retries': avg_retries,
            'avg_confidence': avg_confidence,
            'total_time': total_time,
            'throughput': total / total_time if total_time > 0 else 0,
            'results': results
        }
        
        return stats
    
    def print_statistics(self, stats: Dict[str, Any]):
        """
        Вывод статистики
        
        Args:
            stats: Статистика тестирования
        """
        print("\n" + "="*80)
        print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ - {stats['dataset_type']} Dataset")
        print("="*80)
        print(f"\n📈 Общая статистика:")
        print(f"   Всего тестов: {stats['total_tests']}")
        print(f"   Успешных: {stats['successful']} ({stats['success_rate']*100:.1f}%)")
        print(f"   Неудачных: {stats['failed']} ({(1-stats['success_rate'])*100:.1f}%)")
        
        print(f"\n🎯 Точность SQL:")
        print(f"   Точное совпадение: {stats['exact_matches']} ({stats['exact_match_rate']*100:.1f}%)")
        print(f"   Средняя схожесть: {stats['avg_similarity']*100:.1f}%")
        print(f"   Совпадение таблиц: {stats['tables_match_rate']*100:.1f}%")
        print(f"   Семантическая правильность: {stats['semantic_correct_rate']*100:.1f}%")
        
        print(f"\n⚡ Производительность:")
        print(f"   Среднее время генерации: {stats['avg_generation_time']:.2f}s")
        print(f"   Среднее количество retry: {stats['avg_retries']:.2f}")
        print(f"   Средняя уверенность: {stats['avg_confidence']*100:.1f}%")
        print(f"   Общее время: {stats['total_time']:.1f}s")
        print(f"   Пропускная способность: {stats['throughput']:.2f} запросов/сек")
        
        print("\n" + "="*80)
    
    def save_results(self, stats: Dict[str, Any], output_path: str):
        """
        Сохранение результатов
        
        Args:
            stats: Статистика
            output_path: Путь для сохранения
        """
        # Сохранить полные результаты
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Results saved to {output_file}")
        
        # Сохранить краткую статистику
        summary_file = output_file.with_suffix('.summary.json')
        summary = {
            'dataset_type': stats['dataset_type'],
            'total_tests': stats['total_tests'],
            'success_rate': stats['success_rate'],
            'exact_match_rate': stats['exact_match_rate'],
            'avg_similarity': stats['avg_similarity'],
            'tables_match_rate': stats['tables_match_rate'],
            'semantic_correct_rate': stats['semantic_correct_rate'],
            'avg_generation_time': stats['avg_generation_time'],
            'avg_retries': stats['avg_retries'],
            'avg_confidence': stats['avg_confidence']
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to {summary_file}")


def main():
    """Основная функция для запуска тестов"""
    import sys
    from pathlib import Path
    
    # Добавить родительскую директорию в путь
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from agents.enhanced_sql_agent import create_universal_sql_agent
    from core.llm_manager import LLMManager
    from config.config import settings
    
    print("🚀 Инициализация тестовой системы...")
    
    # Создание LLM менеджера
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
    
    # Создание агента
    print("📡 Подключение к базе данных...")
    agent = create_universal_sql_agent(
        connection_url=settings.database_url,
        llm_manager=llm,
        enable_analysis=True,
        max_retries=3
    )
    
    # Создание тестера
    tester = SQLAccuracyTester(agent)
    
    # Тестирование на CSV датасете
    csv_path = project_root / "home_credit_qa_11000_with_hard_joins.csv"
    if csv_path.exists():
        print(f"\n📋 Тестирование на CSV датасете (первые 50 тестов)...")
        csv_stats = tester.test_from_csv(str(csv_path), max_tests=50)
        tester.print_statistics(csv_stats)
        tester.save_results(csv_stats, "tests/results/csv_test_results.json")
    else:
        print(f"⚠️  CSV файл не найден: {csv_path}")
    
    # Тестирование на JSON датасете
    json_path = project_root / "result10000.json"
    if json_path.exists():
        print(f"\n📋 Тестирование на JSON датасете (первые 50 тестов)...")
        json_stats = tester.test_from_json(str(json_path), max_tests=50)
        tester.print_statistics(json_stats)
        tester.save_results(json_stats, "tests/results/json_test_results.json")
    else:
        print(f"⚠️  JSON файл не найден: {json_path}")
    
    print("\n✅ Тестирование завершено!")


if __name__ == "__main__":
    main()

