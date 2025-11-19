"""
Быстрое тестирование на небольшой выборке
"""

import json
import csv
import sys
from pathlib import Path
import logging

# Добавить родительскую директорию в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_accuracy import SQLAccuracyTester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_test_csv(agent, num_tests=10):
    """Быстрый тест на CSV датасете"""
    csv_path = project_root / "home_credit_qa_11000_with_hard_joins.csv"
    
    if not csv_path.exists():
        print(f"❌ CSV файл не найден: {csv_path}")
        return None
    
    print(f"\n📋 Тестирование на CSV датасете (первые {num_tests} тестов)...")
    tester = SQLAccuracyTester(agent)
    stats = tester.test_from_csv(str(csv_path), max_tests=num_tests)
    tester.print_statistics(stats)
    
    return stats


def quick_test_json(agent, num_tests=10):
    """Быстрый тест на JSON датасете"""
    json_path = project_root / "result10000.json"
    
    if not json_path.exists():
        print(f"❌ JSON файл не найден: {json_path}")
        return None
    
    print(f"\n📋 Тестирование на JSON датасете (первые {num_tests} тестов)...")
    tester = SQLAccuracyTester(agent)
    stats = tester.test_from_json(str(json_path), max_tests=num_tests)
    tester.print_statistics(stats)
    
    return stats


if __name__ == "__main__":
    from agents.enhanced_sql_agent import create_universal_sql_agent
    from core.llm_manager import LLMManager
    from config.config import settings
    
    print("🚀 Быстрое тестирование SQL генерации\n")
    
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
            max_retries=2  # Меньше retry для быстрого теста
        )
        
        print("✅ Агент инициализирован\n")
        
        # Тестирование
        num_tests = 10  # Количество тестов для быстрой проверки
        
        csv_stats = quick_test_csv(agent, num_tests)
        json_stats = quick_test_json(agent, num_tests)
        
        # Сохранение результатов
        if csv_stats:
            import json as json_lib
            results_dir = project_root / "tests" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            with open(results_dir / "quick_csv_test.json", 'w', encoding='utf-8') as f:
                json_lib.dump(csv_stats, f, indent=2, ensure_ascii=False, default=str)
        
        if json_stats:
            import json as json_lib
            results_dir = project_root / "tests" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            with open(results_dir / "quick_json_test.json", 'w', encoding='utf-8') as f:
                json_lib.dump(json_stats, f, indent=2, ensure_ascii=False, default=str)
        
        print("\n✅ Тестирование завершено!")
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}", exc_info=True)
        print(f"\n❌ Ошибка: {e}")

