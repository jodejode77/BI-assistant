#!/usr/bin/env python3
import logging
import sys
import os
from pathlib import Path
import asyncio
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

from bot.telegram_bot import HomeCreditBot
from core.database import DatabaseManager
from config.config import settings


def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def check_database_connection():
    try:
        db = DatabaseManager(settings.database_url)
        tables = db.get_table_list()
        logging.info(f"База данных успешно подключена. Найдено таблиц: {len(tables)}.")
        
        if len(tables) == 0:
            logging.warning("В базе данных не найдено таблиц. Пожалуйста, запустите скрипт загрузки данных.")
            return False
        
        if "application_train" not in tables:
            logging.warning("Основная таблица 'application_train' не найдена. Пожалуйста, загрузите данные.")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Ошибка подключения к базе данных: {e}")
        logging.error("Пожалуйста, убедитесь, что PostgreSQL запущен (docker compose up -d)")
        return False


def check_environment():
    required_vars = ["TELEGRAM_BOT_TOKEN"]
    llm_vars = ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    if missing_required:
        logging.error(f"Отсутствуют обязательные переменные окружения: {', '.join(missing_required)}")
        logging.error("Пожалуйста, создайте файл .env с необходимой конфигурацией.")
        return False
    
    has_llm = any(os.getenv(var) for var in llm_vars)
    if not has_llm:
        logging.warning("Не найдены API ключи LLM (GEMINI_API_KEY, OPENAI_API_KEY или ANTHROPIC_API_KEY)")
        logging.warning("Бот будет иметь ограниченную функциональность без LLM провайдера.")
        
        response = input("Продолжить без LLM? (y/n): ").lower()
        if response != 'y':
            return False
    else:
        if os.getenv("GEMINI_API_KEY"):
            logging.info("Используется Google Gemini как LLM провайдер")
        elif os.getenv("OPENAI_API_KEY"):
            logging.info("Используется OpenAI как LLM провайдер")
        elif os.getenv("ANTHROPIC_API_KEY"):
            logging.info("Используется Anthropic как LLM провайдер")
    
    return True


def print_startup_banner():
    banner = """
╔══════════════════════════════════════════════════╗
║   БОТ АНАЛИЗА HOME CREDIT - v1.0.0              ║
║                                                  ║
║  Мульти-агентная система анализа кредитных      ║
║  рисков на базе Gemini AI + RAG + SQL агентов  ║
╚══════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    print_startup_banner()
    
    setup_logging()
    logging.info("Запуск бота анализа Home Credit...")
    
    if not check_environment():
        logging.error("Проверка окружения не пройдена. Выход.")
        sys.exit(1)
    
    if not check_database_connection():
        logging.error("Проверка базы данных не пройдена. Пожалуйста, убедитесь, что данные загружены.")
        print("\nДля загрузки данных:")
        print("1. Запустите PostgreSQL: docker compose up -d")
        print("2. Скачайте данные: ./download_kaggle_dataset.sh")
        print("3. Загрузите в базу данных: python load_data_to_postgres.py")
        sys.exit(1)
    
    try:
        bot = HomeCreditBot()
        
        logging.info("Бот успешно инициализирован")
        logging.info(f"Используется LLM провайдер: {settings.llm_provider}")
        logging.info(f"Используется модель: {settings.llm_model}")
        logging.info("Бот запускается... Нажмите Ctrl+C для остановки.")
        
        bot.run()
        
    except KeyboardInterrupt:
        logging.info("Бот остановлен пользователем")
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
