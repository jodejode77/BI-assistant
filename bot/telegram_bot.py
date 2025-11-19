import logging
import asyncio
from typing import Dict, Any, Optional
import io
import json
import re
from datetime import datetime, timezone

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    ReplyKeyboardMarkup,
    KeyboardButton
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)
import pandas as pd

from agents.orchestrator import AgentOrchestrator, TaskResult
from core.database_adapter import create_database_adapter
from core.llm_manager import LLMManager
from config.config import settings

logger = logging.getLogger(__name__)


def escape_markdown(text: str) -> str:
    """Escape markdown special characters, but preserve already escaped ones"""
    if not text:
        return text
    if '\\*' in text or '\\_' in text:
        return text
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text


def safe_markdown(text: str) -> str:
    if not text:
        return text
    problematic_patterns = ['_', '*', '[', ']', '(', ')', '`']
    if any(pattern in text for pattern in problematic_patterns):
        if text.count('*') % 2 == 0 and text.count('_') % 2 == 0:
            return text
        else:
            return escape_markdown(text)
    return text


async def safe_send_markdown(message_func, text: str, **kwargs):
    try:
        return await message_func(text, parse_mode="Markdown", **kwargs)
    except Exception as e:
        logger.warning(f"Failed to send with Markdown: {e}")
        kwargs.pop('parse_mode', None)
        return await message_func(text, **kwargs)


class HomeCreditBot:
    def __init__(self):
        self.db_adapter = create_database_adapter(
            settings.database_url,
            max_query_time=settings.max_query_time,
            max_rows=settings.max_rows_return
        )
        
        self.llm_manager = LLMManager(
            provider=settings.llm_provider,
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            anthropic_api_key=settings.anthropic_api_key,
            gemini_api_key=settings.gemini_api_key,
            mistral_api_key=settings.mistral_api_key,
        )
        
        self.orchestrator = AgentOrchestrator(
            database_adapter=self.db_adapter,
            llm_manager=self.llm_manager
        )
        
        self.user_sessions = {}
        
        logger.info("HomeCreditBot initialized successfully with new pipeline")
    
    def _initialize_rag_data(self):
        """Initialize RAG system with database schema and examples.
        
        Note: Enhanced RAG system auto-indexes on initialization,
        but we can add additional examples here if needed.
        """
        try:
            rag_system = self.orchestrator.rag_system
            
            example_queries = [
                {
                    "question": "Show me clients with high income",
                    "sql": "SELECT * FROM application_train WHERE AMT_INCOME_TOTAL > 500000 LIMIT 10"
                },
                {
                    "question": "What is the average loan amount?",
                    "sql": "SELECT AVG(AMT_CREDIT) as avg_loan FROM application_train"
                },
                {
                    "question": "Show default rate by education level",
                    "sql": """SELECT NAME_EDUCATION_TYPE, 
                              AVG(TARGET) * 100 as default_rate 
                              FROM application_train 
                              GROUP BY NAME_EDUCATION_TYPE 
                              ORDER BY default_rate DESC"""
                }
            ]
            rag_system.index_sql_examples(example_queries)
            
            logger.info("RAG system initialized with database schema and examples")
        except Exception as e:
            logger.error(f"Failed to initialize RAG data: {e}")
    
    def _get_main_keyboard(self):
        """Get main keyboard with Status and Help buttons"""
        keyboard = [
            [
                KeyboardButton("📊 Статус"),
                KeyboardButton("❓ Помощь")
            ]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command.
        
        Initializes user session and displays welcome message.
        """
        user = update.effective_user
        user_id = user.id
        
        self.user_sessions[user_id] = {
            "history": [],
            "context": {},
            "preferences": {}
        }
        
        welcome_message = f"""
**Добро пожаловать в SQL бот, {user.first_name}!** 🤖

Я AI-ассистент для анализа данных Home Credit. Просто задайте мне вопрос на естественном языке, и я:
• Сгенерирую и выполню SQL запрос
• Проанализирую результаты
• Создам визуализацию (если нужно)

**Примеры вопросов:**
• "Каков средний доход по одобренным займам?"
• "Покажи уровень дефолтов по полу"
• "Создай график сумм займов по уровню образования"

Просто напишите ваш вопрос! 💬
"""
        
        await safe_send_markdown(
            update.message.reply_text,
            welcome_message,
            reply_markup=self._get_main_keyboard()
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle query mode messages.
        
        Processes user queries through the orchestrator and sends results.
        Ignores messages older than 5 minutes to prevent processing stale messages.
        
        Args:
            update: Telegram update object.
            context: Bot context.
            
        """
        if update.message.date:
            message_age = (datetime.now(timezone.utc) - update.message.date).total_seconds()
            if message_age > 300:
                logger.info(f"Ignoring old message (age: {message_age:.0f}s)")
                return
        
        user_input = update.message.text
        user_id = update.effective_user.id
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "history": [],
                "context": {},
                "preferences": {}
            }
        
        logger.info(f"[TELEGRAM_BOT] [QUERY] Получен запрос от пользователя {user_id}: '{user_input[:80]}...'")
        
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
        
        try:
            logger.info(f"[TELEGRAM_BOT] [QUERY] → Вызов оркестратора для обработки запроса")
            result = await self.orchestrator.process_request(
                user_input=user_input,
                context=self.user_sessions[user_id].get("context", {})
            )
            logger.info(f"[TELEGRAM_BOT] [QUERY] Оркестратор вернул результат: success={result.success}, task_type={result.task_type.value}")
            
            await self._send_task_result(update, context, result)
            
            self.user_sessions[user_id]["history"].append({
                "timestamp": datetime.now().isoformat(),
                "input": user_input,
                "result": result
            })
            
            if result.sql_result and result.sql_result.query:
                self.user_sessions[user_id]["context"]["last_sql"] = result.sql_result.query
            
        except Exception as e:
            logger.error(f"[TELEGRAM_BOT] [QUERY] ОШИБКА при обработке запроса: {e}", exc_info=True)
            error_msg = escape_markdown(str(e))
            await update.message.reply_text(
                f"Ошибка при обработке запроса: {error_msg}\n\n"
                "Попробуйте переформулировать ваш вопрос или используйте /help для помощи."
            )
    
    async def handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button presses (Status, Help)"""
        user_input = update.message.text
        user_id = update.effective_user.id
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "history": [],
                "context": {},
                "preferences": {}
            }
        
        if user_input == "📊 Статус" or user_input == "Статус":
            await self.show_status(update, context)
        elif user_input == "❓ Помощь" or user_input == "Помощь":
            await self.show_help(update, context)
        else:
            await self.handle_message(update, context)
    
    async def _send_task_result(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        result: TaskResult
    ):
        """Send task result to user.
        
        Formats and sends task results in a clean, unified format.
        Combines explanation, data, and visualization into a single message flow.
        
        Args:
            update: Telegram update object.
            context: Bot context.
            result: TaskResult from orchestrator.
        """
        logger.info(f"[TELEGRAM_BOT] Отправка результата задачи: success={result.success}, task_type={result.task_type.value}")
        
        if not result.success:
            logger.warning(f"[TELEGRAM_BOT] Результат содержит ошибку: {result.error}")
            error_msg = escape_markdown(result.error) if result.error else "Неизвестная ошибка"
            await update.message.reply_text(f"❌ Ошибка: {error_msg}")
            return
        
        message_parts = []
        reply_markup = None
        df = None
        
        if result.explanation:
            message_parts.append(result.explanation)
        
        if result.sql_result and result.sql_result.data is not None:
            df = result.sql_result.data
            if not df.empty:
                sql_upper = result.sql_result.query.upper() if result.sql_result.query else ""
                is_simple_aggregate = (
                    len(df) == 1 and 
                    len(df.columns) <= 2 and 
                    any(kw in sql_upper for kw in ['AVG(', 'SUM(', 'COUNT(', 'MAX(', 'MIN(']) and
                    'GROUP BY' not in sql_upper
                )
                
                if not is_simple_aggregate and len(df) <= 10:
                    preview = df.to_string(index=False)
                    if len(preview) <= 2000:
                        message_parts.append(f"```\n{preview}\n```")
                    else:
                        preview = df.head(5).to_string(index=False)
                        message_parts.append(f"```\n{preview}\n... (показано 5 из {len(df)} строк)\n```")
                elif len(df) > 10:
                    preview = df.head(5).to_string(index=False)
                    message_parts.append(f"```\n{preview}\n... (показано 5 из {len(df)} строк)\n```")
                    
                    keyboard = [[
                        InlineKeyboardButton(
                            f"📥 Скачать все ({len(df)} строк)",
                            callback_data="download_csv"
                        )
                    ]]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    context.user_data['last_df'] = df
        
        def prepare_html(text: str) -> str:
            """Clean text and ensure proper HTML formatting for Telegram"""
            if not text:
                return text
            
            text = re.sub(r'\\([*_`\[\]()~>#+\-=|{}.!])', r'\1', text)
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
            text = re.sub(r'###\s+(.+?)(?=\n|$)', r'<b>\1</b>', text, flags=re.MULTILINE)
            text = re.sub(r'##\s+(.+?)(?=\n|$)', r'<b>\1</b>', text, flags=re.MULTILINE)
            text = re.sub(r'#\s+(.+?)(?=\n|$)', r'<b>\1</b>', text, flags=re.MULTILINE)
            text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
            text = re.sub(r'```([^`]+)```', r'<pre>\1</pre>', text)
            text = re.sub(r'^\*\s+', '• ', text, flags=re.MULTILINE)
            
            text = text.replace('&', '&amp;')
            text = text.replace('<', '&lt;')
            text = text.replace('>', '&gt;')
            
            text = text.replace('&lt;b&gt;', '<b>')
            text = text.replace('&lt;/b&gt;', '</b>')
            text = text.replace('&lt;i&gt;', '<i>')
            text = text.replace('&lt;/i&gt;', '</i>')
            text = text.replace('&lt;code&gt;', '<code>')
            text = text.replace('&lt;/code&gt;', '</code>')
            text = text.replace('&lt;pre&gt;', '<pre>')
            text = text.replace('&lt;/pre&gt;', '</pre>')
            
            return text
        
        if message_parts:
            message_parts = [prepare_html(part) for part in message_parts]
            combined_message = "\n\n".join(message_parts)
            if len(combined_message) > 4000:
                for i, part in enumerate(message_parts):
                    if len(part) > 4000:
                        chunks = [part[j:j+4000] for j in range(0, len(part), 4000)]
                        for chunk in chunks:
                            try:
                                await update.message.reply_text(chunk, parse_mode="HTML")
                            except Exception as e:
                                logger.warning(f"Failed to send with HTML: {e}")
                                await update.message.reply_text(chunk)
                    else:
                        use_markup = reply_markup if (i == len(message_parts) - 1) else None
                        try:
                            await update.message.reply_text(part, parse_mode="HTML", reply_markup=use_markup)
                        except Exception as e:
                            logger.warning(f"Failed to send with HTML: {e}")
                            await update.message.reply_text(part, reply_markup=use_markup)
            else:
                try:
                    await update.message.reply_text(combined_message, parse_mode="HTML", reply_markup=reply_markup)
                except Exception as e:
                    logger.warning(f"Failed to send with HTML: {e}")
                    await update.message.reply_text(combined_message, reply_markup=reply_markup)
        
        if result.visualization_result and result.visualization_result.success:
            logger.info(f"[TELEGRAM_BOT] Отправка визуализации: type={result.visualization_result.chart_type}")
            if result.visualization_result.image_data:
                caption = result.visualization_result.description or "📊 Визуализация данных"
                if len(caption) > 200:
                    caption = caption[:200] + "..."
                await update.message.reply_photo(
                    photo=io.BytesIO(result.visualization_result.image_data),
                    caption=caption
                )
    
    async def download_csv(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle CSV download request.
        
        Generates and sends a CSV file containing the full query results
        stored in user context.
        
        Args:
            update: Telegram update object.
            context: Bot context containing last_df.
        """
        query = update.callback_query
        await query.answer()
        
        if 'last_df' in context.user_data:
            df = context.user_data['last_df']
            
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            await context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=io.BytesIO(csv_buffer.getvalue().encode()),
                filename=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                caption="Результаты запроса в формате CSV"
            )
        else:
            await query.edit_message_text("Нет данных для скачивания.")
    
    async def show_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help information.
        
        Displays comprehensive help documentation including available commands,
        usage tips, and example queries.
        
        Args:
            query: Callback query object.
        """
        help_text = """
**Справка и документация** 📚

**Как использовать:**
Просто напишите ваш вопрос на естественном языке! Бот автоматически:
• Поймет ваш запрос
• Сгенерирует и выполнит SQL
• Проанализирует результаты
• Создаст визуализацию (если нужно)

**Доступные команды:**
• `/start` - Перезапустить бота
• `/help` - Показать эту справку
• `/status` - Проверить статус системы

**Советы для лучших результатов:**
• Будьте конкретны в ваших вопросах
• Используйте описательные термины: "среднее", "тренд", "сравнить"
• Запрашивайте визуализацию: "и визуализируй", "создай график"

**Примеры запросов:**
• "Каков средний доход по одобренным займам?"
• "Покажи уровень дефолтов по полу и визуализируй"
• "Создай график сумм займов по уровню образования"
• "Проанализируй связь между доходом и дефолтом"

Просто напишите ваш вопрос! 💬
"""
        
        await safe_send_markdown(
            update.message.reply_text,
            help_text,
            reply_markup=self._get_main_keyboard()
        )
    
    async def show_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show system status.
        
        Displays current system status including agent status, RAG system info,
        database connection, and configuration details.
        
        Args:
            query: Callback query object.
        """
        try:
            status = await self.orchestrator.get_agent_status()
            
            status_text = f"""
**Статус системы** 📊

**Оркестратор:** {status.get('orchestrator', 'unknown')}
**Пайплайн:** {status.get('pipeline', 'unknown')}
**Асинхронность:** {'✅' if status.get('async') else '❌'}

**Агенты:**
"""
            for agent_name, agent_info in status.get('agents', {}).items():
                agent_type = agent_info.get('type', '')
                status_text += f"• {agent_name}: {agent_info.get('status', 'unknown')}"
                if agent_type:
                    status_text += f" ({agent_type})"
                status_text += "\n"
            
            status_text += f"\n**База данных:**\n"
            db_info = status.get('database', {})
            status_text += f"• Подключение: {'✅' if db_info.get('connected') else '❌'}\n"
            status_text += f"• Диалект: {db_info.get('dialect', 'unknown')}\n"
            status_text += f"• Таблиц: {db_info.get('tables', 0)}\n"
            
            status_text += f"\n**RAG система:**\n"
            rag_info = status.get('rag_system', {})
            status_text += f"• Статус: {rag_info.get('status', 'unknown')}\n"
            status_text += f"• Тип: {rag_info.get('type', 'unknown')}\n"
            
            status_text += f"\n**Конфигурация:**\n"
            status_text += f"• LLM провайдер: {settings.llm_provider}\n"
            status_text += f"• Модель: {settings.llm_model}\n"
        except Exception as e:
            status_text = f"Ошибка при получении статуса: {str(e)}"
        
        await safe_send_markdown(
            update.message.reply_text,
            status_text,
            reply_markup=self._get_main_keyboard()
        )
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors.
        
        Logs errors and sends a user-friendly error message.
        
        Args:
            update: Telegram update object (may be None).
            context: Bot context containing error information.
        """
        logger.error(f"Update {update} caused error {context.error}")
        
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Произошла ошибка. Пожалуйста, попробуйте снова или используйте /help для помощи."
            )
        except:
            pass
    
    def run(self):
        """Run the bot.
        
        Sets up command handlers and message handlers,
        then starts polling for updates from Telegram.
        """
        application = Application.builder().token(settings.telegram_token).build()
        
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.show_help))
        application.add_handler(CommandHandler("status", self.show_status))
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_button
        ))
        application.add_handler(CallbackQueryHandler(self.download_csv, pattern="^download_csv$"))
        
        application.add_error_handler(self.error_handler)
        
        logger.info("Starting HomeCreditBot with simplified UI...")
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    bot = HomeCreditBot()
    bot.run()
