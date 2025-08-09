from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Dict

from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, BotCommand
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from app.persona import build_system_prompt, get_bot_name
from app.memory import InMemoryChatStore
from app.llm import LLMClient


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("nours-bot")

chat_store = InMemoryChatStore()
llm_client = None  # lazy init to fail fast with clear error when first used


async def send_typing_action(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    except Exception:  # do not crash on typing errors
        pass


async def send_typing_action_periodic(context: ContextTypes.DEFAULT_TYPE, chat_id: int, stop_event: asyncio.Event) -> None:
    try:
        while not stop_event.is_set():
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(3)
    except Exception:
        pass


def ensure_llm() -> LLMClient:
    global llm_client
    if llm_client is None:
        llm_client = LLMClient()
    return llm_client


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_name = get_bot_name()
    user_first_name = update.effective_user.first_name if update.effective_user else "صديقي"
    greeting = (
        f"أهلاً {user_first_name}! أنا {bot_name} 🤗\n"
        "صديقك الرقمي المرح — أسعد بمساعدتك في أي شيء: أسئلة عامة، كتابة نصوص إبداعية، تلخيصات، أفكار مشاريع، وحتى تنظيم يومك!\n"
        "ما أول شيء تحب نبدأ به اليوم؟ 😊"
    )
    keyboard = ReplyKeyboardMarkup(
        [[KeyboardButton("✍️ اكتب لي رسالة"), KeyboardButton("💡 اقترح فكرة")],
         [KeyboardButton("🧠 لخص هذا"), KeyboardButton("🧹 اعادة ضبط الذاكرة")] ],
        resize_keyboard=True,
        one_time_keyboard=False,
    )
    await update.message.reply_text(greeting, reply_markup=keyboard)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_name = get_bot_name()
    help_text = (
        f"أنا {bot_name} — روبوت دردشة ذكي وودود 😄\n\n"
        "أستطيع:\n"
        "- الإجابة عن الأسئلة في مجالات متعددة.\n"
        "- كتابة نصوص إبداعية (قصائد، قصص، نصوص أغاني وإعلانات).\n"
        "- توليد أفكار واقتراحات للعصف الذهني.\n"
        "- المساعدة اليومية: تنظيم الجداول، صياغة الرسائل، تلخيص المقالات.\n\n"
        "أوامر سريعة:\n"
        "/start — بدء محادثة جديدة\n"
        "/help — هذه المساعدة\n"
        "/reset — نسيان سياق المحادثة الحالي\n\n"
        "جاهز دائماً — ماذا يدور في بالك الآن؟ 💡"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        chat_store.reset(update.effective_chat.id)
    await update.message.reply_text("تمت إعادة ضبط الذاكرة لهذه المحادثة. لنبدأ من جديد! ✨")


def _build_history_for_llm(chat_id: int) -> List[Dict[str, str]]:
    return chat_store.get_history(chat_id)


async def on_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()

    if user_text == "🧹 اعادة ضبط الذاكرة":
        await cmd_reset(update, context)
        return

    # Store user message
    chat_store.add_user_message(chat_id, user_text)

    # Periodic typing while generating
    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(send_typing_action_periodic(context, chat_id, stop_event))

    # Build prompt and history
    system_prompt = build_system_prompt(update.effective_user.first_name if update.effective_user else None)
    history_messages = _build_history_for_llm(chat_id)

    try:
        reply_text = await asyncio.to_thread(ensure_llm().generate, system_prompt, history_messages)
    except Exception as exc:
        logger.exception("LLM generation failed: %s", exc)
        reply_text = (
            "همم… يبدو أن هناك مشكلة تقنية صغيرة الآن 😅. سأحاول مجدداً بعد لحظات."
        )
    finally:
        stop_event.set()
        await typing_task

    # Save assistant reply
    chat_store.add_assistant_message(chat_id, reply_text)

    await update.message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN)


async def main_async() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Please configure your environment.")

    application = (
        Application.builder()
        .token(token)
        .concurrent_updates(True)
        .build()
    )

    # Bot menu commands
    await application.bot.set_my_commands([
        BotCommand("start", "بدء محادثة جديدة"),
        BotCommand("help", "المساعدة"),
        BotCommand("reset", "إعادة ضبط الذاكرة"),
    ])

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("reset", cmd_reset))

    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), on_text_message))

    logger.info("Bot is starting…")
    await application.initialize()
    await application.start()
    logger.info("Bot is up. Listening for messages…")

    try:
        await application.updater.start_polling()
        await asyncio.Event().wait()
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass