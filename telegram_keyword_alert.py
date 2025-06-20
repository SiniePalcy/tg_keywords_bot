import sys
import io
import asyncio
import os
import logging
import seqlog
from telethon import TelegramClient, events
from telethon.tl.types import User
from datetime import datetime, timezone, timedelta
from collections import deque
from dotenv import load_dotenv

seqlog.log_to_seq(
    server_url="http://localhost:5341",  # если бот и Seq на одном VPS
    api_key=None,
    level=logging.INFO
)

sent_messages_cache = {}

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

load_dotenv()

api_id = int(os.getenv("API_ID"))
api_hash = os.getenv("API_HASH")
session_name = 'keyword_alert_notification'

CONFIGS = [
    {
        "chats": {-1001954706166, -1001676333024, -1001214960694, -1001850398389},
        "keywords": ['ищу', 'ищем', 'ищет', 'нужна', 'нужен', 'нужно', 'кто', 'кто-нибудь'],
        "excluded_keywords": ['визаран', 'виза ран'],
        "excluded_senders": [],
        "recipient": 6472110264,
        "include_questions": True
    },
    {
        "chats": { -1001211521747, -1001609324023 },
        "keywords": ['пианино', 'синтезатор'],
        "excluded_keywords": [],
        "excluded_senders": [7176393076],
        "recipient": 418176416,
        "include_questions": False
    }
]

client = TelegramClient(session_name, api_id, api_hash)

@client.on(events.NewMessage)
async def handler(event):
    sender = await event.get_sender()
    if not isinstance(sender, User) or sender.bot:
        return
    
    sender_id = sender.id
    text = event.raw_text.lower().strip()

    now = datetime.now(timezone.utc)
    message_key = (sender_id, text.strip())

    if message_key in sent_messages_cache:
        last_sent = sent_messages_cache[message_key]
        if last_sent.date() == now.date():
            return

    sent_messages_cache[message_key] = now

    for config in CONFIGS:
        if event.chat_id not in config["chats"]:
            continue

        if sender_id in config.get("excluded_senders", []):
            continue

        matched = any(word in text for word in config["keywords"])
        is_question = '?' in text
        if not (matched or (config.get("include_questions") and is_question)):
            continue

        if any(block_word in text for block_word in config.get("excluded_keywords", [])):
            continue

        chat = await event.get_chat()

        chat_title = getattr(chat, 'title', '')
        sender_name = getattr(sender, 'first_name', '')
        message_link = None

        if hasattr(chat, 'username') and chat.username:
            message_link = f"https://t.me/{chat.username}/{event.id}"

        logging.info(f"[🔔] Chat: {chat_title} | Sender: {sender_name} | Msg: {event.raw_text}")

        message = (
            f"Важное сообщение в чате \"{chat_title}\":\n\n"
            f"{event.raw_text}"
        )

        if message_link:
            message += f"\n\n🔗 [Открыть сообщение]({message_link})"

        await client.send_message(config["recipient"], message, parse_mode='markdown')

async def run_bot():
    await client.start()

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    await client.send_message(CONFIGS[0]["recipient"], f"🚀 Бот запущен в {now}")

    me = await client.get_me()
    logging.info(f"🧾 Signed in as {me.first_name} (bot={me.bot})")
    logging.info("✅ Bot is running. Waiting for messages...")

    await client.run_until_disconnected()

async def shutdown():
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    try:
        await client.send_message(CONFIGS[0]["recipient"], f"🛑 Бот остановлен в {now}")
    except Exception as e:
            logging.error(f"[ERROR] Failed to send stop message: {e}")
    await client.disconnect()

async def clear_cache_at_midnight():
    while True:
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_until_midnight = (tomorrow - now).total_seconds()

        logging.info(f"⏳ Clearing cache after {int(seconds_until_midnight)} seconds")
        await asyncio.sleep(seconds_until_midnight)

        sent_messages_cache.clear()
        logging.info("🧹 Message cache is cleared at midnight UTC")

async def main():
    try:
        asyncio.create_task(clear_cache_at_midnight())
        await run_bot()
    except (KeyboardInterrupt, SystemExit):
        logging.info("⚠️ KeyboardInterrupt — shutting down...")
        await shutdown()


if __name__ == '__main__':
    asyncio.run(main())
