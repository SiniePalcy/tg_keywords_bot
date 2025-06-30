import asyncio
import io
import logging
import os
import re
import seqlog
import sys
import openai
import numpy as np
from telethon import TelegramClient, events
from telethon.tl.types import User
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from collections import defaultdict

user_message_cache = defaultdict(list)

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[\(\)\[\]\{\}]', '', text)       
    text = re.sub(r'[^а-яa-z0-9 ]+', '', text)       
    text = re.sub(r'\s+', ' ', text)                 
    return text

seqlog.log_to_seq(
    server_url="http://localhost:5341",
    api_key=None,
    level=logging.INFO
)

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

load_dotenv()

api_id = int(os.getenv("API_ID"))
api_hash = os.getenv("API_HASH")
session_name = 'keyword_alert_notification'
openai.api_key = os.getenv("OPENAI_API_KEY")

CONFIGS = [
    {
        "chats": {-1001954706166, -1001676333024, -1001214960694, -1001850398389},
        "keywords": ['ищу', 'ищем', 'ищет', 'нужна', 'нужен', 'нужно', 'кто', 'кто-нибудь'],
        "excluded_keywords": ['визаран', 'виза ран'],
        "excluded_senders": [],
        "recipient": 1742604792,
        "include_questions": True
    }
]

PERIOD_MINUTES = 5

client = TelegramClient(session_name, api_id, api_hash)

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[\(\)\[\]\{\}]', '', text)    
    text = re.sub(r'[^а-яa-z0-9 ]+', '', text)   
    text = re.sub(r'\s+', ' ', text)              
    return text

def add_to_user_cache(user_id: int, raw_text: str):
    normalized = normalize_text(raw_text)
    now = datetime.now()
    user_message_cache[user_id].append((normalized, now))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def is_semantically_duplicate(user_id, text: str) -> bool:
    try:
        response = await openai.Embedding.acreate(
            input=[text],
            model="text-embedding-3-small"
        )
        new_embedding = response["data"][0]["embedding"]

        for prev in user_message_cache[user_id]:
            prev_response = await openai.Embedding.acreate(
                input=[prev],
                model="text-embedding-3-small"
            )
            prev_embedding = prev_response["data"][0]["embedding"]
            sim = cosine_similarity(new_embedding, prev_embedding)
            if sim > 0.9:  # порог можно подбирать
                logging.info(f"🔁 Семантический дубликат от пользователя {user_id}")
                return True
    except Exception as e:
        logging.warning(f"Ошибка при семантическом сравнении: {e}")
    return False


@client.on(events.NewMessage)
async def handler(event):
    sender = await event.get_sender()
    if not isinstance(sender, User) or sender.bot:
        return

    sender_id = sender.id
    text = normalize_text(event.raw_text)

    for config in CONFIGS:
        if event.chat_id not in config["chats"]:
            continue

        if sender_id in config.get("excluded_senders", []):
            continue

        if text in user_message_cache[sender_id]:
            logging.info(f"⛔ Повтор от пользователя {sender_id}: {text}")
            continue

        matched = any(word in text for word in config["keywords"])
        is_question = '?' in text
        if not (matched or (config.get("include_questions") and is_question)):
            continue

        if any(block_word in text for block_word in config.get("excluded_keywords", [])):
            logging.info(f"⛔ Игнор по слову для пользователя {sender_id}: {text}")
            continue

        now = datetime.now().strftime("%H:%M:%S")
        recent_messages = user_message_cache[sender_id]
        if any((now - ts) < timedelta(minutes=PERIOD_MINUTES) for _, ts in recent_messages):
            logging.info(f"⏱️ Игнор: пользователь {sender_id} уже писал за последние 5 минут")
            continue

        if await is_semantically_duplicate(sender_id, text):
            logging.info(f"⏱️ Игнор: пользователь {sender_id} уже писал об этом")
            continue

        chat = await event.get_chat()
        chat_title = getattr(chat, 'title', '')
        sender_name = getattr(sender, 'first_name', 'пользователь')
        sender_link = f"[{sender_name}](tg://user?id={sender_id})"
        message_link = None

        if hasattr(chat, 'username') and chat.username:
            message_link = f"https://t.me/{chat.username}/{event.id}"

        logging.info(f"[🔔] Chat: {chat_title} | Sender: {sender_name} | Msg: {event.raw_text}")

        message = (
            f"Cообщение в чате \"{chat_title}\" от {sender_link} в {now}:\n\n"
            f"{event.raw_text}"
        )

        if message_link:
            message += f"\n🔗 [Открыть сообщение]({message_link})"

        await asyncio.sleep(1)
        await client.send_message(config["recipient"], message, parse_mode='markdown')
        logging.info(f"Message sent: {message} | Sender: {sender_name} | Recipient: {config["recipient"]}");

        add_to_user_cache(sender_id, text)


async def run_bot():
    await client.start()

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    logging.info(f"🧾 Bot run at {now}")

    me = await client.get_me()
    logging.info(f"🧾 Signed in as {me.first_name} (bot={me.bot})")
    logging.info("✅ Bot is running. Waiting for messages...")

    await client.run_until_disconnected()

async def shutdown():
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    logging.info(f"🧾 Bot stopped at {now}")
    await client.disconnect()

async def clear_cache_at_midnight():
    while True:
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_until_midnight = (tomorrow - now).total_seconds()

        logging.info(f"⏳ Clearing cache after {int(seconds_until_midnight)} seconds")
        await asyncio.sleep(seconds_until_midnight)

        user_message_cache.clear()
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
