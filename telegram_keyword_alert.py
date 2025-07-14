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
from telethon.errors import PeerFloodError
from telethon.tl.types import User
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import defaultdict
from zoneinfo import ZoneInfo

user_message_cache = defaultdict(list)
last_sent = {}

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

ENABLE_SEMANTIC_FILTER = os.getenv("ENABLE_SEMANTIC_FILTER", "False").lower() == "true"
DELAY_BETWEEN_MESSAGES = 1
DELAY_TOO_MANY_REQUESTS = 120

CONFIGS = [
    {
        "chats": {-1001954706166, -1001676333024, -1001214960694, -1001850398389},
        "keywords": ['ищу', 'ищем', 'ищет', 'нужна', 'нужен', 'нужно', 'кто', 'кто-нибудь'],
        "excluded_keywords": [
            'визаран', 'виза ран', 'визоран', 'визо ран',
            'доход', 'подработка', "удаленный", "удаленно", "удаленная", "работа", "занятость", "шкипер"
            'тирана', 'тираны', 'тирану',
            'дубровник', 'дубровника',
            'требинье',
            'босния', 'боснии', 'боснию',
            'белград', 'белграда',
            'албания', 'албанию', 'албании',
            'хорватия', 'хорватию', 'хорватии',
            'сербия', 'сербию', 'сербии',
            'херцег-нови',  'херцегнови',  'херцег', 'герцег',  'герцегнови',  'херцог',  'герцог'
        ],
        "excluded_senders": [],
        "recipient": 6472110264,
        "include_questions": True
    }
    # {
    #    "chats": { -1001211521747, -1001609324023 },
    #    "keywords": ['кровать', 'кресло', 'ковер', 'ковёр'],
    #    "excluded_keywords": ['автокресло'],
    #    "excluded_senders": [7176393076],
    #    "recipient": 418176416,
    #    "include_questions": False
    # }
]

PERIOD_MINUTES = 5

client = TelegramClient(session_name, api_id, api_hash)

openAIclient = openai.AsyncOpenAI()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[\(\)\[\]\{\}]', '', text)
    text = re.sub(r'[^а-яa-z0-9 ]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def getnow():
    return datetime.now(ZoneInfo("Etc/GMT-2"))


def add_to_user_cache(user_id: int, raw_text: str):
    normalized = normalize_text(raw_text)
    now = getnow()
    user_message_cache[user_id].append((normalized, now))


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


async def get_embedding(text: str) -> list[float]:
    response = await openAIclient.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


async def is_semantically_duplicate(user_id, text: str) -> bool:
    try:
        new_embedding = await get_embedding(text)

        for prev_text, _ in user_message_cache[user_id]:
            prev_embedding = await get_embedding(prev_text)
            sim = cosine_similarity(new_embedding, prev_embedding)
            if sim > 0.9:
                logging.info(f"🔁 Семантический дубликат от пользователя"
                f"{user_id}")
                return True
    except Exception as e:
        logging.warning(f"Ошибка при семантическом сравнении: {e}")
    return False


async def send_message_safe(recipient, message):
    now = getnow()
    if recipient in last_sent and now - last_sent[recipient] < timedelta(seconds=60):
        print(f"Too soon to message {recipient}")
        return

    try:
        await client.send_message(recipient, message, parse_mode='markdown')
        last_sent[recipient] = now
    except PeerFloodError:
        print("Hit PeerFloodError — backing off")
        await asyncio.sleep(DELAY_TOO_MANY_REQUESTS)


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

        recent_messages = user_message_cache[sender_id]
        if any(prev_text == text for prev_text, _ in recent_messages):
            logging.info(f"⛔ Повтор от пользователя {sender_id}: {text}")
            continue

        matched = any(word in text for word in config["keywords"])
        is_question = '?' in text
        if not (matched or (config.get("include_questions") and is_question)):
            continue

        if any(block_word in text for block_word in config.get("excluded_keywords", [])):
            logging.info(f"⛔ Игнор по слову для пользователя {sender_id}:"
            f"{text}")
            continue

        now = getnow()
        if any((now - ts) < timedelta(minutes=PERIOD_MINUTES) for _, ts in recent_messages):
            logging.info(f"⏱️ Игнор: пользователь {sender_id} уже писал за"
            f"последние 5 минут")
            continue

        if ENABLE_SEMANTIC_FILTER and await is_semantically_duplicate(sender_id, text):
            logging.info(f"⛔ Игнор: пользователь {sender_id} уже писал об"
            f"этом")
            continue

        chat = await event.get_chat()
        chat_title = getattr(chat, 'title', '')
        sender_name = getattr(sender, 'first_name', 'пользователь')
        sender_link = f"[{sender_name}](tg://user?id={sender_id})"
        message_link = None

        if hasattr(chat, 'username') and chat.username:
            message_link = f"https://t.me/{chat.username}/{event.id}"

        logging.info(f"[🔔] Chat: {chat_title} | Sender: {sender_name} | Msg:"
        f"{event.raw_text}")

        message = (
            f"Cообщение в чате \"{chat_title}\" от {sender_link} в {now.strftime('%H:%M:%S')}:\n\n"
            f"{event.raw_text}"
        )

        if message_link:
            message += f"\n🔗 [Открыть сообщение]({message_link})"

        await asyncio.sleep(DELAY_BETWEEN_MESSAGES)
        await send_message_safe(config["recipient"], message)
        logging.info(f"Message sent: {message} | Sender: {sender_name} |"
        f"Recipient: {config['recipient']}")

        add_to_user_cache(sender_id, text)


async def run_bot():
    await client.start()

    now = getnow().strftime("%d-%m-%Y %H:%M:%S")
    logging.info(f"🧾 Bot run at {now}")

    me = await client.get_me()
    logging.info(f"🧾 Signed in as {me.first_name} (bot={me.bot})")
    logging.info("✅ Bot is running. Waiting for messages...")

    await client.run_until_disconnected()


async def shutdown():
    now = getnow().strftime("%d-%m-%Y %H:%M:%S")
    logging.info(f"🧾 Bot stopped at {now}")
    await client.disconnect()


async def clear_cache_at_midnight():
    while True:
        now = getnow()
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_until_midnight = (tomorrow - now).total_seconds()

        logging.info(f"⏳ Clearing cache after {int(seconds_until_midnight)}"
        f"seconds")
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
