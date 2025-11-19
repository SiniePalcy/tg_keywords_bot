import asyncio
import logging
import os
import re
import seqlog
import openai
import numpy as np
from telethon import TelegramClient, events
from telethon.errors import PeerFloodError
from telethon.tl.types import User, MessageEntityTextUrl
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import defaultdict
from zoneinfo import ZoneInfo
from typing import Optional

user_message_cache = defaultdict(list)
last_sent: dict[int, datetime] = {}

seqlog.log_to_seq(
    server_url="http://127.0.0.1:5341",
    api_key=None,  # set if you created an API key in Seq
    level=logging.INFO,
    auto_flush_timeout=1,
    override_root_logger=True,  # attach handler to root logger
)

load_dotenv()

api_id_str = os.getenv("API_ID")
if api_id_str is None:
    raise ValueError("API_ID not set")
api_id = int(api_id_str)
api_hash = os.getenv("API_HASH")
session_name = "keyword_alert_notification"
openai.api_key = os.getenv("OPENAI_API_KEY")

ENABLE_SEMANTIC_FILTER = os.getenv("ENABLE_SEMANTIC_FILTER", "False").lower() == "true"
DELAY_BETWEEN_MESSAGES = 0.5
DELAY_TOO_MANY_REQUESTS = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFER_IMAGE_PATH = os.path.join(BASE_DIR, "transfer.jpg")

CONFIGS = [
    {
        "chats": {-1001954706166, -1001676333024, -1001214960694, -1001850398389},
        "keywords": [
            "ищу",
            "ищем",
            "ищет",
            "нужна",
            "нужен",
            "нужно",
            "кто",
            "кто-нибудь",
            "требуется",
            "сколько",
        ],
        "excluded_keywords": [
            "usdt",
            "визаран",
            "виза ран",
            "визоран",
            "визо ран",
            "доход",
            "подработка",
            "удаленный",
            "удаленно",
            "удаленная",
            "работа",
            "занятость",
            "шкипер",
            "тирана",
            "тираны",
            "тирану",
            "дубровник",
            "дубровника",
            "требинье",
            "босния",
            "боснии",
            "боснию",
            "албания",
            "албанию",
            "албании",
            "хорватия",
            "хорватию",
            "хорватии",
            "херцег-нови",
            "херцегнови",
            "херцег",
            "герцег",
            "герцегнови",
            "херцог",
            "герцог",
        ],
        "excluded_senders": [],
        "recipient": 6472110264,
        "include_questions": True,
    }
]

PERIOD_MINUTES = 5

client = TelegramClient(session_name, api_id, api_hash)

openAIclient = openai.AsyncOpenAI()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\(\)\[\]\{\}]", "", text)
    text = re.sub(r"[^a-яa-z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def getnow() -> datetime:
    return datetime.now(ZoneInfo("Europe/Podgorica"))


def add_to_user_cache(user_id: int, raw_text: str) -> None:
    normalized = normalize_text(raw_text)
    now = getnow()
    user_message_cache[user_id].append((normalized, now))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


async def get_embedding(text: str) -> np.ndarray:
    response = await openAIclient.embeddings.create(
        input=text, model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)


async def is_semantically_duplicate(user_id: int, text: str) -> bool:
    try:
        new_embedding = np.array(await get_embedding(text))

        for prev_text, _ in user_message_cache[user_id]:
            prev_embedding = np.array(await get_embedding(prev_text))
            sim = cosine_similarity(new_embedding, prev_embedding)
            if sim > 0.9:
                logging.info(f"🔁 Семантический дубликат от пользователя {user_id}")
                return True
    except Exception as e:
        logging.warning(f"Ошибка при семантическом сравнении: {e}")
    return False


async def send_message_safe(recipient: int, message: str) -> None:
    now = getnow()
    if recipient in last_sent and now - last_sent[recipient] < timedelta(seconds=60):
        print(f"Too soon to message {recipient}")
        return

    try:
        await client.send_message(recipient, message, parse_mode="markdown")
        last_sent[recipient] = now
    except PeerFloodError:
        print("Hit PeerFloodError — backing off")
        await asyncio.sleep(DELAY_TOO_MANY_REQUESTS)


async def handle_transfer_offer(event: events.NewMessage.Event, raw_text: str, prefix_used: str) -> None:
    rest = raw_text[len(prefix_used):].strip(" :,-")

    if not event.is_reply:
        await event.reply("Команда должна быть ответом на сообщение бота с метаданными 🙂")
        return

    reply_msg = await event.get_reply_message()
    if reply_msg is None:
        await event.reply("Не удалось получить сообщение, на которое вы отвечали.")
        return

    target_user_id: Optional[int] = None

    entities = getattr(reply_msg, "entities", None)
    if entities:
        for ent in entities:
            if isinstance(ent, MessageEntityTextUrl) and ent.url and ent.url.startswith("tg://user?id="):
                try:
                    target_user_id = int(ent.url.split("=", 1)[1])
                except ValueError:
                    logging.warning(f"Не удалось распарсить user id из url: {ent.url}")
                break

    if target_user_id is None:
        match = re.search(r"tg://user\?id=(\d+)", reply_msg.raw_text or "")
        if match is not None:
            target_user_id = int(match.group(1))

    if target_user_id is None:
        await event.reply(
            "Не нашёл пользователя в тексте уведомления. "
            "Проверьте, что вы отвечаете на сообщение бота, а не на другое."
        )
        return

    if not rest:
        await event.reply("Добавьте описание: например `предложи попутку Бар — Будва`")
        return

    if prefix_used.startswith("предложи попутку") or prefix_used.startswith("предложить попутку"):
        caption = f"Здравствуйте. Могу предложить вам попутный трансфер {rest}."
    else:
        caption = (
            f"Здравствуйте. Могу предложить трансфер {rest}. "
            f"Машина 2019 года, кондиционер, багажник 400 литров, хетчбек. "
            f"В салоне не курят. Включаю музыку по запросу, работает CarPlay."
        )

    try:
        await client.send_file(
            target_user_id,
            TRANSFER_IMAGE_PATH,
            caption=caption,
        )
        await event.reply("✅ Отправлено.")
    except Exception as e:
        logging.exception(e)
        await event.reply("Не получилось отправить сообщение 😔")


@client.on(events.NewMessage)
async def handler(event: events.NewMessage.Event) -> None:
    sender = await event.get_sender()
    if not isinstance(sender, User) or sender.bot:
        return

    sender_id = sender.id
    raw_text = (event.raw_text or "").strip()

    alert_recipients = {
        cfg["recipient"]
        for cfg in CONFIGS
        if isinstance(cfg.get("recipient"), int)
    }

    if sender_id in alert_recipients and event.is_reply and raw_text:
        lower = raw_text.lower()
        prefixes = (
            "предложи трансфер",
            "предложить трансфер",
            "предложи попутку",
            "предложить попутку"
        )

        prefix_used = next((p for p in prefixes if lower.startswith(p)), None)
        if prefix_used:
            await handle_transfer_offer(event, raw_text, prefix_used)
            return

    text = normalize_text(raw_text)

    config: dict[str, object]
    for config in CONFIGS:
        if isinstance(config["chats"], set) and event.chat_id not in config["chats"]:
            continue

        excluded_senders = config.get("excluded_senders", [])
        if isinstance(excluded_senders, list) and sender_id in excluded_senders:
            continue

        recent_messages = user_message_cache[sender_id]
        if any(prev_text == text for prev_text, _ in recent_messages):
            logging.info(f"⛔ Повтор от пользователя {sender_id}: {text}")
            continue

        keywords = config.get("keywords", [])
        matched = (
            any(word in text for word in keywords)
            if isinstance(keywords, list)
            else False
        )
        is_question = "?" in text
        if not (matched or (config.get("include_questions") and is_question)):
            continue

        excluded_keywords = config.get("excluded_keywords", [])
        if isinstance(excluded_keywords, list) and any(
            block_word in text for block_word in excluded_keywords
        ):
            logging.info(f"⛔ Игнор по слову для пользователя {sender_id}:{text}")
            continue

        now = getnow()
        if any(
            (now - ts) < timedelta(minutes=PERIOD_MINUTES) for _, ts in recent_messages
        ):
            logging.info(
                f"⏱️ Игнор: пользователь {sender_id} уже писал за последние {PERIOD_MINUTES} минут"
            )
            continue

        if ENABLE_SEMANTIC_FILTER and await is_semantically_duplicate(sender_id, text):
            logging.info(f"⛔ Игнор: пользователь {sender_id} уже писал об" f"этом")
            continue

        chat = await event.get_chat()
        chat_title = getattr(chat, "title", "")
        sender_name = getattr(sender, "first_name", "пользователь")
        sender_link = f"[{sender_name}](tg://user?id={sender_id})"
        message_link = None

        if hasattr(chat, "username") and chat.username:
            message_link = f"https://t.me/{chat.username}/{event.id}"

        logging.info(
            f"[🔔] Chat: {chat_title} | Sender: {sender_name} | Msg:{event.raw_text}"
        )

        message = (
            f"Сообщение в чате \"{chat_title}\" от {sender_link} в {now.strftime('%H:%M:%S')}:\n\n"
            f"{event.raw_text}"
        )

        if message_link:
            message += f"\n[Открыть сообщение]({message_link})"

        if isinstance(config["recipient"], int):
            await send_message_safe(config["recipient"], message)
        logging.info(
            f"Message sent: {message} | Sender: {sender_name} | Recipient: {config['recipient']}"
        )

        add_to_user_cache(sender_id, text)

        await asyncio.sleep(DELAY_BETWEEN_MESSAGES)


async def run_bot() -> None:
    await client.start()

    now = getnow().strftime("%d-%m-%Y %H:%M:%S")
    logging.info(f"🧾 Bot run at {now}")

    me = await client.get_me()
    logging.info(f"🧾 Signed in as {me.first_name} (bot={me.bot})")
    logging.info("✅ Bot is running. Waiting for messages...")

    await client.run_until_disconnected()


async def shutdown() -> None:
    now = getnow().strftime("%d-%m-%Y %H:%M:%S")
    logging.info(f"🧾 Bot stopped at {now}")
    await client.disconnect()


async def clear_cache_at_midnight() -> None:
    while True:
        now = getnow()
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        seconds_until_midnight = (tomorrow - now).total_seconds()

        logging.info(f"⏳ Clearing cache after {int(seconds_until_midnight)} seconds")
        await asyncio.sleep(seconds_until_midnight)

        user_message_cache.clear()
        last_sent.clear()
        logging.info("🧹 Message cache is cleared at midnight UTC")


async def main() -> None:
    try:
        asyncio.create_task(clear_cache_at_midnight())
        await run_bot()
    except (KeyboardInterrupt, SystemExit):
        logging.info("⚠️ KeyboardInterrupt — shutting down...")
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
