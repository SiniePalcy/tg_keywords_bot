import asyncio
import logging
import os
import re
import seqlog
import openai
import numpy as np

from telethon import TelegramClient, events
from telethon.errors import PeerFloodError
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import defaultdict
from zoneinfo import ZoneInfo
from typing import Optional

user_message_cache: dict[int, list[tuple[str, datetime]]] = defaultdict(list)
last_sent: dict[int, datetime] = {}
chat_title_cache: dict[int, str] = {}
chat_username_cache: dict[int, str] = {}

cache_lock = asyncio.Lock()
last_sent_lock = asyncio.Lock()
metrics_lock = asyncio.Lock()


class State:
    last_handler_start: Optional[datetime] = None


state = State()

load_dotenv()

api_id_str = os.getenv("API_ID")
if api_id_str is None:
    raise ValueError("API_ID not set")

api_id = int(api_id_str)
api_hash = os.getenv("API_HASH")
session_name = "keyword_alert_notification"

openai.api_key = os.getenv("OPENAI_API_KEY")
openAIclient = openai.AsyncOpenAI()

ENABLE_SEMANTIC_FILTER = os.getenv("ENABLE_SEMANTIC_FILTER", "False").lower() == "true"

DELAY_TOO_MANY_REQUESTS = 30
MIN_SECONDS_BETWEEN_NOTIFICATIONS = int(os.getenv("MIN_SECONDS_BETWEEN_NOTIFICATIONS", "3"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFER_IMAGE_PATH = os.path.join(BASE_DIR, "transfer.jpg")
SEQ_URL = os.getenv("SEQ_URL")

if SEQ_URL:
    seqlog.log_to_seq(
        server_url=SEQ_URL,
        api_key=None,
        level=logging.INFO,
        auto_flush_timeout=1,
        override_root_logger=True,
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

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
            "кто-то",
            "ктото",
            "попутку",
            "кто-нибудь",
            "ктонибудь",
            "требуется",
            "сколько",
        ],
        "excluded_keywords": [
            "usdt",
            "вакансия",
            "₽",
            "работник",
            "сотрудник",
            "дружный коллектив",
            "стабильный график",
            "свободный график",
            "опыт",
            "рублей",
            "визаран",
            "виза ран",
            "визоран",
            "визо ран",
            "доход",
            "подработка",
            "удаленный",
            "удаленно",
            "удаленная",
            "удалёнка",
            "удаленка",
            "удалёнку",
            "удаленку",
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
            "москва",
            "москву",
            "москвы",
            "питер",
            "питера"
        ],
        "excluded_senders": [],
        "recipient": 6472110264,
        "include_questions": True,
    },
    {
        "chats": {-1001211521747, -1001609324023, -1001860403939, -1001201487135},
        "keywords": [
            "комод",
            "икеа",
            "икея",
            "ikea",
            "зеркало",
            "шезлонг",
            "шезлонг-качалка",
            "качалка",
            "кроватка",
            "кроватку"
        ],
        "excluded_keywords": [],
        "excluded_senders": [7176393076],
        "recipient": 418176416,
        "include_questions": False,
    }
]

PERIOD_MINUTES = 5

client = TelegramClient(
    session_name,
    api_id,
    api_hash,
    auto_reconnect=True,
    connection_retries=-1,
    retry_delay=2,
    sequential_updates=False,
)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\(\)\[\]\{\}]", "", text)
    text = re.sub(r"[^a-яa-z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def getnow() -> datetime:
    return datetime.now(ZoneInfo("Europe/Podgorica"))


async def get_recent_messages(user_id: int) -> list[tuple[str, datetime]]:
    async with cache_lock:
        return list(user_message_cache[user_id])


async def add_to_user_cache(user_id: int, normalized_text: str) -> None:
    async with cache_lock:
        user_message_cache[user_id].append((normalized_text, getnow()))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


async def get_embedding(text: str) -> np.ndarray:
    response = await openAIclient.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return np.array(response.data[0].embedding)


async def is_semantically_duplicate(user_id: int, text: str) -> bool:
    try:
        new_embedding = await get_embedding(text)
        recent_messages = await get_recent_messages(user_id)

        for prev_text, _ in recent_messages:
            prev_embedding = await get_embedding(prev_text)
            sim = cosine_similarity(new_embedding, prev_embedding)

            if sim > 0.9:
                logging.info("🔁 Семантический дубликат от пользователя %s", user_id)
                return True

    except Exception:
        logging.exception("Ошибка при семантическом сравнении")

    return False


async def send_message_safe(recipient: int, message: str) -> bool:
    started_at = getnow()

    async with last_sent_lock:
        now = getnow()

        if (
            MIN_SECONDS_BETWEEN_NOTIFICATIONS > 0
            and recipient in last_sent
            and now - last_sent[recipient] < timedelta(seconds=MIN_SECONDS_BETWEEN_NOTIFICATIONS)
        ):
            logging.info("Too soon to message %s", recipient)
            return False

        last_sent[recipient] = now

    try:
        logging.info("send_message_safe: sending to %s", recipient)
        await client.send_message(recipient, message, parse_mode="markdown")

        logging.info(
            "send_message_safe: sent to %s in %.3fs",
            recipient,
            (getnow() - started_at).total_seconds(),
        )
        return True

    except PeerFloodError:
        logging.warning("Hit PeerFloodError — backing off")
        await asyncio.sleep(DELAY_TOO_MANY_REQUESTS)
        return False

    except Exception:
        logging.exception("Failed to send message to %s", recipient)
        return False


async def handle_transfer_offer(
    event: events.NewMessage.Event,
    raw_text: str,
    prefix_used: str,
) -> None:
    rest = raw_text[len(prefix_used):].strip(" :,-")

    if not event.is_reply:
        await event.reply("Команда должна быть ответом на сообщение бота с метаданными")
        return

    reply_msg = await event.get_reply_message()
    if reply_msg is None:
        await event.reply("Не удалось получить сообщение, на которое вы отвечали.")
        return

    reply_text = reply_msg.raw_text or ""
    match = re.search(r"USER_ID:(\d+)", reply_text)

    if match is None:
        await event.reply("Не нашёл USER_ID в уведомлении.")
        return

    target_user_id = int(match.group(1))

    if not rest:
        await event.reply("Добавьте описание: например `предложи попутку Бар — Будва`")
        return

    is_carpool = prefix_used.startswith("предложи попутку") or prefix_used.startswith(
        "предложить попутку"
    )

    try:
        if is_carpool:
            caption = f"Здравствуйте. Могу предложить вам попутный трансфер {rest}."
            await client.send_message(target_user_id, caption)
        else:
            caption = (
                f"Здравствуйте. Могу предложить трансфер {rest}. "
                f"Машина 2019 года, кондиционер, багажник 400 литров, хетчбек. "
                f"В салоне не курят. Включаю музыку по запросу, работает CarPlay."
            )
            await client.send_file(target_user_id, TRANSFER_IMAGE_PATH, caption=caption)

        await event.reply("Отправлено")

    except Exception:
        logging.exception("Не получилось отправить сообщение клиенту")
        await event.reply("Не получилось отправить сообщение")


@client.on(events.NewMessage)
async def handler(event: events.NewMessage.Event) -> None:
    task = asyncio.create_task(process_event_safe(event))
    task.add_done_callback(log_task_exception)


def log_task_exception(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logging.exception("Unhandled exception in process_event task")


async def process_event_safe(event: events.NewMessage.Event) -> None:
    try:
        await process_event(event)
    except Exception:
        logging.exception("process_event failed")


async def process_event(event: events.NewMessage.Event) -> None:
    started_at = getnow()

    msg_time_local = event.message.date.astimezone(ZoneInfo("Europe/Podgorica"))
    lag_sec = (started_at - msg_time_local).total_seconds()

    async with metrics_lock:
        since_prev = (
            None
            if state.last_handler_start is None
            else (started_at - state.last_handler_start).total_seconds()
        )
        state.last_handler_start = started_at

    logging.info(
        "HANDLER START chat_id=%s event_id=%s msg_time_local=%s handler_now=%s lag_sec=%.3f since_prev=%.3f",
        event.chat_id,
        event.id,
        msg_time_local.isoformat(),
        started_at.isoformat(),
        lag_sec,
        since_prev if since_prev is not None else -1.0,
    )

    sender_id = event.sender_id
    if sender_id is None:
        return

    raw_text = (event.raw_text or "").strip()
    if not raw_text:
        return

    alert_recipients = {
        cfg["recipient"] for cfg in CONFIGS if isinstance(cfg.get("recipient"), int)
    }

    if sender_id in alert_recipients and event.is_reply:
        lower = raw_text.lower()
        prefixes = (
            "предложи трансфер",
            "предложить трансфер",
            "предложи попутку",
            "предложить попутку",
        )

        prefix_used = next((p for p in prefixes if lower.startswith(p)), None)
        if prefix_used:
            await handle_transfer_offer(event, raw_text, prefix_used)
            return

    text = normalize_text(raw_text)
    recent_messages = await get_recent_messages(sender_id)

    for config in CONFIGS:
        chats = config.get("chats", set())
        if isinstance(chats, set) and event.chat_id not in chats:
            continue

        excluded_senders = config.get("excluded_senders", [])
        if isinstance(excluded_senders, list) and sender_id in excluded_senders:
            continue

        if any(prev_text == text for prev_text, _ in recent_messages):
            logging.info("⛔ Повтор от пользователя %s: %s", sender_id, text)
            continue

        keywords = config.get("keywords", [])
        matched = any(word in text for word in keywords) if isinstance(keywords, list) else False

        is_question = "?" in raw_text
        include_questions = bool(config.get("include_questions"))

        if not (matched or (include_questions and is_question)):
            continue

        excluded_keywords = config.get("excluded_keywords", [])
        if isinstance(excluded_keywords, list) and any(
            block_word in text for block_word in excluded_keywords
        ):
            logging.info("⛔ Игнор по слову для пользователя %s: %s", sender_id, text)
            continue

        now = getnow()
        if any((now - ts) < timedelta(minutes=PERIOD_MINUTES) for _, ts in recent_messages):
            logging.info(
                "⏱️ Игнор: пользователь %s уже писал за последние %s минут",
                sender_id,
                PERIOD_MINUTES,
            )
            continue

        if ENABLE_SEMANTIC_FILTER and await is_semantically_duplicate(sender_id, text):
            logging.info("⛔ Игнор: пользователь %s уже писал об этом", sender_id)
            continue

        chat_id = event.chat_id
        chat_title = chat_title_cache.get(chat_id, str(chat_id))
        chat_username = chat_username_cache.get(chat_id)

        sender_name = f"user_{sender_id}"
        sender_link = f"[{sender_name}](tg://user?id={sender_id})"

        message_link = None
        if chat_username:
            message_link = f"https://t.me/{chat_username}/{event.id}"

        logging.info(
            "[🔔] Chat: %s | SenderId: %s | Msg: %s",
            chat_title,
            sender_id,
            raw_text,
        )

        message = (
            f"Сообщение в чате \"{chat_title}\" от {sender_link} "
            f"в {msg_time_local.strftime('%H:%M:%S')}:\n\n"
            f"{raw_text}"
        )

        if message_link:
            message += f"\n[Открыть сообщение]({message_link})"

        message += f"\nUSER_ID:{sender_id}"

        recipient = config.get("recipient")
        if isinstance(recipient, int):
            logging.info(
                "before send_message_safe +%.3fs",
                (getnow() - started_at).total_seconds(),
            )

            sent = await send_message_safe(recipient, message)

            logging.info(
                "after send_message_safe +%.3fs sent=%s",
                (getnow() - started_at).total_seconds(),
                sent,
            )

            if sent:
                logging.info(
                    "Message sent | SenderId: %s | Recipient: %s",
                    sender_id,
                    recipient,
                )

        await add_to_user_cache(sender_id, text)

        logging.info(
            "HANDLER END total=%.3fs",
            (getnow() - started_at).total_seconds(),
        )


async def preload_chats() -> None:
    chat_ids: set[int] = set()

    for cfg in CONFIGS:
        chats = cfg.get("chats", set())
        if isinstance(chats, set):
            chat_ids.update(chats)

    for chat_id in chat_ids:
        try:
            entity = await client.get_entity(chat_id)

            title = getattr(entity, "title", str(chat_id))
            username = getattr(entity, "username", None)

            chat_title_cache[chat_id] = title

            if username:
                chat_username_cache[chat_id] = username

            logging.info(
                "Preloaded chat %s title=%s username=%s",
                chat_id,
                title,
                username,
            )

        except Exception:
            logging.exception("Failed to preload chat %s", chat_id)


async def run_bot() -> None:
    await client.start()

    await preload_chats()

    now = getnow().strftime("%d-%m-%Y %H:%M:%S")
    logging.info("🧾 Bot run at %s", now)

    me = await client.get_me()
    logging.info("🧾 Signed in as %s (bot=%s)", me.first_name, me.bot)
    logging.info("✅ Bot is running. Waiting for messages...")

    await client.run_until_disconnected()


async def shutdown() -> None:
    now = getnow().strftime("%d-%m-%Y %H:%M:%S")
    logging.info("🧾 Bot stopped at %s", now)
    await client.disconnect()


async def clear_cache_at_midnight() -> None:
    while True:
        now = getnow()
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        seconds_until_midnight = (tomorrow - now).total_seconds()

        logging.info("⏳ Clearing cache after %s seconds", int(seconds_until_midnight))
        await asyncio.sleep(seconds_until_midnight)

        async with cache_lock:
            user_message_cache.clear()

        async with last_sent_lock:
            last_sent.clear()

        logging.info("🧹 Message cache is cleared at midnight Europe/Podgorica")


async def heartbeat() -> None:
    while True:
        try:
            me = await client.get_me()
            logging.info("HEARTBEAT ok user_id=%s", me.id)
        except Exception:
            logging.exception("HEARTBEAT failed")

        await asyncio.sleep(60)


async def main() -> None:
    try:
        asyncio.create_task(clear_cache_at_midnight())
        asyncio.create_task(heartbeat())
        await run_bot()
    except (KeyboardInterrupt, SystemExit):
        logging.info("⚠️ KeyboardInterrupt — shutting down...")
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
