import sys
import io
import asyncio
from telethon import TelegramClient, events
from telethon.tl.types import User
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

api_id = 21280897
api_hash = '*****'
session_name = 'keyword_alert_notification'

CONFIGS = [
    {
        "chats": {-1001954706166, -1001676333024, -1001214960694},
        "keywords": ['ищу', 'ищем', 'ищет', 'нужна', 'нужен', 'нужно', 'кто', 'кто-нибудь'],
        "excluded_keywords": ['визаран', 'виза ран'],
        "excluded_senders": [],
        "recipient": 419930845,
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

    text = event.raw_text.lower()
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

        print(f"\n🔔 Chat [{chat_title}] | From {sender_name} | Message: {event.raw_text}", flush=True)

        message = (
            f"🚨 Важное сообщение в чате \"{chat_title}\":\n\n"
            f"👤 {sender_name}\n"
            f"💬 {event.raw_text}"
        )

        if message_link:
            message += f"\n\n🔗 [Открыть сообщение]({message_link})"

        await client.send_message(config["recipient"], message, parse_mode='markdown')

async def run_bot():
    await client.start()

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    await client.send_message(CONFIGS[0]["recipient"], f"🚀 Бот запущен в {now}")

    me = await client.get_me()
    print(f"🧾 Signed in as {me.first_name} (bot={me.bot})", flush=True)
    print("✅ Bot is running. Waiting for messages...", flush=True)

    await client.run_until_disconnected()

async def shutdown():
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    try:
        await client.send_message(CONFIGS[0]["recipient"], f"🛑 Бот остановлен в {now}")
    except Exception as e:
            print(f"[ERROR] Failed to send stop message: {e}", flush=True)
    await client.disconnect()


def main():
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_bot())
    except (KeyboardInterrupt, SystemExit):
        print("⚠️ KeyboardInterrupt — shutting down...", flush=True)
        loop.run_until_complete(shutdown())
    finally:
        loop.close()


if __name__ == '__main__':
    asyncio.run(main())
