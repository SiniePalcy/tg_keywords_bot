import os
from telethon import TelegramClient
from dotenv import load_dotenv

load_dotenv()

api_id_str = os.getenv("API_ID")
if api_id_str is None:
    raise ValueError("API_ID not set")
api_id = int(api_id_str)
api_hash = os.getenv("API_HASH")
session_name = 'list_chats'

client = TelegramClient(session_name, api_id, api_hash)


async def list_chats() -> None:
    async for dialog in client.iter_dialogs():
        print(f"Title: {dialog.name}")
        print(f"Chat ID: {dialog.id}")
        print(f"Is Group: {dialog.is_group}")
        print(f"Username: {getattr(dialog.entity, 'username', None)}")
        print("------")

with client:
    client.loop.run_until_complete(list_chats())
