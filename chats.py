
import sys
import io
from telethon import TelegramClient

api_id = 21280897
api_hash = '*****'
session_name = 'list_chats'

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')


client = TelegramClient(session_name, api_id, api_hash)

async def list_chats():
    async for dialog in client.iter_dialogs():
        print(f"Title: {dialog.name}")
        print(f"Chat ID: {dialog.id}")
        print(f"Is Group: {dialog.is_group}")
        print(f"Username: {getattr(dialog.entity, 'username', None)}")
        print("------")

with client:
    client.loop.run_until_complete(list_chats())
