"""
Parse Telegram channel @msk_live posts, comments and reactions
and put them in postgres tables.

CREATE TABLE IF NOT EXISTS msk_live_posts (
    id INTEGER PRIMARY KEY,
    date TEXT,
    text TEXT
);

CREATE TABLE IF NOT EXISTS msk_live_comments (
    id INTEGER PRIMARY KEY,
    post_id INTEGER,
    date TEXT,
    text TEXT,
    sentiment INTEGER
);

CREATE TABLE IF NOT EXISTS msk_live_reactions (
    id SERIAL PRIMARY KEY,
    post_id INTEGER,
    emoji TEXT,
    count INTEGER
);
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from dotenv import load_dotenv
import os

load_dotenv("/opt/airflow/dags/.env")

API_ID = int(os.getenv("TG_API_ID"))
API_HASH = os.getenv("TG_API_HASH")
CHANNEL = "@msk_live"
SESSION_PATH = "/opt/airflow/dags/prod_session_tg.session"
POSTGRES_CONN_ID = "postgres_news"

default_args = {
    "owner": "airflow",
    "retries": 3,
}

with DAG(
    dag_id="daily_news_msk_live_parser",
    start_date=datetime(2026, 4, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["news", "telegram", "msk_live"],
) as dag:

    @task()
    def parse_and_save():
        import warnings
        warnings.filterwarnings("ignore")
        import socks
        from telethon import TelegramClient

        async def _parse():
            client = TelegramClient(
                SESSION_PATH, API_ID, API_HASH,
                proxy=(socks.HTTP,
                    os.getenv("PROXY_HOST"),
                    int(os.getenv("PROXY_PORT")),
                    True,
                    os.getenv("PROXY_USER"),
                    os.getenv("PROXY_PASS"))
            )
            await client.connect()

            hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
            conn = hook.get_conn()
            cursor = conn.cursor()

            yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
            date_start = datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc)
            date_end = date_start + timedelta(days=1)

            new_posts = []
            async for message in client.iter_messages(CHANNEL, offset_date=date_end, limit=500):
                if message.date < date_start:
                    break
                if message.text:
                    new_posts.append(message)

            new_posts.reverse()
            logging.info(f"Found {len(new_posts)} posts for {yesterday}")

            for post in new_posts:
                cursor.execute(
                    "INSERT INTO msk_live_posts (id, date, text) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (post.id, str(post.date), post.text or ""),
                )
                async for comment in client.iter_messages(CHANNEL, reply_to=post.id):
                    if comment.text:
                        cursor.execute(
                            "INSERT INTO msk_live_comments (id, post_id, date, text) VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                            (comment.id, post.id, str(comment.date), comment.text),
                        )
                if post.reactions:
                    cursor.execute("DELETE FROM msk_live_reactions WHERE post_id = %s", (post.id,))
                    for r in post.reactions.results:
                        emoji = r.reaction.emoticon if hasattr(r.reaction, 'emoticon') else 'custom'
                        cursor.execute(
                            "INSERT INTO msk_live_reactions (post_id, emoji, count) VALUES (%s, %s, %s)",
                            (post.id, emoji, r.count),
                        )
                conn.commit()
                logging.info(f"Saved post {post.id}")
                await asyncio.sleep(0.3)

            cursor.close()
            conn.close()
            await client.disconnect()
            logging.info("Parsing complete")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_parse())
        finally:
            loop.close()

    parse_and_save()