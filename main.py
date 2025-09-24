import os
import uvicorn
from supabase import create_client, AsyncClient
from dotenv import load_dotenv
from sentistrength import PySentiStr
import asyncpraw
from google import genai
from google.genai import types
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import asyncio
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

reddit_base_url = "http://www.reddit.com"

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LAUNCHPAD_URL = os.getenv("LAUNCHPAD_URL")
class ScrapeRequest(BaseModel):
    subreddits: Dict[str, int]

app = FastAPI()
ALLOWED_ORIGINS = [
    LAUNCHPAD_URL,
    "https://launchpad-smu.apps.innovate.sg-cna.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[LAUNCHPAD_URL],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# Pre-compile regex once
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & pictographs
    "\U0001F680-\U0001F6FF"  # Transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002700-\U000027BF"  # Dingbats
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)
def remove_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub("", text)



# Initialize clients asynchronously
genai_client = genai.Client(api_key=GOOGLE_API_KEY)
supabase: AsyncClient = AsyncClient(
    supabase_url=SUPABASE_URL,
    supabase_key=SUPABASE_KEY
)
reddit = asyncpraw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)

# Initialize SentiStrength
senti = PySentiStr()
senti.setSentiStrengthPath("./SentiStrength.jar")
senti.setSentiStrengthLanguageFolderPath("./SentiStrengthDataEnglishOctober2019")

# Concurrency controls
LLM_CONCURRENCY = 5
llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

async def get_llm_reply(text: str):
    payload = (
        f"Provide a short, uplifting message within 130 words in response to the following:\n\n{text}. Redirect them to this website that allows them to go through a survey to determine their emotions if it was a planet. https://website-smu.apps.innovate.sg-cna.com/planet-quiz Be sure to include the exact link in your response."
    )
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=payload,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return response.text

async def get_llm_reply_safe(text: str):
    async with llm_semaphore:
        return await get_llm_reply(text)

async def analyze_and_save(subreddit_name: str, posts: list, buffer: list, batch_size: int = 50):
    """
    Run sentiment analysis in batch, filter negatives, generate LLM replies, and save.
    """
    if not posts:
        return

    texts = [p["title"] + " " + p["body"] for p in posts]

    # Run one JVM call for ALL posts in this subreddit
    results = await asyncio.to_thread(senti.getSentiment, texts, score="trinary")

    negative_tasks = []
    for post, sentiment in zip(posts, results):
        score = sentiment[1]
        if score < -2:
            negative_tasks.append((post, score))

    logging.info(f"r/{subreddit_name}: {len(negative_tasks)} negative posts out of {len(posts)} scraped.")

    for post, score in negative_tasks:
        llm_reply = await get_llm_reply_safe(
            post["body"].split("\n")[0] if post["body"] else post["title"]
        )
        buffer.append({
            "username": post["author"],
            "content": post["body"],
            "score": score,
            "source": "reddit",
            "link": reddit_base_url + post["permalink"],
            "suggested_outreach": llm_reply,
        })

        # Batch flush to Supabase
        if len(buffer) >= batch_size:
            await supabase.table("messages").insert(buffer).execute()
            logging.info(f"💾 Inserted {len(buffer)} rows into Supabase (batch flush).")
            buffer.clear()

async def main(subreddits: dict[str, int]):
    """
    Main entrypoint for scraping and batch sentiment analysis.
    """
    buffer = []

    for subreddit_name, limit in subreddits.items():
        logging.info(f"📡 Scraping r/{subreddit_name} (limit={limit})...")
        posts = []
        try:
            subreddit = await reddit.subreddit(subreddit_name)
            async for post in subreddit.new(limit=limit):
                posts.append({
                    "title": post.title,
                    "body": remove_emojis(post.selftext) or "",
                    "author": str(post.author) if post.author else "[deleted]",
                    "url": post.url,
                })

            # Batch analyze and save after collecting posts
            await analyze_and_save(subreddit_name, posts, buffer)

        except Exception as e:
            logging.error(f"❌ Error fetching subreddit {subreddit_name}: {e}")

    # Final flush if buffer has leftovers
    if buffer:
        await supabase.table("messages").insert(buffer).execute()
        logging.info(f"💾 Inserted final {len(buffer)} rows into Supabase.")

    logging.info("✅ Scraping completed.")

@app.get("/")
async def root():
    return {"message": "Reddit Scraper is running. Use the /scrape endpoint to start scraping."}

@app.post("/scrape")
async def scrape_subreddits(request: ScrapeRequest):
    await main(request.subreddits)
    return {"status": "Scraping completed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5005)