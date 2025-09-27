import os
import uvicorn
from supabase import create_client, AsyncClient
from dotenv import load_dotenv
from sentistrength import PySentiStr
import asyncpraw
from google import genai
from google.genai import types
import re
from fastapi import FastAPI, HTTPException
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
LAUNCHPAD_URL = "https://launchpad-smu.apps.innovate.sg-cna.com"

# logging.info(f"🔑 REDDIT_CLIENT_ID={REDDIT_CLIENT_ID}")
# logging.info(f"🔑 REDDIT_CLIENT_SECRET={'set' if REDDIT_CLIENT_SECRET else 'MISSING'}")
# logging.info(f"🔑 REDDIT_USER_AGENT={REDDIT_USER_AGENT}")


class ScrapeRequest(BaseModel):
    subreddits: Dict[str, int]

app = FastAPI()
ALLOWED_ORIGINS = [
    LAUNCHPAD_URL
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
INVISIBLE_PATTERN = re.compile(r'[\u200B\u200C\u200D\uFEFF\u00A0]')
def remove_emojis(text: str) -> str:
    text = EMOJI_PATTERN.sub("", text)       # Remove emojis
    text = INVISIBLE_PATTERN.sub("", text)   # Remove zero-width/invisible chars
    return text.strip()

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
LLM_CONCURRENCY = 10
llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

async def get_llm_reply(text: str):
    try:
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
    except Exception as e:
        logging.error(f"❌ Error generating LLM reply: {e}")
        return "An uplifting message could not be generated at this time."

async def get_llm_reply_safe(text: str):
    async with llm_semaphore:
        return await get_llm_reply(text)

async def analyze_and_save(subreddit_name: str, posts: list, buffer: list, batch_size: int = 50):
    """
    Run sentiment analysis in batch, filter negatives, generate LLM replies concurrently,
    and save to Supabase in batches.
    """
    if not posts:
        return

    texts = [p["title"] + " " + p["body"] for p in posts]

    try:
        # Run SentiStrength on all posts at once (single JVM call)
        results = await asyncio.to_thread(senti.getSentiment, texts, score="trinary")
    except Exception as e:
        logging.error(f"❌ Error running SentiStrength on r/{subreddit_name}: {e}")
        return

    negative_posts = [
        (post, sentiment[1])
        for post, sentiment in zip(posts, results)
        if sentiment[1] < -2
    ]

    logging.info(f"r/{subreddit_name}: {len(negative_posts)} negative posts out of {len(posts)} scraped.")

    # Prepare all LLM tasks for negative posts
    llm_tasks = []
    for post, score in negative_posts:
        text_for_llm = post["body"].split("\n")[0] if post["body"] else post["title"]
        llm_tasks.append(get_llm_reply_safe(text_for_llm))

    # Execute LLM calls concurrently, respecting semaphore
    llm_results = await asyncio.gather(*llm_tasks)

    # Collect results into buffer
    for (post, score), llm_reply in zip(negative_posts, llm_results):
        buffer.append({
            "username": post["author"],
            "content": post["body"],
            "score": score,
            "source": "reddit",
            "link": reddit_base_url + post["permalink"],
            "suggested_outreach": llm_reply,
        })

    # Batch flush to Supabase if buffer is full
    if len(buffer) >= batch_size:
        try:
            await supabase.table("messages").upsert(buffer, on_conflict=["link"]).execute()
            logging.info(f"💾 Inserted {len(buffer)} rows into Supabase (batch flush).")
            buffer.clear()
        except Exception as e:
            logging.error(f"❌ Error inserting batch into Supabase: {e}")
            buffer.clear()


async def fetch_subreddit(subreddit_name: str, limit: int):
    """
    Fetch posts from a subreddit and return structured data.
    """
    posts = []
    try:
        subreddit = await reddit.subreddit(subreddit_name)
        async for post in subreddit.new(limit=limit):
            if post.selftext == "" and post.url != "":
                continue
            posts.append({
                "title": post.title,
                "body": remove_emojis(post.selftext) or "",
                "author": str(post.author) if post.author else "[deleted]",
                "permalink": post.permalink,
            })
    except Exception as e:
        logging.error(f"❌ Error fetching subreddit {subreddit_name}: {e}")
    return posts

async def main(subreddits: dict[str, int]):
    buffer = []
    errors = []

    # Launch all subreddit fetches concurrently
    fetch_tasks = {name: asyncio.create_task(fetch_subreddit(name, limit)) for name, limit in subreddits.items()}

    for name, task in fetch_tasks.items():
        posts = await task
        if posts:
            await analyze_and_save(name, posts, buffer)
        else:
            errors.append(f"❌ No posts fetched from r/{name}")

    # Final flush
    if buffer:
        try:
            await supabase.table("messages").upsert(buffer, on_conflict=["link"]).execute()
            logging.info(f"💾 Inserted final {len(buffer)} rows into Supabase.")
            buffer.clear()
        except Exception as e:
            error_message = f"❌ Error inserting final batch into Supabase: {e}"
            logging.error(error_message)
            errors.append(error_message)

    logging.info("✅ Scraping completed.")
    return {"status": "Scraping completed", "errors": errors}

@app.get("/")
async def root():
    return {"message": "Reddit Scraper is running. Use the /scrape endpoint to start scraping."}

@app.post("/scrape")
async def scrape_subreddits(request: ScrapeRequest):
    try:
        result = await main(request.subreddits)
        if result["errors"]:
            # If there were errors, return a 500 status with the error details
            raise HTTPException(status_code=500, detail={"status": "Scraping completed with errors", "errors": result["errors"]})
        return {"status": "Scraping completed successfully"}
    except HTTPException:
        raise # Re-raise the HTTPException
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5005)