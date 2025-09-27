import os
import uvicorn
import asyncio
import logging
import re
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentistrength import PySentiStr
import asyncpraw
from supabase import AsyncClient, acreate_client
from dotenv import load_dotenv
from google import genai
from google.genai import types
import more_itertools

# ---------------------------
# Setup logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LAUNCHPAD_URL = "https://launchpad-smu.apps.innovate.sg-cna.com"
reddit_base_url = "http://www.reddit.com"

# ---------------------------
# FastAPI setup
# ---------------------------
app = FastAPI()
ALLOWED_ORIGINS = [LAUNCHPAD_URL]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[LAUNCHPAD_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# ---------------------------
# Regex helpers
# ---------------------------
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  
    "\U0001F300-\U0001F5FF"  
    "\U0001F680-\U0001F6FF"  
    "\U0001F1E0-\U0001F1FF"  
    "\U00002700-\U000027BF"  
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)
INVISIBLE_PATTERN = re.compile(r'[\u200B\u200C\u200D\uFEFF\u00A0]')

def remove_emojis(text: str) -> str:
    text = EMOJI_PATTERN.sub("", text)
    text = INVISIBLE_PATTERN.sub("", text)
    return text.strip()

# ---------------------------
# Models
# ---------------------------
class ScrapeRequest(BaseModel):
    subreddits: Dict[str, int]

# ---------------------------
# Global clients
# ---------------------------
supabase: AsyncClient | None = None
reddit: asyncpraw.Reddit | None = None
genai_client: genai.Client | None = None
senti: PySentiStr | None = None
existing_links_set: set[str] = set()

# Semaphore for LLM concurrency
LLM_CONCURRENCY = 10
llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

# ---------------------------
# Startup / Shutdown events
# ---------------------------
@app.on_event("startup")
async def startup_event():
    global supabase, reddit, genai_client, senti, existing_links_set

    # Supabase
    supabase = await acreate_client(supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY)
    logging.info("✅ Supabase client initialized")

    try:
        response = await (
            supabase.table("messages")
            .select("link") 
            .execute()
        )
        existing_links_set = {item['link'] for item in response.data}
        logging.info(f"✅ Fetched {len(existing_links_set)} unique links.")
    except Exception as e:
        logging.error(f"❌ Error fetching links from Supabase: {e}")# Return an empty set to prevent errors downstream
    
    # Reddit
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    logging.info("✅ Reddit client initialized")

    # GenAI
    genai_client = genai.Client(api_key=GOOGLE_API_KEY)

    # SentiStrength
    senti = PySentiStr()
    senti.setSentiStrengthPath("./SentiStrength.jar")
    senti.setSentiStrengthLanguageFolderPath("./SentiStrengthDataEnglishOctober2019")
    logging.info("✅ SentiStrength initialized")

@app.on_event("shutdown")
async def shutdown_event():
    global supabase, reddit
    if supabase:
        await supabase.remove_all_channels() 
        logging.info("✅ Supabase client closed")
    if reddit:
        await reddit.close()
        logging.info("✅ Reddit client closed")

# ---------------------------
# LLM helper
# ---------------------------
async def get_llm_reply(text: str):
    try:
        payload = (
            f"Provide a short, uplifting message within 130 words in response to the following:\n\n{text}. "
            "Redirect them to this website that allows them to go through a survey to determine their emotions if it was a planet. "
            "https://website-smu.apps.innovate.sg-cna.com/planet-quiz Be sure to include the exact link in your response."
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

# ---------------------------
# Analysis and Supabase saving
# ---------------------------
async def analyze_and_save(subreddit_name: str, posts: list, buffer: list, batch_size: int = 50):
    global existing_links_set
    if not posts:
        return

    texts = [p["title"] + " " + p["body"] for p in posts]

    try:
        results = await asyncio.to_thread(senti.getSentiment, texts, score="trinary")
    except Exception as e:
        logging.error(f"❌ Error running SentiStrength on r/{subreddit_name}: {e}")
        return

    negative_posts = [(post, sentiment[1]) for post, sentiment in zip(posts, results) if sentiment[1] < -2 and reddit_base_url + post["permalink"] not in existing_links_set]
    logging.info(f"r/{subreddit_name}: {len(negative_posts)} negative posts out of {len(posts)} scraped.")

    # Batched LLM calls to avoid memory issues
    for chunk in more_itertools.chunked(negative_posts, 10):
        llm_tasks = [get_llm_reply_safe(post["body"].split("\n")[0] if post["body"] else post["title"]) for post, _ in chunk]
        llm_results = await asyncio.gather(*llm_tasks)

        for (post, score), llm_reply in zip(chunk, llm_results):
            buffer.append({
                "username": post["author"],
                "content": post["body"],
                "score": score,
                "source": "reddit",
                "link": reddit_base_url + post["permalink"],
                "suggested_outreach": llm_reply,
            })

        # Batch flush
        if len(buffer) >= batch_size:
            try:
                await supabase.table("messages").upsert(buffer, on_conflict=["link"]).execute()
                logging.info(f"💾 Inserted {len(buffer)} rows into Supabase (batch flush).")
                buffer.clear()
            except Exception as e:
                logging.error(f"❌ Error inserting batch into Supabase: {e}")
                buffer.clear()

# ---------------------------
# Reddit fetching
# ---------------------------
async def fetch_subreddit(subreddit_name: str, limit: int):
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

# ---------------------------
# Main scraping logic with timeout
# ---------------------------
async def main_with_timeout(subreddits: dict[str,int], timeout: int = 300):
    try:
        return await asyncio.wait_for(main(subreddits), timeout=timeout)
    except asyncio.TimeoutError:
        logging.error("❌ Scraping timed out")
        return {"status": "timeout", "errors": ["Scraping timed out"]}
    except asyncio.CancelledError:
        logging.warning("⚠️ Scraping cancelled")
        return {"status": "cancelled", "errors": ["Scraping was cancelled"]}

async def main(subreddits: dict[str, int]):
    buffer = []
    errors = []

    fetch_tasks = {name: asyncio.create_task(fetch_subreddit(name, limit)) for name, limit in subreddits.items()}

    for name, task in fetch_tasks.items():
        try:
            posts = await task
            if posts:
                await analyze_and_save(name, posts, buffer)
            else:
                errors.append(f"❌ No posts fetched from r/{name}")
        except asyncio.CancelledError:
            errors.append(f"❌ Fetch task for r/{name} was cancelled")

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

# ---------------------------
# FastAPI endpoints
# ---------------------------
@app.get("/")
async def root():
    return {"message": "Reddit Scraper is running. Use the /scrape endpoint to start scraping."}

@app.post("/scrape")
async def scrape_subreddits(request: ScrapeRequest):
    result = await main_with_timeout(request.subreddits, timeout=600)  # 10 minutes max
    if result["errors"]:
        raise HTTPException(status_code=500, detail=result)
    return {"status": "Scraping completed successfully"}

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5005)
