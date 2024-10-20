import os

from fastapi.responses import HTMLResponse
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from socialscope.routers import tweets, tweet

load_dotenv()


# Initialize the FastAPI app
app = FastAPI(
    title="socialscope",
    debug=os.getenv("DEBUG_MODE", "0") == "1",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "https://socialscope.108122.xyz",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tweets.router, prefix="/api/v1")
app.include_router(tweet.router, prefix="/api/v1")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/hello")
async def welcome():
    """
    Welcome message for the socialscope API.
    """
    return {"message": "Welcome to socialscope API"}


@app.get("/x", response_class=HTMLResponse)
async def get_tweets_page():
    with open("static/tweets.html") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
async def get_text_sentiment():
    with open("static/tweet.html") as f:
        return f.read()


def main():
    debug_mode = os.getenv("DEBUG_MODE", "0") == "1"

    uvicorn.run(
        "socialscope.main:app",
        host=("127.0.0.1" if debug_mode else "0.0.0.0"),
        port=8080,
        reload=debug_mode,
        reload_dirs=["./socialscope"],
        workers=1,
    )


if __name__ == "__main__":
    main()
