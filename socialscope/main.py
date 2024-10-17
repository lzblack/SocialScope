import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from socialscope.routers import tweets

load_dotenv()

app = FastAPI(title="socialscope", debug=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tweets.router, prefix="/api/v1")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_root():
    """
    Welcome message for the socialscope API.
    """
    return {"message": "Welcome to socialscope API"}


def main():
    debug_mode = os.getenv("DEBUG_MODE", "0") == "1"

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=debug_mode,
        # debug=debug_mode,
        workers=1,
    )


if __name__ == "__main__":
    main()
