from fastapi import FastAPI
from app.api.routes import router


# uvicorn app.main:app --reload , use to run the application
# curl -X POST "http://127.0.0.1:8000/chat" \
#   -H "Content-Type: application/json" \
#   -d '{"messages":"what is the next step in document?"}'


app = FastAPI(title= 'Enterprise Agent')
app.include_router(router)
