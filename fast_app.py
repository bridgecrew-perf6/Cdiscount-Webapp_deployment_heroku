from fastapi import FastAPI


app = FastAPI()


@app.get("/")
def home():
    return {"Data": "Test"}


@app.get("/about")
def about():
    return {"Data": "About"}


"""""
#to run this type following command in the terminal
command to run ---> uvicorn fast_app:app --reload
"""
