
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Annotated
from pydantic import BaseModel
import base64
import tensorflow
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import os
import torch
from collections import Counter
from google.cloud import bigquery, storage
from google.oauth2 import service_account
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import psycopg2
from fastapi.responses import RedirectResponse
import torchvision
from torchvision import transforms




conn = psycopg2.connect(
    dbname="sampledb",
    user="app",
    password="pOud4unh16k5Xp9b1HE754U2",
    host="absolutely-verified-stag.a1.pgedge.io",
    port="5432"
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


UPLOAD_FOLDER='static'

class Item(BaseModel):
    image_Path : str | None = None

@app.get("/")
async def dynamic_file(request: Request):
    path = "No Image Uploaded Yet"
    prediction = [[0]]
    return templates.TemplateResponse("index.html", {"request": request, "img_Path": path ,"probability": prediction})


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("open.html", {"request": request})

@app.get('/index')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/login')
def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get('/sign')
def sign(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.get('/about')
def sign(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})




@app.post("/sign")
async def signup(
    request: Request, username: str = Form(...), email: str = Form(...),password1: str = Form(...),password2:str = Form(...) 
):
   
    cur = conn.cursor()
    cur.execute("INSERT INTO cancertb (uname,email,password1,password2) VALUES (%s, %s,%s, %s)", (username,email,password1,password2))
    conn.commit()
    cur.close() 
 
    return RedirectResponse("/login", status_code=303)


@app.post("/login",response_class=HTMLResponse)
async def do_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    cur = conn.cursor()
    cur.execute("SELECT * FROM cancertb WHERE uname=%s and password1=%s", (username,password))
    existing_user = cur.fetchone()
    cur.close()
    
    print(username)
    print(password)
    if existing_user:
        print(existing_user)
        return templates.TemplateResponse("index.html",{"request": request, "username": username, "password": password,"existing_user": existing_user})
    
    else:
        return HTMLResponse(status_code=401, content="Wrong credentials")




@app.post("/upload_image")
async def upload_image(request: Request, image_file: UploadFile = File(...)):
    # Save the uploaded image to the specified folder
    save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    with open(save_path, "wb") as f:
        content = await image_file.read()
        f.write(content)

    # Load the PyTorch model
    try:
        model_file = "Resnet_fineTuning.pth"
        bucket_name = "sandeep_personal"
        key_path = "ck-eams-9260619158c0.json"
        client = storage.Client.from_service_account_json(key_path)
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(model_file)
        blob.download_to_filename(model_file)
        model = torch.load(model_file, map_location=torch.device('cpu'))  # Load model on CPU
    except Exception as e:
        return {"error": str(e)}

    # Define class names for predictions
    class_names = ['benign', 'malignant', 'normal']  # Replace with your actual class names

    # Perform inference on the uploaded image
    try:
        image = Image.open(io.BytesIO(content))
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            model.eval()
            output = model(input_batch)

        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
    except Exception as e:
        return {"error": str(e)}

    # Prepare response data
    context = {
        "request": request,
        "predicted_class": predicted_class
    }

    # Render an HTML template with the prediction result
    return templates.TemplateResponse("result.html", context)
