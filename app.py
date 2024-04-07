
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

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from glob import glob



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
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    with open(image_path, "wb") as f:
        content = await image_file.read()
        f.write(content)

    # Load the PyTorch model
    try:
        model_file = "cancer2.pt"
        bucket_name = "sandeep_personal"
        key_path = "ck-eams-9260619158c0.json"
        client = storage.Client.from_service_account_json(key_path)
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(model_file)
        blob.download_to_filename(model_file)
        model = torch.load(model_file, map_location=torch.device('cpu'))  # Load model on CPU
    except Exception as e:
        return {"error": str(e)}

    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to match the input size of the model
        transforms.ToTensor(),         # Convert to tensor format
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
    ])
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # 2. Load the Pretrained Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Ensure device compatibility
    model = LNN(NCP_INPUT_SIZE, HIDDEN_NEURONS, NUM_OUTPUT_CLASSES, SEQUENCE_LENGTH).to(device)
    model.load_state_dict(torch.load(saved_model_path))  # Load pretrained model weights
    model.eval()

    # 3. Run Inference on the Single Image
    with torch.no_grad():
        image_tensor = image_tensor.to(device)  # Move tensor to device
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # 4. Interpret the Prediction
    predicted_class=""
    if prediction == 0:
        predicted_class=predicted_class+"The image has cancer."
    else:
        predicted_class=predicted_class+"The image does not have cancer."


    # Prepare response data
    context = {
        "request": request,
        "predicted_class": predicted_class,
        "path":image_path
    }

    # Render an HTML template with the prediction result
    return templates.TemplateResponse("result.html", context)
