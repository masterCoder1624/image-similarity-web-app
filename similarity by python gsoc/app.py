import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import List, Optional
import uvicorn

app = FastAPI(title="Image Similarity Detector")

os.makedirs("static/uploads", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Function to calculate histogram-based similarity
def histogram_similarity(img1, img2):
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL) * 100

# Function to calculate SSIM similarity
def ssim_similarity(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    height = min(gray1.shape[0], gray2.shape[0])
    width = min(gray1.shape[1], gray2.shape[1])
    gray1_resized = cv2.resize(gray1, (width, height))
    gray2_resized = cv2.resize(gray2, (width, height))
    

    similarity_index, _ = ssim(gray1_resized, gray2_resized, full=True)
    return similarity_index * 100

# Function to calculate feature matching similarity
def feature_matching_similarity(img1, img2):

    sift = cv2.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0
    
    # Using FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return 0
    
    # ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Calculate similarity score based on good matches
    max_possible_matches = min(len(kp1), len(kp2))
    if max_possible_matches == 0:
        return 0
    
    return (len(good_matches) / max_possible_matches) * 100

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/compare")
async def compare_images(
    request: Request,
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    # Create unique filenames
    filename1 = f"{uuid.uuid4()}{os.path.splitext(image1.filename)[1]}"
    filename2 = f"{uuid.uuid4()}{os.path.splitext(image2.filename)[1]}"
    
    # in this saving our uploaded files 
    filepath1 = f"static/uploads/{filename1}"
    filepath2 = f"static/uploads/{filename2}"
    

    with open(filepath1, "wb") as f:
        shutil.copyfileobj(image1.file, f)
    with open(filepath2, "wb") as f:
        shutil.copyfileobj(image2.file, f)
    

    img1 = cv2.imread(filepath1)
    img2 = cv2.imread(filepath2)
    
  
    hist_sim = histogram_similarity(img1, img2)
    ssim_sim = ssim_similarity(img1, img2)
    feature_sim = feature_matching_similarity(img1, img2)
    
    avg_similarity = (hist_sim + ssim_sim + feature_sim) / 3
    
    # Return result with image paths
    return templates.TemplateResponse(
        "result.html", 
        {
            "request": request,
            "image1": filepath1,
            "image2": filepath2,
            "histogram_similarity": f"{hist_sim:.2f}%",
            "ssim_similarity": f"{ssim_sim:.2f}%",
            "feature_similarity": f"{feature_sim:.2f}%",
            "average_similarity": f"{avg_similarity:.2f}%"
        }
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True) 