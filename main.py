from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import io
from core import process_eye_image

app = FastAPI(title="Eye Analysis Microservice")

@app.get("/")
def read_root():
    return {"message": "Göz Analiz Servisine Hoşgeldiniz. /analyze endpointine POST atın."}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Dosya bir resim olmalıdır.")
    
    try:
        
        contents = await file.read()
        
        
        processed_image_bytes = process_eye_image(contents)
        
        
        return Response(content=processed_image_bytes, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))