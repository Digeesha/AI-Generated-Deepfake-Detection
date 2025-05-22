from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import shutil
from deepfake_model import load_model, predict_on_video

app = FastAPI()
model = load_model("model.pth")  # Load your trained model

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    prediction = predict_on_video(tmp_path, model)
    return JSONResponse(content={"result": prediction})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
