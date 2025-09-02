from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import traceback

from model import DataPreprocessing, GRUHHO

app = FastAPI()

origins = [
    "https://mkii-forecast.vercel.app/",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_state = {
    "preprocessor": None,
    "model": None,
    "pola_training": None,
    "pola_testing": None,
    "training_log": [],
    "test_results": None,
    "future_predictions": None,
}

class TrainingParams(BaseModel):
    jml_hdnunt: int
    batas_MSE: float
    batch_size: int
    maks_epoch: int
    elang: int
    iterasi: int

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        preprocessor = DataPreprocessing()
        preprocessor.load_and_preprocess(io.BytesIO(contents))
        
        model_state["preprocessor"] = preprocessor
        model_state["pola_training"] = preprocessor.create_pola(preprocessor.train_data)
        model_state["pola_testing"] = preprocessor.create_pola(preprocessor.test_data)
        model_state["model"] = None 
        model_state["test_results"] = None
        model_state["future_predictions"] = None
        
        df_display = preprocessor.df[['Tanggal', 'Terakhir']].copy()
        df_display['Tanggal'] = pd.to_datetime(df_display['Tanggal'], dayfirst=True).dt.strftime('%d/%m/%Y')

        return {
            "message": "Data loaded and preprocessed successfully.",
            "filename": file.filename,
            "data": df_display.to_dict(orient='records'),
            "max_batch_size": len(model_state["pola_training"])
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/train")
async def train_model(params: TrainingParams):
    if not model_state["preprocessor"]:
        raise HTTPException(status_code=400, detail="Data not loaded yet. Please upload a file first.")

    try:
        model = GRUHHO(
            jml_hdnunt=params.jml_hdnunt,
            batas_MSE=params.batas_MSE,
            batch_size=params.batch_size,
            maks_epoch=params.maks_epoch,
            elang=params.elang,
            iterasi=params.iterasi
        )
        model_state["model"] = model
        
        best_weights, best_mse = model.training_gru(model_state["pola_training"])
        
        model_state["training_log"] = model.training_log
        
        return {
            "message": "Training completed.",
            "best_mse": best_mse[1],
            "training_log": model.training_log
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.get("/test")
async def test_model():
    if not model_state["model"]:
        raise HTTPException(status_code=400, detail="Model not trained yet. Please train the model first.")
    
    try:
        prediksi_norm, mse_test = model_state["model"].test_gru(model_state["pola_testing"])
        
        preprocessor = model_state["preprocessor"]
        hasil_prediksi_denorm = preprocessor.data_denormalisasi([p[0] for p in prediksi_norm])

        testing_data_start_index = len(preprocessor.train_data) + 5
        testing_data = preprocessor.df["Terakhir"].iloc[testing_data_start_index : testing_data_start_index + len(hasil_prediksi_denorm)].tolist()
        
        tanggal_test_start_index = len(preprocessor.train_data)
        tanggal_validasi_start_index = tanggal_test_start_index + 5
        
        df_tanggal = preprocessor.df['Tanggal'].copy()
        df_tanggal = pd.to_datetime(df_tanggal, dayfirst=True)

        tanggal_test = df_tanggal.iloc[tanggal_test_start_index : tanggal_test_start_index + len(testing_data)].dt.strftime('%Y-%m-%d').tolist()
        tanggal_validasi = df_tanggal.iloc[tanggal_validasi_start_index : tanggal_validasi_start_index + len(hasil_prediksi_denorm)].dt.strftime('%Y-%m-%d').tolist()

        model_state["test_results"] = {
            "testing_data": {"dates": tanggal_test, "values": testing_data},
            "prediction_data": {"dates": tanggal_validasi, "values": hasil_prediksi_denorm},
            "mse": mse_test
        }

        return model_state["test_results"]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during testing: {str(e)}")

@app.get("/predict")
async def predict_future(n_hari: int = Query(..., gt=0)):
    if not model_state["model"]:
        raise HTTPException(status_code=400, detail="Model not trained yet. Please train the model first.")
    
    try:
        preprocessor = model_state["preprocessor"]
        data_awal = list(preprocessor.df["Terakhir"].iloc[-5:])
        
        hasil_forw_predict_raw = model_state["model"].forw_predict(data_awal, n_hari, preprocessor)
        
        df_tanggal = pd.to_datetime(preprocessor.df['Tanggal'], dayfirst=True)
        tanggal_terakhir = df_tanggal.iloc[-1]
        start_date = tanggal_terakhir + pd.offsets.BDay(1)
        tanggal_prediksi = pd.bdate_range(start=start_date, periods=n_hari)

        predictions = [
            {"date": date.strftime('%Y-%m-%d'), "value": value}
            for date, value in zip(tanggal_prediksi, hasil_forw_predict_raw)
        ]

        model_state["future_predictions"] = predictions
        
        return {"predictions": predictions}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)