from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from fastapi.encoders import jsonable_encoder
import io
import asyncio
import numpy as np
import json
import traceback
import yfinance as yf
from datetime import datetime, timezone

from model import DataPreprocessing, GRUHHO

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://mkii-forecast.vercel.app",
    "https://forecast-gru-hho-production.up.railway.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = {}

STATE_RAW_DF = "raw_df"
STATE_PREPROCESSOR = "preprocessor"
STATE_MODEL = "model"
STATE_POLA_TRAINING = "pola_training"
STATE_POLA_TESTING = "pola_testing"
STATE_TRAINING_LOG = "training_log"
STATE_TEST_RESULTS = "test_results"
STATE_FUTURE_PREDICTIONS = "future_predictions"

model_state = {
    STATE_RAW_DF: None,
    STATE_PREPROCESSOR: None,
    STATE_MODEL: None,
    STATE_POLA_TRAINING: None,
    STATE_POLA_TESTING: None,
    STATE_TRAINING_LOG: [],
    STATE_TEST_RESULTS: None,
    STATE_FUTURE_PREDICTIONS: None,
}

class TrainingParams(BaseModel):
    jml_hdnunt: int = Field(..., gt=1, description="Jumlah unit di hidden layer GRU, harus lebih besar dari 1.")
    batas_MSE: float = Field(..., gt=0, lt=1, description="Ambang batas MSE untuk menghentikan training, harus antara 0 dan 1.")
    batch_size: int = Field(..., gt=0, description="Jumlah sampel per batch, harus lebih besar dari 0.")
    maks_epoch: int = Field(..., gt=0, description="Jumlah epoch maksimum, harus lebih besar dari 0.")
    elang: int = Field(..., gt=1, description="Jumlah elang di HHO, harus lebih besar dari 1.")
    iterasi: int = Field(..., gt=0, description="Jumlah iterasi per epoch HHO, harus lebih besar dari 0.")
    input_size: int = Field(5, gt=0, description="Ukuran sekuens input, harus lebih besar dari 0.")

@app.get("/get-data")
async def get_data(pair: str = Query(..., description="Contoh: 'EURUSD=X'")):
    """
    Mengambil data forex dari yfinance dengan caching.
    """
    now_utc = datetime.now(timezone.utc)
    
    if pair in cache and cache[pair]["last_updated"].date() == now_utc.date():
        print(f"Menggunakan data dari cache untuk {pair}")
        df = cache[pair]["data"]
    else:
        print(f"Mengambil data baru untuk {pair} dari yfinance...")
        try:
            data = yf.download(pair, period="5y", interval="1d", progress=False, auto_adjust=True)

            if data.empty:
                raise HTTPException(status_code=404, detail=f"Tidak ada data untuk pair: {pair}. Mungkin ticker tidak valid.")

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if 'Close' not in data.columns:
                raise ValueError(f"Kolom 'Close' tidak ditemukan dalam data untuk {pair}")

            df = data[['Close']].dropna().copy()
            df.rename(columns={'Close': 'Terakhir'}, inplace=True)
            df.index.name = 'Tanggal'

            cache[pair] = {"data": df, "last_updated": now_utc}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Gagal mengambil data dari yfinance: {str(e)}")

    model_state[STATE_RAW_DF] = df.copy()

    df_display = df.reset_index()
    df_display['Tanggal'] = pd.to_datetime(df_display['Tanggal']).dt.strftime('%Y-%m-%d')
    df_display['Terakhir'] = df_display['Terakhir'].astype(float)

    model_state.update({
        STATE_PREPROCESSOR: None, STATE_MODEL: None, STATE_POLA_TRAINING: None,
        STATE_POLA_TESTING: None, STATE_TRAINING_LOG: [], STATE_TEST_RESULTS: None,
        STATE_FUTURE_PREDICTIONS: None
    })

    return JSONResponse(content={
        "message": f"Data untuk {pair} berhasil dimuat.",
        "data": df_display.to_dict(orient='records'),
        "max_batch_size": int(len(df) * 0.7) 
    })

@app.post("/train")
async def train_model(params: TrainingParams):
    try:
        df = model_state.get(STATE_RAW_DF)
        if df is None:
            raise HTTPException(status_code=400, detail="Data tidak ditemukan. Silakan muat data melalui endpoint /get-data terlebih dahulu.")
        
        preprocessor = DataPreprocessing()
        preprocessor.load_and_preprocess(dataframe=df)

        if 'Tanggal' in preprocessor.df.columns:
            preprocessor.df['Tanggal'] = pd.to_datetime(preprocessor.df['Tanggal'])
            preprocessor.df.set_index('Tanggal', inplace=True)
        
        model_state[STATE_PREPROCESSOR] = preprocessor
        model_state[STATE_POLA_TRAINING] = preprocessor.create_pola(preprocessor.train_data, input_size=params.input_size)
        model_state[STATE_POLA_TESTING] = preprocessor.create_pola(preprocessor.test_data, input_size=params.input_size)

        model = GRUHHO(
            jml_hdnunt=params.jml_hdnunt,
            batas_MSE=params.batas_MSE,
            batch_size=params.batch_size,
            maks_epoch=params.maks_epoch,
            elang=params.elang,
            iterasi=params.iterasi,
            input_size=params.input_size
        )
        model_state[STATE_MODEL] = model
        
        async def train_event_stream(model_instance, training_data):
            """Generator untuk stream event training."""
            try:
                for log in model_instance.training_gru_generator(*training_data):
                    log_data = {"type": "log", "message": str(log)}
                    yield f"{json.dumps(log_data)}\n"
                    await asyncio.sleep(0.01) 
                
                final_result = {
                    "type": "complete",
                    "message": "Training completed.",
                    "best_mse": float(model_instance.best_mse_info[1])
                }
                yield f"{json.dumps(final_result)}\n"

            except Exception as e:
                traceback.print_exc()
                error_data = {"type": "error", "message": f"Error during training: {str(e)}"}
                yield f"{json.dumps(error_data)}\n"

        return StreamingResponse(train_event_stream(model, model_state[STATE_POLA_TRAINING]), media_type="application/x-ndjson")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during training setup: {str(e)}")

@app.get("/test")
async def test_model():
    model = model_state.get(STATE_MODEL)
    preprocessor = model_state.get(STATE_PREPROCESSOR)
    pola_testing = model_state.get(STATE_POLA_TESTING)

    if not all([model, preprocessor, pola_testing]):
        raise HTTPException(status_code=400, detail="Model atau preprocessor belum siap. Latih model terlebih dahulu.")

    try:
        time_step = model.input_size

        prediksi_norm, mse_test = model.test_gru(*pola_testing)

        hasil_prediksi_denorm = preprocessor.data_denormalisasi(prediksi_norm.flatten())
        hasil_prediksi_denorm = [float(v) for v in hasil_prediksi_denorm]

        y_test_norm = pola_testing[1]
        y_test_denorm = preprocessor.data_denormalisasi(y_test_norm.flatten())
        y_test_denorm = [float(v) for v in y_test_denorm]

        train_size = len(preprocessor.train_data)
        start_index = train_size + time_step
        end_index = start_index + len(y_test_denorm)
        
        tanggal_plot = preprocessor.df.index[start_index:end_index].strftime('%Y-%m-%d').tolist()

        if len(tanggal_plot) != len(y_test_denorm):
            min_len = min(len(tanggal_plot), len(y_test_denorm))
            tanggal_plot = tanggal_plot[:min_len]
            y_test_denorm = y_test_denorm[:min_len]
            hasil_prediksi_denorm = hasil_prediksi_denorm[:min_len]

        model_state[STATE_TEST_RESULTS] = {
            "testing_data": {"dates": tanggal_plot, "values": y_test_denorm},
            "prediction_data": {"dates": tanggal_plot, "values": hasil_prediksi_denorm},
            "mse": float(mse_test)
        }

        return JSONResponse(content=jsonable_encoder(model_state[STATE_TEST_RESULTS]))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during testing: {str(e)}")

@app.get("/predict")
async def predict_future(n_hari: int = Query(..., gt=0)):
    model = model_state.get(STATE_MODEL)
    preprocessor = model_state.get(STATE_PREPROCESSOR)

    if not all([model, preprocessor]):
        raise HTTPException(status_code=400, detail="Model atau preprocessor belum siap. Latih model terlebih dahulu.")

    try:
        data_terakhir = preprocessor.df["Terakhir"].iloc[-model.input_size:].tolist()

        hasil_prediksi_denorm = model.forw_predict(data_terakhir, n_hari, preprocessor)

        tanggal_terakhir = preprocessor.df.index[-1]
        start_date = tanggal_terakhir + pd.offsets.BDay(1)
        tanggal_prediksi = pd.bdate_range(start=start_date, periods=n_hari)

        predictions = [
            {"date": date.strftime('%Y-%m-%d'), "value": float(value)}
            for date, value in zip(tanggal_prediksi, hasil_prediksi_denorm)
        ]

        model_state[STATE_FUTURE_PREDICTIONS] = predictions
        
        return JSONResponse(content={"predictions": jsonable_encoder(predictions)})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)