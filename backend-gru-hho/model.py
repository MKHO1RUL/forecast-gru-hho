import pandas as pd
import numpy as np
import random
import math
from typing import List, Tuple, Dict

class DataPreprocessing:
    """Handles loading, normalizing, splitting, and shaping the time-series data."""
    def __init__(self, train_percent: float = 0.7):
        self.train_percent = train_percent
        self.df: pd.DataFrame = None
        self.df_normalisasi: pd.Series = None
        self.train_data: np.ndarray = None
        self.test_data: np.ndarray = None
        self.nilai_min: float = None
        self.nilai_max: float = None
    
    def load_and_preprocess(self, dataframe: pd.DataFrame) -> 'DataPreprocessing':
        """Loads and preprocesses the dataframe."""
        self.df = dataframe.copy()
        if 'Tanggal' in self.df.columns:
             self.df['Tanggal'] = pd.to_datetime(self.df['Tanggal'])
             self.df.set_index('Tanggal', inplace=True)
        
        self.df['Terakhir'] = self.df['Terakhir'].astype(float)
        self.df_normalisasi = self._data_normalisasi(self.df['Terakhir'])
        self._split_data()
        return self

    def _normalisasi(self, nilai: float, nilai_min: float, nilai_max: float) -> float:
        """Normalizes a single value to a range of [0, 1]."""
        if (nilai_max - nilai_min) == 0:
            return 0
        return (nilai - nilai_min) / (nilai_max - nilai_min)

    def _denormalisasi(self, nilai: float, nilai_min: float, nilai_max: float) -> float:
        """Denormalizes a single value from [0, 1] back to its original scale."""
        return (nilai * (nilai_max - nilai_min) + nilai_min)

    def _data_normalisasi(self, data: pd.Series) -> pd.Series:
        """Normalizes a pandas Series."""
        self.nilai_min = data.min()
        self.nilai_max = data.max()
        return data.apply(lambda x: self._normalisasi(x, self.nilai_min, self.nilai_max))

    def data_denormalisasi(self, output: List[float]) -> List[float]:
        """Denormalizes a list of values."""
        return [self._denormalisasi(x, self.nilai_min, self.nilai_max) for x in output]

    def _split_data(self):
        """Splits the normalized data into training and testing sets."""
        train_size = int(len(self.df_normalisasi) * self.train_percent)
        self.train_data = self.df_normalisasi.iloc[:train_size].values
        self.test_data = self.df_normalisasi.iloc[train_size:].values

    def create_pola(self, data: np.ndarray, input_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Creates input patterns (X) and target values (y) for the GRU model."""
        X, y = [], []
        for i in range(len(data) - input_size):
            X.append(data[i:i + input_size])
            y.append(data[i + input_size])
        return np.array(X), np.array(y)

class GRU:
    """A from-scratch implementation of a Gated Recurrent Unit (GRU) using NumPy."""
    def __init__(self, jml_hdnunt: int, batch_size: int):
        self.jml_hdnunt = jml_hdnunt
        self.batch_size = batch_size

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function."""
        return np.tanh(x)

    def gru_forward(self, X_data: np.ndarray, y_data: np.ndarray, bobot_elang: List[Tuple[str, List]], ht_min: np.ndarray = None) -> Tuple[List, np.ndarray, List]:
        """Performs the forward pass for a set of hawks."""
        mse_per_elang = []
        total_samples = len(X_data)

        for tipe_solusi, bobot in bobot_elang:
            W_r, U_r, W_z, U_z, W_h, U_h, W_y = [np.array(w) for w in bobot]

            h_t = np.zeros((self.batch_size, self.jml_hdnunt))
            if ht_min is not None:
                rows, cols = min(h_t.shape[0], ht_min.shape[0]), min(h_t.shape[1], ht_min.shape[1])
                h_t[:rows, :cols] = ht_min[:rows, :cols]

            prediksi = []
            for t in range(0, total_samples, self.batch_size):
                end_t = min(t + self.batch_size, total_samples)
                x_batch = X_data[t:end_t]
                h_prev = h_t[:len(x_batch)]

                r_t = self._sigmoid((x_batch @ W_r.T) + (h_prev @ U_r.T))
                z_t = self._sigmoid((x_batch @ W_z.T) + (h_prev @ U_z.T))
                h_tilde = self._tanh((x_batch @ W_h.T) + ((r_t * h_prev) @ U_h.T))
                h_t_current = (1 - z_t) * h_prev + z_t * h_tilde
                y_pred_batch = self._sigmoid(h_t_current @ W_y)

                prediksi.extend(y_pred_batch.tolist())
                h_t[:len(x_batch)] = h_t_current

            prediksi_np = np.array(prediksi).flatten()
            mse = np.mean((prediksi_np - y_data) ** 2) if len(prediksi_np) > 0 else float('inf')
            mse_per_elang.append((tipe_solusi, mse))

        return mse_per_elang, h_t, np.array(prediksi).tolist()

    def gru_forw_predict(self, x_t: np.ndarray, ht_min: np.ndarray, bobot: List) -> Tuple[float, np.ndarray]:
        """Performs a forward pass for a single prediction step."""
        W_r, U_r, W_z, U_z, W_h, U_h, W_y = [np.array(w) for w in bobot]

        r_t = self._sigmoid((x_t @ W_r.T) + (ht_min @ U_r.T))
        z_t = self._sigmoid((x_t @ W_z.T) + (ht_min @ U_z.T))
        h_tilde = self._tanh((x_t @ W_h.T) + ((r_t * ht_min) @ U_h.T))
        h_t = (1 - z_t) * ht_min + z_t * h_tilde
        y_t = self._sigmoid(h_t @ W_y)

        return y_t[0, 0], h_t

class HHO:
    """Implements the Harris Hawks Optimization algorithm to tune GRU weights."""
    def __init__(self, elang: int, iterasi: int, elang_y: int, jml_hdnunt: int, batch_size: int, input_size: int = 5):
        self.elang = elang
        self.iterasi = iterasi
        self.elang_y = elang_y
        self.jml_hdnunt = jml_hdnunt
        self.batch_size = batch_size
        self.input_size = input_size
        self.pop_elang = self._init_population()
        self.gru = GRU(jml_hdnunt, batch_size)

    def _init_population(self) -> List[List[float]]:
        """Initializes the hawk population with random weights."""
        return [[round(random.uniform(-1, 1), 6) for _ in range(self.elang_y)] for _ in range(self.elang)]

    def hawks_conv(self, populasi: List) -> List[Tuple[str, List]]:
        """Converts a flat list of weights into the GRU's matrix structure."""
        hasil_konversi = []
        
        shapes = {
            "W_r": (self.jml_hdnunt, self.input_size), "U_r": (self.jml_hdnunt, self.jml_hdnunt),
            "W_z": (self.jml_hdnunt, self.input_size), "U_z": (self.jml_hdnunt, self.jml_hdnunt),
            "W_h": (self.jml_hdnunt, self.input_size), "U_h": (self.jml_hdnunt, self.jml_hdnunt),
            "W_y": (self.jml_hdnunt, 1),
        }

        for item in populasi:
            tipe_solusi, solusi = ("X", item) if not isinstance(item, tuple) else item
            if not solusi: continue

            bobot = []
            start = 0
            for name, shape in shapes.items():
                rows, cols = shape
                num_elements = rows * cols
                if start + num_elements > len(solusi):
                    raise ValueError(f"Not enough elements in solution to form weight matrix {name}")
                
                flat_weights = solusi[start : start + num_elements]
                matrix = [flat_weights[i*cols : (i+1)*cols] for i in range(rows)]
                bobot.append(matrix)
                start += num_elements
            
            hasil_konversi.append((tipe_solusi, bobot))
        return hasil_konversi

    def hho(self, f_error: List[Tuple[str, float]], max_iter: int, populasi: List[List[float]], ht_min: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, batas_MSE: float) -> Tuple[List, List, np.ndarray]:
        """The main HHO optimization loop."""
        beta = 1.5
        sigma = ((math.gamma(1 + beta) * math.sin((math.pi * beta) / 2)) / (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        for iter_no in range(max_iter):
            rabbit_idx = min(range(len(f_error)), key=lambda i: f_error[i][1])
            x_rabbit = populasi[rabbit_idx][:]
            x_m = [sum(populasi[k][j] for k in range(self.elang)) / self.elang for j in range(self.elang_y)]

            pop_new = []
            for i in range(self.elang):
                E_0 = random.uniform(-1, 1)
                J = 2 * (1 - random.uniform(0, 1))
                E = 2 * E_0 * (1 - (iter_no / max_iter))
                if abs(E) >= 1:
                    q = random.uniform(0, 1)
                    if q >= 0.5:
                        rand_hawk_idx = random.randint(0, self.elang - 1)
                        popbaru = [x_rabbit[j] - random.uniform(0, 1) * abs(populasi[rand_hawk_idx][j] - 2 * random.uniform(0, 1) * populasi[i][j]) for j in range(self.elang_y)]
                    else:
                        popbaru = [(x_rabbit[j] - x_m[j]) - random.uniform(0, 1) * (1 - (-1)) + (-1) for j in range(self.elang_y)]
                    pop_new.append(("X", popbaru))
                else:
                    r = random.uniform(0, 1)
                    if r >= 0.5:
                        if abs(E) >= 0.5:
                            popbaru = [x_rabbit[j] - populasi[i][j] - E * abs(J * x_rabbit[j] - populasi[i][j]) for j in range(self.elang_y)]
                        else:
                            popbaru = [x_rabbit[j] - E * abs(x_rabbit[j] - populasi[i][j]) for j in range(self.elang_y)]
                        pop_new.append(("X", popbaru))
                    else:
                        lf_step = np.array([0.01 * ((random.uniform(0, 1) * sigma) / (abs(random.uniform(0, 1)) ** (1 / beta))) for _ in range(self.elang_y)])
                        if abs(E) >= 0.5:
                            Y = np.array(x_rabbit) - E * abs(J * np.array(x_rabbit) - np.array(populasi[i]))
                        else:
                            Y = np.array(x_rabbit) - E * abs(J * np.array(x_rabbit) - np.array(x_m))
                        
                        bobot_Y = self.hawks_conv([("Y", Y.tolist())])
                        mse_Y, _, _ = self.gru.gru_forward(X_train, y_train, bobot_Y, ht_min)
                        
                        if mse_Y[0][1] < f_error[i][1]:
                             pop_new.append(("Y", Y.tolist()))
                        else:
                            Z = Y + lf_step * random.uniform(0, 1)
                            pop_new.append(("Z", Z.tolist()))

            bobot_hho = self.hawks_conv(pop_new)
            mse_hho, _, _ = self.gru.gru_forward(X_train, y_train, bobot_hho, ht_min)

            next_populasi = []
            next_f_error = []
            pop_new_idx = 0
            for i in range(self.elang):
                if pop_new_idx >= len(pop_new) or pop_new[pop_new_idx][0] == "X":
                    if mse_hho[pop_new_idx][1] < f_error[i][1]:
                        next_f_error.append(mse_hho[pop_new_idx])
                        next_populasi.append(pop_new[pop_new_idx][1])
                    else:
                        next_f_error.append(f_error[i])
                        next_populasi.append(populasi[i])
                    pop_new_idx += 1
                else:
                    if mse_hho[pop_new_idx][1] < f_error[i][1]:
                        next_f_error.append(mse_hho[pop_new_idx])
                        next_populasi.append(pop_new[pop_new_idx][1])
                    else:
                        next_f_error.append(f_error[i])
                        next_populasi.append(populasi[i])
                    pop_new_idx += 1

            populasi = next_populasi
            f_error = next_f_error
            
            if min(e[1] for e in f_error) < batas_MSE:
                print(f"MSE elang telah memenuhi batas: {min(e[1] for e in f_error)} <= {batas_MSE}")
                break

        self.pop_elang = populasi
        return f_error, populasi

class GRUHHO:
    """Orchestrates the GRU training process using the HHO algorithm."""
    def __init__(self, jml_hdnunt: int, batas_MSE: float, batch_size: int, maks_epoch: int, elang: int, iterasi: int, input_size: int = 5):
        self.jml_hdnunt = jml_hdnunt
        self.batas_MSE = batas_MSE
        self.batch_size = batch_size
        self.maks_epoch = maks_epoch
        self.elang = elang
        self.iterasi = iterasi
        self.input_size = input_size
        self.elang_y = (self.input_size * 3 + self.jml_hdnunt * 3 + 1) * self.jml_hdnunt
        self.gru = GRU(jml_hdnunt, batch_size)
        self.hho = HHO(elang, iterasi, self.elang_y, jml_hdnunt, batch_size, self.input_size)
        self.best_weights: List[float] = None
        self.best_mse_info: Tuple[str, float] = (None, float('inf'))
        self.training_log = []

    def training_gru_generator(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains the GRU model using HHO, yielding logs for each epoch."""
        self.training_log = []
        epoch = 0
        ht_min = None
        pop_elang = self.hho.pop_elang
        
        bobot_elang = self.hho.hawks_conv(pop_elang)
        f_error, ht_min, _ = self.gru.gru_forward(X_train, y_train, bobot_elang, ht_min)

        initial_mse = min(e[1] for e in f_error)
        yield f"Initial MSE before training: {initial_mse}"

        while min(e[1] for e in f_error) > self.batas_MSE and epoch < self.maks_epoch:
            epoch += 1
            
            f_error, pop_elang = self.hho.hho(f_error, self.iterasi, pop_elang, ht_min, X_train, y_train, self.batas_MSE)
            
            current_best_mse = min(e[1] for e in f_error)
            log_entry = f"Epoch {epoch}/{self.maks_epoch}, Best MSE: {current_best_mse}"
            self.training_log.append(log_entry)
            yield log_entry

            best_idx_epoch = min(range(len(f_error)), key=lambda i: f_error[i][1])
            bobot_terbaik_epoch = self.hho.hawks_conv([pop_elang[best_idx_epoch]])
            _, ht_min, _ = self.gru.gru_forward(X_train, y_train, bobot_terbaik_epoch, ht_min)

        best_idx = min(range(len(f_error)), key=lambda i: f_error[i][1])
        self.best_mse_info = f_error[best_idx]
        self.best_weights = pop_elang[best_idx]

    def test_gru(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, float]:
        """Tests the trained GRU model."""
        if self.best_weights is None:
            raise RuntimeError("Model has not been trained yet.")
        
        bobot_test = self.hho.hawks_conv([self.best_weights])
        _, _, prediksi = self.gru.gru_forward(X_test, y_test, bobot_test)
        
        prediksi_np = np.array(prediksi).flatten() if prediksi else np.array([])
        mse_test = np.mean((prediksi_np - y_test) ** 2) if len(prediksi_np) > 0 else float('inf')
        
        return np.array(prediksi), mse_test

    def forw_predict(self, data_terakhir: List[float], n_hari: int, preprocessor: DataPreprocessing) -> List[float]:
        """Predicts future values autoregressively."""
        data_terakhir_norm = np.array([preprocessor._normalisasi(x, preprocessor.nilai_min, preprocessor.nilai_max) for x in data_terakhir])
        bobot_maju = self.hho.hawks_conv([self.best_weights])[0][1]
        
        prediksi_norm = []
        ht_min = np.zeros((1, self.jml_hdnunt))
        current_input = data_terakhir_norm.copy()

        for _ in range(n_hari):
            x_t = current_input[-self.input_size:].reshape(1, -1)
            prediksi_satu_langkah, ht_min = self.gru.gru_forw_predict(x_t, ht_min, bobot_maju)
            prediksi_norm.append(prediksi_satu_langkah)
            current_input = np.append(current_input, prediksi_satu_langkah)[1:]

        return preprocessor.data_denormalisasi(prediksi_norm)
