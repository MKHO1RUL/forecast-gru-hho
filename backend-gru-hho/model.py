import pandas as pd
import random
import math
import time

class DataPreprocessing:
    def __init__(self, train_percent=0.7):
        self.train_percent = train_percent
        self.df = None
        self.df_normalisasi = None
        self.train_data = None
        self.test_data = None
        self.nilai_min = None
        self.nilai_max = None

    def load_and_preprocess(self, file_path):
        data = pd.read_csv(file_path)
        self.df = pd.DataFrame(data)
        self.df.drop(['Pembukaan', 'Tertinggi', 'Terendah', 'Vol.', 'Perubahan%'], axis=1, inplace=True)
        self.df['Terakhir'] = self.df['Terakhir'].str.replace('.', '', regex=False)
        self.df['Terakhir'] = self.df['Terakhir'].str.replace(',', '.', regex=False)
        self.df['Terakhir'] = self.df['Terakhir'].astype(float)
        self.df = self.df.iloc[::-1].reset_index(drop=True)
        self.df_normalisasi = self.data_normalisasi(self.df)
        self.split_data()
        return self

    def normalisasi(self, nilai, nilai_min, nilai_max):
        return (nilai - nilai_min) / (nilai_max - nilai_min)

    def denormalisasi(self, nilai, nilai_min, nilai_max):
        return (nilai * (nilai_max - nilai_min) + nilai_min)

    def data_normalisasi(self, df):
        self.nilai_min = df["Terakhir"].min()
        self.nilai_max = df["Terakhir"].max()
        return df["Terakhir"].apply(lambda x: self.normalisasi(x, self.nilai_min, self.nilai_max))

    def data_denormalisasi(self, output):
        return [self.denormalisasi(x, self.nilai_min, self.nilai_max) for x in output]

    def split_data(self):
        train_size = int(len(self.df_normalisasi) * self.train_percent)
        self.train_data = self.df_normalisasi.iloc[:train_size]
        self.test_data = self.df_normalisasi.iloc[train_size:].reset_index(drop=True)

    def create_pola(self, data, input_size=5):
        ndata = len(data)
        return [[float(data[i + j]) for j in range(input_size + 1)] for i in range(ndata - input_size)]

class GRU:
    def __init__(self, jml_hdnunt, batch_size):
        self.jml_hdnunt = jml_hdnunt
        self.batch_size = batch_size

    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0 if x < 0 else 1

    def tanh(self, x):
        try:
            return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        except OverflowError:
            return -1.0 if x < 0 else 1.0

    def transpose(self, matrix):
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        transposed = [[0 for _ in range(rows)] for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                transposed[j][i] = matrix[i][j]
        return transposed
    
    def gru_forward(self, pola_data, bobot_elang, ht_min=None):
        mse_per_elang = []
        total_batch = len(pola_data)
        current_batch_size = 0
        for elang_group in bobot_elang:
            tipe_solusi, bobot = elang_group
            W_r, U_r, W_z, U_z, W_h, U_h, W_y = bobot
            h_t = [[0 for _ in range(self.jml_hdnunt)] for _ in range(self.batch_size)]
            if ht_min is not None:
                for i in range(min(self.batch_size, len(ht_min))):
                    for j in range(self.jml_hdnunt):
                        if i < len(ht_min) and j < len(ht_min[0]):
                            h_t[i][j] = ht_min[i][j]

            prediksi = []
            for t in range(0, total_batch, self.batch_size):
                current_batch_size = min(self.batch_size, total_batch - t)
                x_t = [pola_data[i][:5] for i in range(t, t + current_batch_size)]
                x_t_transposed = self.transpose(x_t)
                h_t_temp = h_t[:current_batch_size]
                h_t_transposed = self.transpose(h_t_temp)

                dum_r_t = []
                for i in range(self.jml_hdnunt):
                    row = []
                    for j in range(current_batch_size):
                        sum_wx = sum(W_r[i][k] * x_t_transposed[k][j] for k in range(len(x_t_transposed)))
                        sum_uh = sum(U_r[i][k] * h_t_transposed[k][j] for k in range(len(h_t_transposed)))
                        row.append(self.sigmoid(sum_wx + sum_uh))
                    dum_r_t.append(row)
                r_t = self.transpose(dum_r_t)

                dum_z_t = []
                for i in range(self.jml_hdnunt):
                    row = []
                    for j in range(current_batch_size):
                        sum_wx = sum(W_z[i][k] * x_t_transposed[k][j] for k in range(len(x_t_transposed)))
                        sum_uh = sum(U_z[i][k] * h_t_transposed[k][j] for k in range(len(h_t_transposed)))
                        row.append(self.sigmoid(sum_wx + sum_uh))
                    dum_z_t.append(row)
                z_t = self.transpose(dum_z_t)

                dum_h_tilde = []
                for i in range(self.jml_hdnunt):
                    row = []
                    for j in range(current_batch_size):
                        sum_wx = sum(W_h[i][k] * x_t_transposed[k][j] for k in range(len(x_t_transposed)))
                        sum_uh = sum(U_h[i][k] * (dum_r_t[k][j] * h_t_transposed[k][j]) for k in range(self.jml_hdnunt))
                        row.append(self.tanh(sum_wx + sum_uh))
                    dum_h_tilde.append(row)
                h_tilde = self.transpose(dum_h_tilde)

                h_t = []
                for i in range(current_batch_size):
                    row_ht = []
                    for j in range(self.jml_hdnunt):
                        row_ht.append((1 - z_t[i][j]) * h_t_temp[i][j] + z_t[i][j] * h_tilde[i][j])
                    h_t.append(row_ht)

                Y_t = []
                for i in range(current_batch_size):
                    row = []
                    for j in range(1):
                        sum_hy = sum(h_t[i][k] * W_y[k][j] for k in range(self.jml_hdnunt))
                        row.append(self.sigmoid(sum_hy))
                    Y_t.append(row)

                prediksi.extend(Y_t)

            mse = 0
            if len(prediksi) > 0:
                for i, output in enumerate(prediksi):
                    actual = pola_data[i][5]
                    predicted = output[0]
                    mse += (predicted - actual) ** 2
                mse /= len(prediksi)
            mse_per_elang.append((tipe_solusi, mse))
        
        ht_min = [[0 for _ in range(self.jml_hdnunt)] for _ in range(self.batch_size)]
        if current_batch_size > 0:
            for i in range(current_batch_size):
                for j in range(self.jml_hdnunt):
                    ht_min[i][j] = h_t[i][j]

        return mse_per_elang, ht_min, prediksi

    def gru_forw_predict(self, x_t, ht_min, bobot):
        W_r, U_r, W_z, U_z, W_h, U_h, W_y = bobot
        x_t_transposed = self.transpose(x_t)
        h_t_transposed = self.transpose(ht_min)

        dum_r_t = []
        for i in range(self.jml_hdnunt):
            row = []
            for j in range(1):
                sum_wx = sum(W_r[i][k] * x_t_transposed[k][j] for k in range(len(x_t_transposed)))
                sum_uh = sum(U_r[i][k] * h_t_transposed[k][j] for k in range(len(h_t_transposed)))
                row.append(self.sigmoid(sum_wx + sum_uh))
            dum_r_t.append(row)
        r_t = self.transpose(dum_r_t)

        dum_z_t = []
        for i in range(self.jml_hdnunt):
            row = []
            for j in range(1):
                sum_wx = sum(W_z[i][k] * x_t_transposed[k][j] for k in range(len(x_t_transposed)))
                sum_uh = sum(U_z[i][k] * h_t_transposed[k][j] for k in range(len(h_t_transposed)))
                row.append(self.sigmoid(sum_wx + sum_uh))
            dum_z_t.append(row)
        z_t = self.transpose(dum_z_t)

        dum_h_tilde = []
        for i in range(self.jml_hdnunt):
            row = []
            for j in range(1):
                sum_wx = sum(W_h[i][k] * x_t_transposed[k][j] for k in range(len(x_t_transposed)))
                sum_uh = sum(U_h[i][k] * (dum_r_t[k][j] * h_t_transposed[k][j]) for k in range(self.jml_hdnunt))
                row.append(self.tanh(sum_wx + sum_uh))
            dum_h_tilde.append(row)
        h_tilde = self.transpose(dum_h_tilde)

        h_t = []
        for i in range(1):
            row = []
            for j in range(self.jml_hdnunt):
                row.append((1 - z_t[i][j]) * ht_min[i][j] + z_t[i][j] * h_tilde[i][j])
            h_t.append(row)

        Y_t = []
        for i in range(1):
            row = []
            for j in range(1):
                sum_hy = sum(h_t[i][k] * W_y[k][j] for k in range(self.jml_hdnunt))
                row.append(self.sigmoid(sum_hy))
            Y_t.append(row)

        return Y_t[0][0], h_t

class HHO:
    def __init__(self, elang, iterasi, elang_y, jml_hdnunt, batch_size, input_size=5):
        self.elang = elang
        self.iterasi = iterasi
        self.elang_y = elang_y
        self.jml_hdnunt = jml_hdnunt
        self.pop_elang = self.init_population()
        self.gru = GRU(jml_hdnunt, batch_size)
        self.input_size = input_size
        self.batch_size = batch_size

    def init_population(self):
        return [[round(random.uniform(-1, 1), 6) for _ in range(self.elang_y)] for _ in range(self.elang)]

    def hawks_conv(self, populasi):
        hasil_konversi = []
        for item in populasi:
            if isinstance(item, tuple):
                tipe_solusi, solusi = item
            else:
                tipe_solusi = "X"
                solusi = item
            
            if not solusi: continue

            jml_kolom_baru = len(solusi) // self.jml_hdnunt

            baris_konv = []
            for i in range(self.jml_hdnunt):
                potongan = solusi[i * jml_kolom_baru : (i + 1) * jml_kolom_baru]
                baris_konv.append(potongan)
            elang_konv = baris_konv

            bobot = []
            start = 0
            W_r = [elang_konv[row][start : start + self.input_size] for row in range(self.jml_hdnunt)]
            start += self.input_size
            bobot.append(W_r)
            U_r = [elang_konv[row][start : start + self.jml_hdnunt] for row in range(self.jml_hdnunt)]
            start += self.jml_hdnunt
            bobot.append(U_r)
            W_z = [elang_konv[row][start : start + self.input_size] for row in range(self.jml_hdnunt)]
            start += self.input_size
            bobot.append(W_z)
            U_z = [elang_konv[row][start : start + self.jml_hdnunt] for row in range(self.jml_hdnunt)]
            start += self.jml_hdnunt
            bobot.append(U_z)
            W_h = [elang_konv[row][start : start + self.input_size] for row in range(self.jml_hdnunt)]
            start += self.input_size
            bobot.append(W_h)
            U_h = [elang_konv[row][start : start + self.jml_hdnunt] for row in range(self.jml_hdnunt)]
            start += self.jml_hdnunt
            bobot.append(U_h)
            W_y = [[elang_konv[row][start]] for row in range(self.jml_hdnunt)]
            start += 1
            bobot.append(W_y)
            hasil_konversi.append((tipe_solusi, bobot))
        return hasil_konversi

    def hho(self, f_error, max_iter, populasi, ht_min, pola_training, batas_MSE):
        beta = 1.5
        sigma = ((math.gamma(1 + beta) * math.sin((math.pi * beta) / 2)) / (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        for iter_no in range(max_iter):
            rabbit_idx = f_error.index(min(f_error, key=lambda x: x[1]))
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
                        rand_hawk = populasi[random.randint(0, self.elang - 1)]
                        popbaru = [x_rabbit[j] - random.uniform(0, 1) * abs(rand_hawk[j] - 2 * random.uniform(0, 1) * populasi[i][j]) for j in range(self.elang_y)]
                    else:
                        popbaru = [(x_rabbit[j] - x_m[j]) - random.uniform(0, 1) * (-1 + random.uniform(0, 1) * (1 - (-1))) for j in range(self.elang_y)]
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
                        if abs(E) >= 0.5:
                            Y = [x_rabbit[j] - E * abs(J * x_rabbit[j] - populasi[i][j]) for j in range(self.elang_y)]
                            Z = [Y[j] + random.uniform(0, 1) * (0.01 * ((random.uniform(0, 1) * sigma) / (abs(random.uniform(0, 1)) ** (1 / beta)))) for j in range(self.elang_y)]
                            pop_new.append(("Y", Y))
                            pop_new.append(("Z", Z))
                        else:
                            Y = [x_rabbit[j] - E * abs(J * x_rabbit[j] - x_m[j]) for j in range(self.elang_y)]
                            Z = [Y[j] + random.uniform(0, 1) * (0.01 * ((random.uniform(0, 1) * sigma) / (abs(random.uniform(0, 1)) ** (1 / beta)))) for j in range(self.elang_y)]
                            pop_new.append(("Y", Y))
                            pop_new.append(("Z", Z))
            bobot_hho = self.hawks_conv(pop_new)
            mse_hho, _, _ = self.gru.gru_forward(pola_training, bobot_hho, ht_min)

            for i, (tipe_solusi, mse) in enumerate(mse_hho):
                if mse < batas_MSE:
                    print(f"MSE elang telah memenuhi batas: {mse} <= {batas_MSE}")
                    f_error = [(tipe_solusi, mse)]
                    populasi = [pop_new[i][1]]
                    self.pop_elang = populasi
                    return f_error, populasi, ht_min

            pop_mse_hho = []
            for i in range(len(pop_new)):
                tipe_solusi = mse_hho[i][0]
                mse = mse_hho[i][1]
                pop = pop_new[i][1]
                pop_mse_hho.append((tipe_solusi, mse, pop))

            mse_hho2 = []
            for i in range(0, len(pop_mse_hho)):
                if pop_mse_hho[i][0] == 'X':
                    mse_hho2.append(pop_mse_hho[i])
                else:
                    if i > 0 and pop_mse_hho[i-1][0] == 'Y':
                        if pop_mse_hho[i-1][1] < pop_mse_hho[i][1]:
                            mse_hho2.append(pop_mse_hho[i-1])
                        else:
                            mse_hho2.append(pop_mse_hho[i])

            for i in range(len(mse_hho2)):
                if mse_hho2[i][1] < f_error[i][1]:
                    f_error[i] = ('X', mse_hho2[i][1])
                    populasi[i] = mse_hho2[i][2]
            self.pop_elang = populasi
        return f_error, populasi, ht_min

class GRUHHO:
    def __init__(self, jml_hdnunt, batas_MSE, batch_size, maks_epoch, elang, iterasi):
        self.jml_hdnunt = jml_hdnunt
        self.batas_MSE = batas_MSE
        self.batch_size = batch_size
        self.maks_epoch = maks_epoch
        self.elang = elang
        self.iterasi = iterasi
        self.elang_y = (((5 + jml_hdnunt) * 3) + 1) * jml_hdnunt
        self.gru = GRU(jml_hdnunt, batch_size)
        self.hho = HHO(elang, iterasi, self.elang_y, jml_hdnunt, batch_size)
        self.best_weights = None
        self.best_mse = float('inf')
        self.training_log = []

    def training_gru(self, pola_training):
        self.training_log = []
        epoch = 0
        terkecil_mse = float('inf')
        ht_min = None
        pop_elang = self.hho.pop_elang
        while terkecil_mse >= self.batas_MSE and epoch < self.maks_epoch:
            bobot_elang = self.hho.hawks_conv(pop_elang)
            mse_elang, _, _ = self.gru.gru_forward(pola_training, bobot_elang, ht_min)
            mse_elang, pop_elang, ht_min = self.hho.hho(mse_elang, self.iterasi, pop_elang, ht_min, pola_training, self.batas_MSE)
            
            log_entry = f"Epoch {epoch + 1}/{self.maks_epoch}, MSE: {mse_elang}"
            print(log_entry)
            self.training_log.append(log_entry)

            for i, (tipe_solusi, mse) in enumerate(mse_elang):
                if mse < terkecil_mse:
                    terkecil_mse = mse
                    self.best_mse = (tipe_solusi, mse)
                    self.best_weights = [pop_elang[i]]
            epoch += 1
        return self.best_weights, self.best_mse

    def test_gru(self, pola_testing, ht_min=None):
        bobot_test = self.hho.hawks_conv(self.best_weights)
        mse_test, _, prediksi = self.gru.gru_forward(pola_testing, bobot_test, ht_min)
        return prediksi, mse_test[0][1]

    def forw_predict(self, data_awal, n_prediksi, preprocessor):
        data_awal_norm = [preprocessor.normalisasi(x, preprocessor.nilai_min, preprocessor.nilai_max) for x in data_awal]
        bobot_maju = self.hho.hawks_conv(self.best_weights)[0][1]
        prediksi_norm = []
        ht_min = [[0] * self.jml_hdnunt] * 1  
        for _ in range(n_prediksi):
            x_t = [data_awal_norm[-5:]]
            prediksi_satu_langkah, ht_min = self.gru.gru_forw_predict(x_t, ht_min, bobot_maju)
            prediksi_norm.append(prediksi_satu_langkah)
            data_awal_norm.append(prediksi_satu_langkah)
            data_awal_norm.pop(0)
        prediksi = [preprocessor.denormalisasi(p, preprocessor.nilai_min, preprocessor.nilai_max) for p in prediksi_norm]
        return prediksi