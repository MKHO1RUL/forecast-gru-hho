/**
 * GRU-HHO Forecasting Engine in TypeScript
 * Ported 1:1 from Python backend (model.py) for standalone execution in Next.js.
 */

export interface DataRecord {
  Tanggal: string;
  Terakhir: number;
}

export class DataPreprocessing {
  trainPercent: number;
  df: DataRecord[] = [];
  dfNormalisasi: number[] = [];
  trainData: number[] = [];
  testData: number[] = [];
  nilaiMin: number = 0;
  nilaiMax: number = 0;

  constructor(trainPercent: number = 0.7) {
    this.trainPercent = trainPercent;
  }

  loadAndPreprocess(data: DataRecord[]): this {
    this.df = data.map((d) => ({
      Tanggal: d.Tanggal,
      Terakhir: Number(d.Terakhir),
    }));

    const prices = this.df.map((d) => d.Terakhir);
    this.nilaiMin = Math.min(...prices);
    this.nilaiMax = Math.max(...prices);

    this.dfNormalisasi = prices.map((v) => this._normalisasi(v, this.nilaiMin, this.nilaiMax));
    this._splitData();
    return this;
  }

  _normalisasi(nilai: number, nilaiMin: number, nilaiMax: number): number {
    if (nilaiMax - nilaiMin === 0) return 0;
    return (nilai - nilaiMin) / (nilaiMax - nilaiMin);
  }

  _denormalisasi(nilai: number, nilaiMin: number, nilaiMax: number): number {
    return nilai * (nilaiMax - nilaiMin) + nilaiMin;
  }

  dataDenormalisasi(output: number[]): number[] {
    return output.map((x) => this._denormalisasi(x, this.nilaiMin, this.nilaiMax));
  }

  _splitData(): void {
    const trainSize = Math.floor(this.dfNormalisasi.length * this.trainPercent);
    this.trainData = this.dfNormalisasi.slice(0, trainSize);
    this.testData = this.dfNormalisasi.slice(trainSize);
  }

  createPola(data: number[], inputSize: number = 5): [number[][], number[]] {
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < data.length - inputSize; i++) {
      X.push(data.slice(i, i + inputSize));
      y.push(data[i + inputSize]);
    }
    return [X, y];
  }
}

// Helper matrix operations
function matMul(A: number[][], B: number[][]): number[][] {
  const rowsA = A.length;
  const colsA = A[0].length;
  const colsB = B[0].length;
  const result: number[][] = Array.from({ length: rowsA }, () => Array(colsB).fill(0));

  for (let i = 0; i < rowsA; i++) {
    for (let k = 0; k < colsA; k++) {
      const aVal = A[i][k];
      for (let j = 0; j < colsB; j++) {
        result[i][j] += aVal * B[k][j];
      }
    }
  }
  return result;
}

function transpose(A: number[][]): number[][] {
  const rows = A.length;
  const cols = A[0].length;
  const result: number[][] = Array.from({ length: cols }, () => Array(rows).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = A[i][j];
    }
  }
  return result;
}

function gamma(z: number): number {
  // Lanczos approximation for Gamma function
  const p = [
    676.5203681218851, -1259.1392167224028, 771.3234287776531,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.984369578019571e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
  z -= 1;
  let x = 0.99999999999980993;
  for (let i = 0; i < p.length; i++) {
    x += p[i] / (z + i + 1);
  }
  const t = z + p.length - 0.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
}

function randomUniform(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export type HawkWeightMatrix = number[][][]; // [W_r, U_r, W_z, U_z, W_h, U_h, W_y]
export type HawkSolution = [string, HawkWeightMatrix];

export class GRU {
  jmlHdnunt: number;
  batchSize: number;

  constructor(jmlHdnunt: number, batchSize: number) {
    this.jmlHdnunt = jmlHdnunt;
    this.batchSize = batchSize;
  }

  _sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  _sigmoidMat(A: number[][]): number[][] {
    return A.map((row) => row.map((v) => this._sigmoid(v)));
  }

  _tanhMat(A: number[][]): number[][] {
    return A.map((row) => row.map((v) => Math.tanh(v)));
  }

  gruForward(
    X_data: number[][],
    y_data: number[],
    bobotElang: HawkSolution[],
    htMin: number[][] | null = null
  ): [Array<[string, number]>, number[][], number[]] {
    const msePerElang: Array<[string, number]> = [];
    const totalSamples = X_data.length;
    let finalHt: number[][] = [];

    for (const [tipeSolusi, bobot] of bobotElang) {
      const [W_r, U_r, W_z, U_z, W_h, U_h, W_y] = bobot;

      const h_t: number[][] = Array.from({ length: this.batchSize }, () => Array(this.jmlHdnunt).fill(0));
      if (htMin) {
        const rows = Math.min(h_t.length, htMin.length);
        const cols = Math.min(h_t[0].length, htMin[0].length);
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            h_t[r][c] = htMin[r][c];
          }
        }
      }

      const prediksi: number[] = [];

      for (let t = 0; t < totalSamples; t += this.batchSize) {
        const end_t = Math.min(t + this.batchSize, totalSamples);
        const x_batch = X_data.slice(t, end_t);
        const batchLen = x_batch.length;

        const h_prev = h_t.slice(0, batchLen);

        // r_t = sigmoid( x_batch @ W_r.T + h_prev @ U_r.T )
        const x_Wr = matMul(x_batch, transpose(W_r));
        const h_Ur = matMul(h_prev, transpose(U_r));
        const r_t = this._sigmoidMat(x_Wr.map((row, i) => row.map((v, j) => v + h_Ur[i][j])));

        // z_t = sigmoid( x_batch @ W_z.T + h_prev @ U_z.T )
        const x_Wz = matMul(x_batch, transpose(W_z));
        const h_Uz = matMul(h_prev, transpose(U_z));
        const z_t = this._sigmoidMat(x_Wz.map((row, i) => row.map((v, j) => v + h_Uz[i][j])));

        // r_t * h_prev
        const r_hprev = r_t.map((row, i) => row.map((v, j) => v * h_prev[i][j]));
        const x_Wh = matMul(x_batch, transpose(W_h));
        const rh_Uh = matMul(r_hprev, transpose(U_h));
        const h_tilde = this._tanhMat(x_Wh.map((row, i) => row.map((v, j) => v + rh_Uh[i][j])));

        // h_t_current = (1 - z_t) * h_prev + z_t * h_tilde
        const h_t_current = z_t.map((row, i) =>
          row.map((z, j) => (1 - z) * h_prev[i][j] + z * h_tilde[i][j])
        );

        // y_pred_batch = sigmoid( h_t_current @ W_y )
        const y_pred_batch = this._sigmoidMat(matMul(h_t_current, W_y));
        for (let i = 0; i < y_pred_batch.length; i++) {
          prediksi.push(y_pred_batch[i][0]);
        }

        for (let i = 0; i < batchLen; i++) {
          h_t[i] = h_t_current[i];
        }
      }

      let mse = 0;
      if (prediksi.length > 0) {
        let sumErrSq = 0;
        for (let i = 0; i < prediksi.length; i++) {
          sumErrSq += Math.pow(prediksi[i] - y_data[i], 2);
        }
        mse = sumErrSq / prediksi.length;
      } else {
        mse = Infinity;
      }

      msePerElang.push([tipeSolusi, mse]);
      finalHt = h_t;
    }

    // Return prediction array
    const [, lastBobot] = bobotElang[0];
    const singlePred: number[] = [];
    if (bobotElang.length === 1) {
      const [W_r, U_r, W_z, U_z, W_h, U_h, W_y] = lastBobot;
      const h_t: number[][] = Array.from({ length: this.batchSize }, () => Array(this.jmlHdnunt).fill(0));
      for (let t = 0; t < totalSamples; t += this.batchSize) {
        const end_t = Math.min(t + this.batchSize, totalSamples);
        const x_batch = X_data.slice(t, end_t);
        const h_prev = h_t.slice(0, x_batch.length);

        const x_Wr = matMul(x_batch, transpose(W_r));
        const h_Ur = matMul(h_prev, transpose(U_r));
        const r_t = this._sigmoidMat(x_Wr.map((row, i) => row.map((v, j) => v + h_Ur[i][j])));

        const x_Wz = matMul(x_batch, transpose(W_z));
        const h_Uz = matMul(h_prev, transpose(U_z));
        const z_t = this._sigmoidMat(x_Wz.map((row, i) => row.map((v, j) => v + h_Uz[i][j])));

        const r_hprev = r_t.map((row, i) => row.map((v, j) => v * h_prev[i][j]));
        const x_Wh = matMul(x_batch, transpose(W_h));
        const rh_Uh = matMul(r_hprev, transpose(U_h));
        const h_tilde = this._tanhMat(x_Wh.map((row, i) => row.map((v, j) => v + rh_Uh[i][j])));

        const h_t_current = z_t.map((row, i) =>
          row.map((z, j) => (1 - z) * h_prev[i][j] + z * h_tilde[i][j])
        );

        const y_pred_batch = this._sigmoidMat(matMul(h_t_current, W_y));
        for (let i = 0; i < y_pred_batch.length; i++) {
          singlePred.push(y_pred_batch[i][0]);
        }
        for (let i = 0; i < x_batch.length; i++) {
          h_t[i] = h_t_current[i];
        }
      }
      return [msePerElang, finalHt, singlePred];
    }

    return [msePerElang, finalHt, []];
  }

  gruForwPredict(x_t: number[][], htMin: number[][], bobot: HawkWeightMatrix): [number, number[][]] {
    const [W_r, U_r, W_z, U_z, W_h, U_h, W_y] = bobot;

    const x_Wr = matMul(x_t, transpose(W_r));
    const h_Ur = matMul(htMin, transpose(U_r));
    const r_t = this._sigmoidMat(x_Wr.map((row, i) => row.map((v, j) => v + h_Ur[i][j])));

    const x_Wz = matMul(x_t, transpose(W_z));
    const h_Uz = matMul(htMin, transpose(U_z));
    const z_t = this._sigmoidMat(x_Wz.map((row, i) => row.map((v, j) => v + h_Uz[i][j])));

    const r_hprev = r_t.map((row, i) => row.map((v, j) => v * htMin[i][j]));
    const x_Wh = matMul(x_t, transpose(W_h));
    const rh_Uh = matMul(r_hprev, transpose(U_h));
    const h_tilde = this._tanhMat(x_Wh.map((row, i) => row.map((v, j) => v + rh_Uh[i][j])));

    const h_t = z_t.map((row, i) =>
      row.map((z, j) => (1 - z) * htMin[i][j] + z * h_tilde[i][j])
    );

    const y_t = this._sigmoidMat(matMul(h_t, W_y));
    return [y_t[0][0], h_t];
  }
}

export class HHO {
  elang: number;
  iterasi: number;
  elangY: number;
  jmlHdnunt: number;
  batchSize: number;
  inputSize: number;
  popElang: number[][];
  gru: GRU;

  constructor(
    elang: number,
    iterasi: number,
    elangY: number,
    jmlHdnunt: number,
    batchSize: number,
    inputSize: number = 5
  ) {
    this.elang = elang;
    this.iterasi = iterasi;
    this.elangY = elangY;
    this.jmlHdnunt = jmlHdnunt;
    this.batchSize = batchSize;
    this.inputSize = inputSize;
    this.popElang = this._initPopulation();
    this.gru = new GRU(jmlHdnunt, batchSize);
  }

  _initPopulation(): number[][] {
    const pop: number[][] = [];
    for (let i = 0; i < this.elang; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.elangY; j++) {
        row.push(Number(randomUniform(-1, 1).toFixed(6)));
      }
      pop.push(row);
    }
    return pop;
  }

  hawksConv(populasi: Array<number[] | [string, number[]]>): HawkSolution[] {
    const hasilKonversi: HawkSolution[] = [];
    const shapes: Record<string, [number, number]> = {
      W_r: [this.jmlHdnunt, this.inputSize],
      U_r: [this.jmlHdnunt, this.jmlHdnunt],
      W_z: [this.jmlHdnunt, this.inputSize],
      U_z: [this.jmlHdnunt, this.jmlHdnunt],
      W_h: [this.jmlHdnunt, this.inputSize],
      U_h: [this.jmlHdnunt, this.jmlHdnunt],
      W_y: [this.jmlHdnunt, 1],
    };

    for (const item of populasi) {
      let tipeSolusi = 'X';
      let solusi: number[] = [];

      if (Array.isArray(item) && typeof item[0] === 'string') {
        tipeSolusi = item[0] as string;
        solusi = item[1] as number[];
      } else {
        solusi = item as number[];
      }

      if (!solusi || solusi.length === 0) continue;

      const bobot: HawkWeightMatrix = [];
      let start = 0;
      for (const shape of Object.values(shapes)) {
        const [rows, cols] = shape;
        const numElements = rows * cols;
        const flatWeights = solusi.slice(start, start + numElements);
        const matrix: number[][] = [];
        for (let r = 0; r < rows; r++) {
          matrix.push(flatWeights.slice(r * cols, (r + 1) * cols));
        }
        bobot.push(matrix);
        start += numElements;
      }

      hasilKonversi.push([tipeSolusi, bobot]);
    }
    return hasilKonversi;
  }

  hho(
    fError: Array<[string, number]>,
    maxIter: number,
    populasi: number[][],
    htMin: number[][],
    X_train: number[][],
    y_train: number[],
    batasMSE: number
  ): [Array<[string, number]>, number[][]] {
    const beta = 1.5;
    const num = gamma(1 + beta) * Math.sin((Math.PI * beta) / 2);
    const den = gamma((1 + beta) / 2) * beta * Math.pow(2, (beta - 1) / 2);
    const sigma = Math.pow(num / den, 1 / beta);

    let currentPop = populasi.map((row) => [...row]);
    let currentFError = fError.map((item) => [...item] as [string, number]);

    for (let iterNo = 0; iterNo < maxIter; iterNo++) {
      let rabbitIdx = 0;
      let minErr = Infinity;
      for (let i = 0; i < currentFError.length; i++) {
        if (currentFError[i][1] < minErr) {
          minErr = currentFError[i][1];
          rabbitIdx = i;
        }
      }

      const xRabbit = [...currentPop[rabbitIdx]];
      const xM: number[] = [];
      for (let j = 0; j < this.elangY; j++) {
        let sum = 0;
        for (let k = 0; k < this.elang; k++) {
          sum += currentPop[k][j];
        }
        xM.push(sum / this.elang);
      }

      const popNew: Array<[string, number[]]> = [];

      for (let i = 0; i < this.elang; i++) {
        const E_0 = randomUniform(-1, 1);
        const J = 2 * (1 - randomUniform(0, 1));
        const E = 2 * E_0 * (1 - iterNo / maxIter);

        if (Math.abs(E) >= 1) {
          const q = randomUniform(0, 1);
          let popbaru: number[] = [];
          if (q >= 0.5) {
            const randHawkIdx = randomInt(0, this.elang - 1);
            popbaru = xRabbit.map((rj, j) => rj - randomUniform(0, 1) * Math.abs(currentPop[randHawkIdx][j] - 2 * randomUniform(0, 1) * currentPop[i][j]));
          } else {
            popbaru = xRabbit.map((rj, j) => (rj - xM[j]) - randomUniform(0, 1) * 2 - 1);
          }
          popNew.push(['X', popbaru]);
        } else {
          const r = randomUniform(0, 1);
          if (r >= 0.5) {
            let popbaru: number[] = [];
            if (Math.abs(E) >= 0.5) {
              popbaru = xRabbit.map((rj, j) => rj - currentPop[i][j] - E * Math.abs(J * rj - currentPop[i][j]));
            } else {
              popbaru = xRabbit.map((rj, j) => rj - E * Math.abs(rj - currentPop[i][j]));
            }
            popNew.push(['X', popbaru]);
          } else {
            const lfStep = Array.from({ length: this.elangY }, () =>
              0.01 * ((randomUniform(0, 1) * sigma) / Math.pow(Math.abs(randomUniform(0, 1)), 1 / beta))
            );

            let Y: number[] = [];
            if (Math.abs(E) >= 0.5) {
              Y = xRabbit.map((rj, j) => rj - E * Math.abs(J * rj - currentPop[i][j]));
            } else {
              Y = xRabbit.map((rj, j) => rj - E * Math.abs(J * rj - xM[j]));
            }

            const bobotY = this.hawksConv([['Y', Y]]);
            const [mseY] = this.gru.gruForward(X_train, y_train, bobotY, htMin);

            if (mseY[0][1] < currentFError[i][1]) {
              popNew.push(['Y', Y]);
            } else {
              const Z = Y.map((yj, j) => yj + lfStep[j] * randomUniform(0, 1));
              popNew.push(['Z', Z]);
            }
          }
        }
      }

      const bobotHho = this.hawksConv(popNew);
      const [mseHho] = this.gru.gruForward(X_train, y_train, bobotHho, htMin);

      const nextPopulasi: number[][] = [];
      const nextFError: Array<[string, number]> = [];

      for (let i = 0; i < this.elang; i++) {
        if (mseHho[i][1] < currentFError[i][1]) {
          nextFError.push(mseHho[i]);
          nextPopulasi.push(popNew[i][1]);
        } else {
          nextFError.push(currentFError[i]);
          nextPopulasi.push(currentPop[i]);
        }
      }

      currentPop = nextPopulasi;
      currentFError = nextFError;

      const currentMinMse = Math.min(...currentFError.map((e) => e[1]));
      if (currentMinMse < batasMSE) {
        break;
      }
    }

    this.popElang = currentPop;
    return [currentFError, currentPop];
  }
}

export class GRUHHO {
  jmlHdnunt: number;
  batasMSE: number;
  batchSize: number;
  maksEpoch: number;
  elang: number;
  iterasi: number;
  inputSize: number;
  elangY: number;
  gru: GRU;
  hho: HHO;
  bestWeights: number[] | null = null;
  bestMseInfo: [string, number] = ['', Infinity];
  trainingLog: string[] = [];

  constructor(
    jmlHdnunt: number,
    batasMSE: number,
    batchSize: number,
    maksEpoch: number,
    elang: number,
    iterasi: number,
    inputSize: number = 5
  ) {
    this.jmlHdnunt = jmlHdnunt;
    this.batasMSE = batasMSE;
    this.batchSize = batchSize;
    this.maksEpoch = maksEpoch;
    this.elang = elang;
    this.iterasi = iterasi;
    this.inputSize = inputSize;

    this.elangY = (this.inputSize * 3 + this.jmlHdnunt * 3 + 1) * this.jmlHdnunt;
    this.gru = new GRU(jmlHdnunt, batchSize);
    this.hho = new HHO(elang, iterasi, this.elangY, jmlHdnunt, batchSize, this.inputSize);
  }

  async *trainingGruGenerator(
    X_train: number[][],
    y_train: number[]
  ): AsyncGenerator<string, void, unknown> {
    this.trainingLog = [];
    let epoch = 0;
    let htMin: number[][] | null = null;
    let popElang = this.hho.popElang;

    const bobotElang = this.hho.hawksConv(popElang);
    const [initialFError, htMinUpdated] = this.gru.gruForward(X_train, y_train, bobotElang, htMin);
    let fError = initialFError;
    htMin = htMinUpdated;

    const initialMse = Math.min(...fError.map((e) => e[1]));
    yield `Initial MSE before training: ${initialMse.toFixed(6)}`;

    while (Math.min(...fError.map((e) => e[1])) > this.batasMSE && epoch < this.maksEpoch) {
      epoch++;

      const [updatedFError, updatedPop] = this.hho.hho(
        fError,
        this.iterasi,
        popElang,
        htMin,
        X_train,
        y_train,
        this.batasMSE
      );
      fError = updatedFError;
      popElang = updatedPop;

      const currentBestMse = Math.min(...fError.map((e) => e[1]));
      const logEntry = `Epoch ${epoch}/${this.maksEpoch}, Best MSE: ${currentBestMse.toFixed(6)}`;
      this.trainingLog.push(logEntry);
      yield logEntry;

      // Yield control briefly to keep UI responsive
      await new Promise((resolve) => setTimeout(resolve, 5));

      let bestIdxEpoch = 0;
      let minErrEpoch = Infinity;
      for (let i = 0; i < fError.length; i++) {
        if (fError[i][1] < minErrEpoch) {
          minErrEpoch = fError[i][1];
          bestIdxEpoch = i;
        }
      }

      const bobotTerbaikEpoch = this.hho.hawksConv([popElang[bestIdxEpoch]]);
      const [, htMinBest] = this.gru.gruForward(X_train, y_train, bobotTerbaikEpoch, htMin);
      htMin = htMinBest;
    }

    let bestIdx = 0;
    let minErr = Infinity;
    for (let i = 0; i < fError.length; i++) {
      if (fError[i][1] < minErr) {
        minErr = fError[i][1];
        bestIdx = i;
      }
    }

    this.bestMseInfo = fError[bestIdx];
    this.bestWeights = popElang[bestIdx];
  }

  testGru(X_test: number[][], y_test: number[]): [number[], number] {
    if (!this.bestWeights) {
      throw new Error('Model has not been trained yet.');
    }

    const bobotTest = this.hho.hawksConv([this.bestWeights]);
    const [, , prediksi] = this.gru.gruForward(X_test, y_test, bobotTest);

    let mseTest = 0;
    if (prediksi.length > 0) {
      let sumErrSq = 0;
      for (let i = 0; i < prediksi.length; i++) {
        sumErrSq += Math.pow(prediksi[i] - y_test[i], 2);
      }
      mseTest = sumErrSq / prediksi.length;
    } else {
      mseTest = Infinity;
    }

    return [prediksi, mseTest];
  }

  forwPredict(
    dataTerakhir: number[],
    nHari: number,
    preprocessor: DataPreprocessing
  ): number[] {
    if (!this.bestWeights) {
      throw new Error('Model has not been trained yet.');
    }

    const dataTerakhirNorm = dataTerakhir.map((x) =>
      preprocessor._normalisasi(x, preprocessor.nilaiMin, preprocessor.nilaiMax)
    );
    const bobotMaju = this.hho.hawksConv([this.bestWeights])[0][1];

    const prediksiNorm: number[] = [];
    let htMin: number[][] = Array.from({ length: 1 }, () => Array(this.jmlHdnunt).fill(0));
    const currentInput = [...dataTerakhirNorm];

    for (let day = 0; day < nHari; day++) {
      const sliceInput = currentInput.slice(-this.inputSize);
      const x_t = [sliceInput];

      const [prediksiSatuLangkah, htMinNext] = this.gru.gruForwPredict(x_t, htMin, bobotMaju);
      prediksiNorm.push(prediksiSatuLangkah);
      htMin = htMinNext;

      currentInput.push(prediksiSatuLangkah);
      currentInput.shift();
    }

    return preprocessor.dataDenormalisasi(prediksiNorm);
  }
}
