"use client";

import { useEffect, useMemo, useCallback, useRef } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  TimeScale,
  ScriptableContext,
  ChartOptions,
  Plugin,
} from "chart.js";
import "chartjs-adapter-date-fns";
import { sub } from "date-fns";

import { Header } from "../components/header";
import { ControlsSidebar } from "../components/controlsidebar";
import { MainContent } from "../components/maincontent";
import { InfoSidebar } from "../components/infosidebar";
import { useForexApp, PlotDataset } from "../hooks/useforexapp";
import { DataPoint, PredictionData, Theme } from "../types";

import { DataPreprocessing, GRUHHO, DataRecord } from "../lib/model";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  TimeScale
);

const FOREX_PAIRS = [
  "EURUSD=X", "GBPUSD=X", "USDJPY=X", "GBPJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"
];

export default function Home() {
  const { state, dispatch, refs, handlers } = useForexApp();
  const { chartRef, paramContainerRef, logContainerRef } = refs;

  const preprocessorRef = useRef<DataPreprocessing | null>(null);
  const modelRef = useRef<GRUHHO | null>(null);
  const rawRecordsRef = useRef<DataRecord[]>([]);

  useEffect(() => {
    let zoomPlugin: Plugin | undefined;
    const registerPlugin = async () => {
      const zoomPluginModule = await import('chartjs-plugin-zoom');
      zoomPlugin = zoomPluginModule.default;
      ChartJS.register(zoomPlugin);
    };
    registerPlugin();
    return () => {
      if (zoomPlugin) {
        ChartJS.unregister(zoomPlugin);
      }
    };
  }, []);

  const trainingParamsMemo = useMemo(() => ({
    jml_hdnunt: state.params.jml_hdnunt,
    batas_MSE: state.params.batas_MSE,
    batch_size: state.params.batch_size,
    maks_epoch: state.params.maks_epoch,
    elang: state.params.elang,
    iterasi: state.params.iterasi,
  }), [
    state.params.jml_hdnunt,
    state.params.batas_MSE,
    state.params.batch_size,
    state.params.maks_epoch,
    state.params.elang,
    state.params.iterasi,
  ]);

  const prevTrainingParamsRef = useRef(trainingParamsMemo);

  useEffect(() => {
    if (JSON.stringify(prevTrainingParamsRef.current) !== JSON.stringify(trainingParamsMemo)) {
      if (state.isTrained) {
        dispatch({ type: 'SET_IS_TRAINED', payload: false });
        dispatch({ type: 'SET_STATUS', payload: 'Parameters changed. Please retrain the model.' });
        dispatch({ type: 'ADD_LOG', payload: 'Parameters changed. Model needs to be retrained.' });
      }
    }

    prevTrainingParamsRef.current = trainingParamsMemo;
  }, [trainingParamsMemo, state.isTrained, dispatch]);

  const handleLoadData = useCallback(async () => {
    if (!state.selectedPair) {
      alert('Please select a forex pair.');
      return;
    }

    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_STATUS', payload: `Loading data for ${state.selectedPair}...` });
    dispatch({ type: 'ADD_LOG', payload: `Fetching 5-year historical data for ${state.selectedPair}...` });
    dispatch({ type: 'SET_TABLE_DATA', payload: { data: [], maxBatchSize: 0 }});

    try {
      const res = await fetch(`/api/get-data?pair=${state.selectedPair}`);

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || `Failed to load data for ${state.selectedPair}.`);
      }

      const data = await res.json();
      rawRecordsRef.current = data.data;
      preprocessorRef.current = null;
      modelRef.current = null;

      dispatch({ type: 'SET_TABLE_DATA', payload: { data: data.data, maxBatchSize: data.max_batch_size } });
      dispatch({ type: 'SET_STATUS', payload: 'Data loaded successfully.' });
      dispatch({ type: 'ADD_LOG', payload: 'Data loaded successfully. Ready for training.' });
      
      const initialDataPoints = data.data.map((point: DataPoint) => ({
        x: new Date(point.Tanggal).getTime(),
        y: Number(point.Terakhir),
      }));

      if (initialDataPoints.length > 0) {
        const xValues = initialDataPoints.map((p: { x: number; y: number }) => p.x);
        const minDate = Math.min(...xValues);
        const maxDate = Math.max(...xValues);
        dispatch({ type: 'SET_DATE_RANGE', payload: { min: minDate, max: maxDate } });
      } else {
        dispatch({ type: 'SET_DATE_RANGE', payload: { min: null, max: null } });
      }

      const allDataPoints = data.data.map((point: DataPoint) => ({
        x: new Date(point.Tanggal).getTime(),
        y: Number(point.Terakhir),
      }));

      const trainSize = Math.floor(allDataPoints.length * 0.7);
      const trainingPoints = allDataPoints.slice(0, trainSize);
      const testingPoints = allDataPoints.slice(trainSize - 1);

      dispatch({ type: 'SET_PLOT_DATA', payload: [
          {
            label: 'Training Data',
            data: trainingPoints,
            borderColor: 'blue',
            tension: 0.1,
            pointRadius: 0, 
            pointHoverRadius: 5,
            fill: true,
            backgroundColor: (context: ScriptableContext<'line'>) => {
              const chart = context.chart;
              const { ctx, chartArea } = chart;
              if (!chartArea) return undefined;
              const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
              gradient.addColorStop(0, 'rgba(0, 0, 255, 0.5)');
              gradient.addColorStop(1, 'rgba(0, 0, 255, 0)');
              return gradient;
            },
          },
          {
            label: 'Testing Data (Actual)',
            data: testingPoints,
            borderColor: 'green',
            tension: 0.1,
            pointRadius: 0,
            pointHoverRadius: 5,
            fill: true,
            backgroundColor: (context: ScriptableContext<'line'>) => {
              const chart = context.chart;
              const { ctx, chartArea } = chart;
              if (!chartArea) return undefined;
              const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
              gradient.addColorStop(0, 'rgba(0, 128, 0, 0.5)');
              gradient.addColorStop(1, 'rgba(0, 128, 0, 0)');
              return gradient;
            },
          },
        ],
      });

    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred during data fetching.';
      dispatch({ type: 'SET_STATUS', payload: `Error: ${errorMessage}` });
      dispatch({ type: 'ADD_LOG', payload: `Error: ${errorMessage}` });
      alert(`Error: ${errorMessage}`);
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [state.selectedPair, dispatch]);

  const handleTrain = useCallback(async () => {
    const requiredParams = [
      'jml_hdnunt', 'batas_MSE', 'batch_size', 'maks_epoch', 'elang', 'iterasi'
    ] as const;

    const missingParams = requiredParams.filter(p => !trainingParamsMemo[p] || trainingParamsMemo[p].trim() === '');

    if (missingParams.length > 0) {
      const message = `Please fill all parameter fields before training.`;
      dispatch({ type: 'SET_STATUS', payload: message });
      dispatch({ type: 'ADD_LOG', payload: `Error: ${message}` });
      alert(message);
      return;
    }

    if (!rawRecordsRef.current || rawRecordsRef.current.length === 0) {
      alert("Data tidak ditemukan. Silakan muat data melalui tombol Load Data terlebih dahulu.");
      return;
    }

    dispatch({ type: 'SET_TRAINING', payload: true });
    dispatch({ type: 'SET_STATUS', payload: 'Training in progress... Please wait.' });
    dispatch({ type: 'ADD_LOG', payload: 'Training started...' });

    dispatch({
      type: 'UPDATE_PLOT_DATA', payload: (datasets: PlotDataset[]) => datasets.filter(
        (ds) => !ds.label?.startsWith('Prediction (')
      )
    });

    try {
      const inputSize = 5;
      const preprocessor = new DataPreprocessing(0.7);
      preprocessor.loadAndPreprocess(rawRecordsRef.current);
      preprocessorRef.current = preprocessor;

      const [X_train, y_train] = preprocessor.createPola(preprocessor.trainData, inputSize);

      const model = new GRUHHO(
        parseInt(trainingParamsMemo.jml_hdnunt),
        parseFloat(trainingParamsMemo.batas_MSE),
        parseInt(trainingParamsMemo.batch_size),
        parseInt(trainingParamsMemo.maks_epoch),
        parseInt(trainingParamsMemo.elang),
        parseInt(trainingParamsMemo.iterasi),
        inputSize
      );
      modelRef.current = model;

      for await (const log of model.trainingGruGenerator(X_train, y_train)) {
        dispatch({ type: 'ADD_LOG', payload: log });
      }

      dispatch({ type: 'SET_STATUS', payload: 'Training complete!' });
      dispatch({ type: 'ADD_LOG', payload: `Training complete! Best MSE: ${model.bestMseInfo[1].toFixed(6)}` });
      dispatch({ type: 'SET_IS_TRAINED', payload: true });

    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred during training.';
      dispatch({ type: 'SET_STATUS', payload: `Error: ${errorMessage}` });
      dispatch({ type: 'ADD_LOG', payload: `Error: ${errorMessage}` });
      alert(`Error: ${errorMessage}`);
    } finally {
      dispatch({ type: 'SET_TRAINING', payload: false });
    }
  }, [trainingParamsMemo, dispatch]);

  const handleTest = useCallback(async () => {
    dispatch({ type: 'SET_TESTING', payload: true });
    dispatch({ type: 'SET_STATUS', payload: 'Testing in progress...' });
    dispatch({ type: 'ADD_LOG', payload: 'Testing started...' });

    try {
      const model = modelRef.current;
      const preprocessor = preprocessorRef.current;
      if (!model || !preprocessor || !model.bestWeights) {
        throw new Error('Model atau preprocessor belum siap. Latih model terlebih dahulu.');
      }

      const inputSize = model.inputSize;
      const [X_test, y_test] = preprocessor.createPola(preprocessor.testData, inputSize);
      const [prediksiNorm, mseTest] = model.testGru(X_test, y_test);

      const hasilPrediksiDenorm = preprocessor.dataDenormalisasi(prediksiNorm);

      const trainSize = preprocessor.trainData.length;
      const startIndex = trainSize + inputSize;
      const endIndex = startIndex + y_test.length;

      const dates = preprocessor.df.slice(startIndex, endIndex).map((d) => d.Tanggal);

      dispatch({ type: 'SET_STATUS', payload: 'Testing complete.' });
      dispatch({ type: 'ADD_LOG', payload: `Testing complete! MSE: ${mseTest.toFixed(6)}` });

      dispatch({
        type: 'UPDATE_PLOT_DATA', payload: (prevDatasets: PlotDataset[]) => {
          const baseDatasets = prevDatasets.filter(
            (ds) => ds.label !== 'Testing Prediction'
          );
          return [
            ...baseDatasets,
            {
              label: 'Testing Prediction',
              data: dates.map((date: string, index: number) => ({
                x: new Date(date).getTime(),
                y: hasilPrediksiDenorm[index],
              })),
              borderColor: 'orange',
              tension: 0.1,
              pointRadius: 0,
              pointHoverRadius: 5,
              fill: true,
              backgroundColor: (context: ScriptableContext<'line'>) => {
                const chart = context.chart;
                const { ctx, chartArea } = chart;
                if (!chartArea) return undefined;
                const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                gradient.addColorStop(0, 'rgba(255, 165, 0, 0.5)');
                gradient.addColorStop(1, 'rgba(255, 165, 0, 0)');
                return gradient;
              },
            },
          ];
        }
      });

    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred during testing.';
      dispatch({ type: 'SET_STATUS', payload: `Error: ${errorMessage}` });
      dispatch({ type: 'ADD_LOG', payload: `Error: ${errorMessage}` });
      alert(`Error: ${errorMessage}`);
    } finally {
      dispatch({ type: 'SET_TESTING', payload: false });
    }
  }, [dispatch]);

  const handlePredict = useCallback(async () => {
    dispatch({ type: 'SET_PREDICTING', payload: true });
    dispatch({ type: 'SET_STATUS', payload: 'Predicting future values...' });
    dispatch({ type: 'ADD_LOG', payload: 'Prediction started...' });

    try {
      const model = modelRef.current;
      const preprocessor = preprocessorRef.current;
      if (!model || !preprocessor || !model.bestWeights) {
        throw new Error('Model atau preprocessor belum siap. Latih model terlebih dahulu.');
      }

      const nHari = typeof state.params.n_hari === 'number' ? state.params.n_hari : parseInt(state.params.n_hari || '1', 10);
      const dataTerakhir = preprocessor.df.slice(-model.inputSize).map((d) => d.Terakhir);
      const hasilPrediksiDenorm = model.forwPredict(dataTerakhir, nHari, preprocessor);

      const lastDateStr = preprocessor.df[preprocessor.df.length - 1].Tanggal;
      const futurePredictions: PredictionData[] = [];
      const currDate = new Date(lastDateStr);

      for (let i = 0; i < nHari; i++) {
        currDate.setDate(currDate.getDate() + 1);
        while (currDate.getDay() === 0 || currDate.getDay() === 6) {
          currDate.setDate(currDate.getDate() + 1);
        }
        const yyyy = currDate.getFullYear();
        const mm = String(currDate.getMonth() + 1).padStart(2, "0");
        const dd = String(currDate.getDate()).padStart(2, "0");
        futurePredictions.push({
          date: `${yyyy}-${mm}-${dd}`,
          value: hasilPrediksiDenorm[i],
        });
      }

      dispatch({ type: 'SET_STATUS', payload: 'Prediction complete.' });
      dispatch({ type: 'ADD_LOG', payload: 'Prediction complete.' });

      if (futurePredictions.length > 0) {
        const lastPredictionDate = new Date(futurePredictions[futurePredictions.length - 1].date).getTime();
        dispatch({ type: 'SET_DATE_RANGE', payload: {
          min: state.dateRange.min,
          max: Math.max(state.dateRange.max ?? 0, lastPredictionDate)
        }});
      }

      dispatch({ type: 'UPDATE_PLOT_DATA', payload: (prevDatasets: PlotDataset[]) => {
        const newDatasets = [...prevDatasets];
        const existingPredIndex = newDatasets.findIndex((ds: PlotDataset) => ds.label?.startsWith('Prediction ('));
        if (existingPredIndex > -1) {
          newDatasets.splice(existingPredIndex, 1);
        }

        newDatasets.push({
          label: `Prediction (${state.params.n_hari} days)`,
          data: futurePredictions.map(p => ({ x: new Date(p.date).getTime(), y: p.value })),
          borderColor: 'red',
          borderDash: [5, 5],
          tension: 0.1,
          pointRadius: 0,
          pointHoverRadius: 5,
          fill: true,
          backgroundColor: (context: ScriptableContext<'line'>) => {
            const chart = context.chart;
            const { ctx, chartArea } = chart;
            if (!chartArea) return undefined;
            const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
            gradient.addColorStop(0, 'rgba(255, 0, 0, 0.5)');
            gradient.addColorStop(1, 'rgba(255, 0, 0, 0)');
            return gradient;
          },
        });
        return newDatasets;
      }});

    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred during prediction.';
      dispatch({ type: 'SET_STATUS', payload: `Error: ${errorMessage}` });
      dispatch({ type: 'ADD_LOG', payload: `Error: ${errorMessage}` });
      alert(`Error: ${errorMessage}`);
    } finally {
      dispatch({ type: 'SET_PREDICTING', payload: false });
    }
  }, [state.params.n_hari, state.dateRange, dispatch]);

  const handleResetZoom = useCallback(() => {
    if (chartRef.current) {
      chartRef.current.resetZoom();
    }
  }, [chartRef]);

  const handleTimeRangeZoom = useCallback((range: '1w' | '1m' | '1y' | '5y') => {
    const chart = chartRef.current;
    if (!chart || !state.dateRange.max) {
      return;
    }

    const max = state.dateRange.max;
    let min;

    switch (range) {
      case '1w':
        min = sub(max, { weeks: 1 }).getTime();
        break;
      case '1m':
        min = sub(max, { months: 1 }).getTime();
        break;
      case '1y':
        min = sub(max, { years: 1 }).getTime();
        break;
      case '5y':
      default:
        min = state.dateRange.min;
        break;
    }

    const dataMin = state.dateRange.min ?? 0;
    chart.zoomScale('x', { min: Math.max(min ?? dataMin, dataMin), max }, 'default');
  }, [chartRef, state.dateRange.min, state.dateRange.max]);

  const chartOptions: ChartOptions<'line'> = useMemo(() => {
    const gridColor = state.theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(26, 26, 26, 0.1)';
    const textColor = state.theme === 'dark' ? '#FFFFFF' : '#1A1A1A';
    const noData = !state.plotData.datasets.some(ds => ds.data.length > 0);

    const options: ChartOptions<'line'> = {
      responsive: true,
      maintainAspectRatio: false,
      parsing: {
        xAxisKey: 'x',
        yAxisKey: 'y',
      },
      interaction: {
        mode: 'nearest',
        intersect: false,
      },
      plugins: {
        legend: { position: 'top' as const, labels: { color: textColor } },
        title: { display: true, text: `${state.selectedPair.replace('=X', '')} Forex Prediction using GRU-HHO`, color: textColor, font: { size: 16 } },
        tooltip: {
          mode: 'nearest',
          intersect: false,
        },
        zoom: {
          pan: {
            enabled: true,
            mode: 'x' as const,
          },
          zoom: {
            wheel: {
              enabled: true,
            },
            pinch: {
              enabled: true,
            },
            mode: 'x' as const,
          },
          limits: {
            x: {
              min: state.dateRange.min ?? undefined,
              max: state.dateRange.max ?? undefined,
              minRange: 1000 * 60 * 60 * 24 * 7
            }
          }
        },
      },
      scales: {
        x: {
          type: 'time' as const,
          time: { unit: 'day', tooltipFormat: 'dd/MM/yyyy' },
          title: { display: true, text: 'Date', color: textColor },
          ticks: { color: textColor },
          grid: { color: gridColor },
        },
        y: {
          title: { display: true, text: 'Forex Price', color: textColor },
          ticks: { color: textColor },
          beginAtZero: false,
          grid: { color: gridColor },
          suggestedMin: noData ? 0 : undefined,
          suggestedMax: noData ? 1 : undefined,
        },
      },
    };

    return options;
  }, [state.theme, state.selectedPair, state.dateRange, state.plotData]);


  return (
    <div className="h-screen flex bg-[var(--background)] overflow-x-hidden lg:overflow-hidden">
      {/* Overlay for mobile */}
      {state.isSidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={() => dispatch({ type: 'SET_SIDEBAR_OPEN', payload: false })}
        />
      )}
      <ControlsSidebar
        selectedPair={state.selectedPair}
        handleSelectedPairChange={handlers.handleSelectedPairChange}
        handleLoadData={handleLoadData}
        isLoadingData={state.isLoadingData}
        isTraining={state.isTraining}
        params={state.params}
        handleParamChange={handlers.handleParamChange}
        handleParamKeyDown={handlers.handleParamKeyDown}
        maxBatchSize={state.maxBatchSize}
        paramContainerRef={paramContainerRef}
        forexPairs={FOREX_PAIRS}
        isSidebarOpen={state.isSidebarOpen}
        setIsSidebarOpen={(isOpen) => dispatch({ type: 'SET_SIDEBAR_OPEN', payload: isOpen })}
      />

      <div className="flex-1 flex flex-col min-h-0 min-w-0">
        <Header theme={state.theme} setTheme={(theme: Theme) => dispatch({ type: 'SET_THEME', payload: theme })} onMenuClick={() => dispatch({ type: 'SET_SIDEBAR_OPEN', payload: true })} />

        <main className="flex-1 container mx-auto p-4 flex flex-col lg:flex-row gap-4 min-h-0 overflow-y-auto">
          <MainContent
            handleTrain={handleTrain}
            handleTest={handleTest}
            handlePredict={handlePredict}
            isTraining={state.isTraining}
            isTesting={state.isTesting}
            isPredicting={state.isPredicting}
            isTrained={state.isTrained}
            tableDataLength={state.tableData.length}
            status={state.status}
            chartRef={chartRef}
            plotData={state.plotData}
            chartOptions={chartOptions}
            handleTimeRangeZoom={handleTimeRangeZoom}
            handleResetZoom={handleResetZoom}
          />

          <InfoSidebar
            tableData={state.tableData}
            outputLog={state.outputLog}
            logContainerRef={logContainerRef}
          />
        </main>
      </div>
    </div>
  );
}