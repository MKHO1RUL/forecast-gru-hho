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
  ChartData,
  ChartArea,
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
import { useForexApp } from "../hooks/useforexapp";
import { DataPoint, PredictionData, Theme } from "../types";

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
  const { refs: { chartRef, paramContainerRef, logContainerRef }, handlers: { handleParamChange, handleSelectedPairChange, handleParamKeyDown } } = { refs, handlers };

  // --- Effects ---
  // Register the zoom plugin only on the client side to avoid SSR issues.
  useEffect(() => {
    let zoomPlugin: Plugin | undefined;
    const registerPlugin = async () => {
      const module = await import('chartjs-plugin-zoom');
      zoomPlugin = module.default;
      ChartJS.register(zoomPlugin);
    };
    registerPlugin();
    return () => {
      if (zoomPlugin) {
        ChartJS.unregister(zoomPlugin);
      }
    };
  }, []);

  // Memoize training parameters to avoid unnecessary effect runs
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

  // Ref untuk melacak nilai parameter sebelumnya agar efek reset berjalan dengan benar.
  const prevTrainingParamsRef = useRef(trainingParamsMemo);

  // Reset model status if training parameters change
  useEffect(() => {
    // Efek ini seharusnya hanya mereset status jika parameter benar-benar berubah.
    // Kita membandingkan parameter saat ini dengan nilai dari render sebelumnya.
    if (JSON.stringify(prevTrainingParamsRef.current) !== JSON.stringify(trainingParamsMemo)) {
      if (state.isTrained) {
        dispatch({ type: 'SET_IS_TRAINED', payload: false });
        dispatch({ type: 'SET_STATUS', payload: 'Parameters changed. Please retrain the model.' });
        dispatch({ type: 'ADD_LOG', payload: 'Parameters changed. Model needs to be retrained.' });
      }
    }

    // Selalu perbarui ref dengan parameter saat ini untuk perbandingan di render berikutnya.
    prevTrainingParamsRef.current = trainingParamsMemo;
  }, [trainingParamsMemo, state.isTrained, dispatch]); // useRef is not needed in dependency array

  // --- Callbacks ---
  const handleLoadData = useCallback(async () => {
    if (!state.selectedPair) {
      alert('Please select a forex pair.');
      return;
    }

    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_STATUS', payload: `Loading data for ${state.selectedPair}...` });
    dispatch({ type: 'ADD_LOG', payload: `Fetching 5-year historical data for ${state.selectedPair}...` });
    dispatch({ type: 'SET_TABLE_DATA', payload: { data: [], maxBatchSize: 0 }});
    dispatch({ type: 'SET_PLOT_DATA', payload: [] });
    dispatch({ type: 'SET_IS_TRAINED', payload: false });

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/get-data?pair=${state.selectedPair}`);

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || `Failed to load data for ${state.selectedPair}.`);
      }

      const data = await res.json();
      dispatch({ type: 'SET_TABLE_DATA', payload: { data: data.data, maxBatchSize: data.max_batch_size } });
      dispatch({ type: 'SET_STATUS', payload: 'Data loaded successfully.' });
      dispatch({ type: 'ADD_LOG', payload: 'Data loaded successfully. Ready for training.' });
      
      const initialDataPoints = data.data.map((point: DataPoint) => ({
        x: new Date(point.Tanggal).getTime(),
        y: point.Terakhir,
      }));

      if (initialDataPoints.length > 0) {
        const xValues = initialDataPoints.map((p: { x: number; y: number }) => p.x);
        const minDate = Math.min(...xValues);
        const maxDate = Math.max(...xValues);
        dispatch({ type: 'SET_DATE_RANGE', payload: { min: minDate, max: maxDate } });
      } else {
        dispatch({ type: 'SET_DATE_RANGE', payload: { min: null, max: null } });
      }

      // PERBAIKAN: Pisahkan data menjadi set training dan testing untuk plot awal
      const allDataPoints = data.data.map((point: DataPoint) => ({
        x: new Date(point.Tanggal).getTime(),
        y: point.Terakhir,
      }));

      // Backend menggunakan 0.7 untuk split, jadi kita sinkronkan di sini
      const trainSize = Math.floor(allDataPoints.length * 0.7);
      const trainingPoints = allDataPoints.slice(0, trainSize);
      // Ambil satu titik dari data training agar garisnya menyambung
      const testingPoints = allDataPoints.slice(trainSize - 1);

      dispatch({ type: 'SET_PLOT_DATA', payload: [
          {
            label: 'Training Data',
            data: trainingPoints,
            borderColor: 'blue',
            tension: 0.1,
            pointRadius: 0, // No circle on the point
            pointHoverRadius: 5, // Circle appears on hover
            fill: true,
            backgroundColor: (context: ScriptableContext<'line'>) => {
              const chart = context.chart;
              const { ctx, chartArea } = chart;
              if (!chartArea) return undefined; // Gradient requires chart area
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
            pointRadius: 0, // No circle on the point
            pointHoverRadius: 5, // Circle appears on hover
            fill: true,
            backgroundColor: (context: ScriptableContext<'line'>) => {
              const chart = context.chart;
              const { ctx, chartArea } = chart;
              if (!chartArea) return undefined; // Gradient requires chart area
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
    // --- Frontend Validation ---
    const requiredParams: (keyof Omit<typeof state.params, 'n_hari'>)[] = [
      'jml_hdnunt', 'batas_MSE', 'batch_size', 'maks_epoch', 'elang', 'iterasi'
    ];

    const missingParams = requiredParams.filter(p => !state.params[p] || state.params[p].trim() === '');

    if (missingParams.length > 0) {
      const message = `Please fill all parameter fields before training.`;
      dispatch({ type: 'SET_STATUS', payload: message });
      dispatch({ type: 'ADD_LOG', payload: `Error: ${message}` });
      alert(message);
      return;
    }
    // --- End of Validation ---

    dispatch({ type: 'SET_TRAINING', payload: true });
    dispatch({ type: 'SET_STATUS', payload: 'Training in progress... Please wait.' });
    dispatch({ type: 'ADD_LOG', payload: 'Training started...' });

    // PERBAIKAN: Hapus plot prediksi masa depan yang lama saat training ulang.
    dispatch({
      type: 'UPDATE_PLOT_DATA', payload: (datasets: any[]) => datasets.filter(
        (ds) => !ds.label?.startsWith('Prediction (')
      )
    });


    try {
      const trainingParams = {
        jml_hdnunt: parseInt(state.params.jml_hdnunt),
        batas_MSE: parseFloat(state.params.batas_MSE),
        batch_size: parseInt(state.params.batch_size),
        maks_epoch: parseInt(state.params.maks_epoch),
        elang: parseInt(state.params.elang),
        iterasi: parseInt(state.params.iterasi),
      };

      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainingParams),
      });

      if (!res.ok || !res.body) {
        const errorData = await res.json();
        // PERBAIKAN: Menangani pesan error dari backend yang lebih detail
        if (errorData.detail && Array.isArray(errorData.detail)) {
          const errorMessages = errorData.detail.map(
            (err: any) => `- ${err.loc.length > 1 ? err.loc[1] : 'Error'}: ${err.msg}`
          ).join('\n');
          throw new Error(`Invalid Parameters:\n${errorMessages}`);
        }
        throw new Error(errorData.detail || 'Training failed to start.');
      }

      // PERBAIKAN: Memproses response sebagai stream untuk update log secara real-time
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        const chunk = decoder.decode(value, { stream: !done });
        
        const lines = chunk.split('\n').filter(line => line.trim() !== '');

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            if (data.type === 'log') {
              dispatch({ type: 'ADD_LOG', payload: data.message });
            } else if (data.type === 'complete') {
              dispatch({ type: 'SET_STATUS', payload: 'Training complete!' });
              dispatch({ type: 'ADD_LOG', payload: `Training complete! Best MSE: ${data.best_mse}` });
              dispatch({ type: 'SET_IS_TRAINED', payload: true });
            } else if (data.type === 'error') {
              throw new Error(data.message);
            }
          } catch (e) {
            console.error("Failed to parse stream line:", line, e);
          }
        }
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred during training.';
      dispatch({ type: 'SET_STATUS', payload: `Error: ${errorMessage}` });
      dispatch({ type: 'ADD_LOG', payload: `Error: ${errorMessage}` });
      alert(`Error: ${errorMessage}`);
    } finally {
      dispatch({ type: 'SET_TRAINING', payload: false });
    }
  }, [state.params, dispatch]);

  const handleTest = useCallback(async () => {
    dispatch({ type: 'SET_TESTING', payload: true });
    dispatch({ type: 'SET_STATUS', payload: 'Testing in progress...' });
    dispatch({ type: 'ADD_LOG', payload: 'Testing started...' });

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/test`);
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Testing failed.');
      }

      const data = await res.json();
      dispatch({ type: 'SET_STATUS', payload: 'Testing complete.' });
      dispatch({ type: 'ADD_LOG', payload: `Testing complete! MSE: ${data.mse}` });

      dispatch({
        type: 'UPDATE_PLOT_DATA', payload: (prevDatasets: any[]) => {
        // PERBAIKAN: Hapus prediksi lama jika ada, dan tambahkan yang baru.
        // Jangan hapus data training/testing yang sudah ada.
        const baseDatasets = prevDatasets.filter(
          (ds) => ds.label !== 'Testing Prediction'
        );
        return [
            ...baseDatasets,
            {
              label: 'Testing Prediction',
              data: data.prediction_data.dates.map((date: string, index: number) => ({ x: new Date(date).getTime(), y: data.prediction_data.values[index] })),
              borderColor: 'orange',
              tension: 0.1,
              pointRadius: 0, // No circle on the point
              pointHoverRadius: 5, // Circle appears on hover
              fill: true,
              backgroundColor: (context: ScriptableContext<'line'>) => {
                const chart = context.chart;
                const { ctx, chartArea } = chart;
                if (!chartArea) return undefined; // Gradient requires chart area
                const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                gradient.addColorStop(0, 'rgba(255, 165, 0, 0.5)');
                gradient.addColorStop(1, 'rgba(255, 165, 0, 0)');
                return gradient;
              },
            },
          ];
      }});

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
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict?n_hari=${state.params.n_hari}`);
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Prediction failed.');
      }

      const data = await res.json();
      dispatch({ type: 'SET_STATUS', payload: 'Prediction complete.' });
      dispatch({ type: 'ADD_LOG', payload: 'Prediction complete.' });

      const futurePredictions: PredictionData[] = data.predictions;

      // PERBAIKAN: Perbarui rentang tanggal maksimum untuk menyertakan data prediksi.
      // Ini akan memperluas batas zoom-out.
      if (futurePredictions.length > 0) {
        const lastPredictionDate = new Date(futurePredictions[futurePredictions.length - 1].date).getTime();
        dispatch({ type: 'SET_DATE_RANGE', payload: {
          min: state.dateRange.min,
          max: Math.max(state.dateRange.max ?? 0, lastPredictionDate)
        }});
      }

      dispatch({ type: 'UPDATE_PLOT_DATA', payload: (prevDatasets: any[]) => {
        const newDatasets = [...prevDatasets];
        // Remove old prediction if it exists
        const existingPredIndex = newDatasets.findIndex((ds) => ds.label?.startsWith('Prediction ('));
        if (existingPredIndex > -1) {
          newDatasets.splice(existingPredIndex, 1);
        }

        newDatasets.push({
          label: `Prediction (${state.params.n_hari} days)`,
          data: futurePredictions.map(p => ({ x: new Date(p.date).getTime(), y: p.value })),
          borderColor: 'red',
          borderDash: [5, 5],
          tension: 0.1,
          pointRadius: 0, // No circle on the point
          pointHoverRadius: 5, // Circle appears on hover
          fill: true,
          backgroundColor: (context: ScriptableContext<'line'>) => {
            const chart = context.chart;
            const { ctx, chartArea } = chart;
            if (!chartArea) return undefined; // Gradient requires chart area
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
  }, [state.params.n_hari, dispatch]);

  const handleResetZoom = useCallback(() => {
    if (chartRef.current) {
      chartRef.current.resetZoom();
    }
  }, []);

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

    // Pastikan zoom tidak melebihi data yang ada
    const dataMin = state.dateRange.min ?? 0;
    chart.zoomScale('x', { min: Math.max(min ?? dataMin, dataMin), max }, 'default');
  }, [state.dateRange]);

  // --- Chart Options ---
  const chartOptions: ChartOptions<'line'> = useMemo(() => {
    const gridColor = state.theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(26, 26, 26, 0.1)';
    const textColor = state.theme === 'dark' ? '#FFFFFF' : '#1A1A1A';

    const options: ChartOptions<'line'> = {
      responsive: true,
      maintainAspectRatio: false,
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
              // Batasi zoom-in minimal hingga 1 minggu
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
          grid: { color: gridColor },
        },
      },
    };

    return options;
  }, [state.theme, state.selectedPair, state.dateRange]);


  return (
    <div className="h-screen flex flex-col lg:flex-row bg-[var(--background)] overflow-hidden">
      <ControlsSidebar
        selectedPair={state.selectedPair}
        handleSelectedPairChange={handleSelectedPairChange}
        handleLoadData={handleLoadData}
        isLoadingData={state.isLoadingData}
        isTraining={state.isTraining}
        params={state.params}
        handleParamChange={handleParamChange}
        handleParamKeyDown={handleParamKeyDown}
        maxBatchSize={state.maxBatchSize}
        paramContainerRef={paramContainerRef}
        forexPairs={FOREX_PAIRS}
      />

      <div className="flex-1 flex flex-col min-h-0 min-w-0">
        <Header theme={state.theme} setTheme={(theme: Theme) => dispatch({ type: 'SET_THEME', payload: theme })} />

        <div className="flex-1 container mx-auto p-4 flex flex-col lg:flex-row gap-4 min-h-0">
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
        </div>
      </div>
    </div>
  );
}