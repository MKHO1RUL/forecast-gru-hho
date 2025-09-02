"use client";

import { useState, useRef, ChangeEvent, useEffect, useMemo } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  ChartData,
} from 'chart.js';
import 'chartjs-adapter-date-fns';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

interface DataPoint {
  Tanggal: string;
  Terakhir: number;
}

interface PredictionData {
  date: string;
  value: number;
}

type Theme = 'light' | 'dark';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [tableData, setTableData] = useState<DataPoint[]>([]);
  const [maxBatchSize, setMaxBatchSize] = useState<number>(0);
  const [status, setStatus] = useState<string>('Please load a CSV file to begin.');
  const [outputLog, setOutputLog] = useState<string[]>([]);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [isTesting, setIsTesting] = useState<boolean>(false);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [isTrained, setIsTrained] = useState<boolean>(false);
  const [theme, setTheme] = useState<Theme>('light');
  const [plotData, setPlotData] = useState<ChartData<'line'>>({ datasets: [] });

  const fileInputRef = useRef<HTMLInputElement>(null);

  const [params, setParams] = useState({
    jml_hdnunt: '4',
    batas_MSE: '0.001',
    batch_size: '32',
    maks_epoch: '10',
    elang: '5',
    iterasi: '10',
    n_hari: '7',
  });

  useEffect(() => {
    const root = window.document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [theme]);


  const handleParamChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setParams(prev => ({ ...prev, [name]: value }));
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setFileName(selectedFile.name);
      handleUpload(selectedFile);
    }
  };

  const handleUpload = async (selectedFile: File) => {
    if (!selectedFile) {
      alert('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    setStatus('Uploading and processing data...');
    setOutputLog(prev => [...prev, 'Uploading and processing data...']);

    try {
      const res = await fetch('https://forecast-gru-hho-production.up.railway.app/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Failed to upload file.');
      }

      const data = await res.json();
      setTableData(data.data);
      setMaxBatchSize(data.max_batch_size);
      setStatus('Data loaded successfully.');
      setOutputLog(prev => [...prev, 'Data loaded successfully.']);
      setIsTrained(false);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred during upload.';
      setStatus(`Error: ${errorMessage}`);
      setOutputLog(prev => [...prev, `Error: ${errorMessage}`]);
      alert(`Error: ${errorMessage}`);
    }
  };

  const handleTrain = async () => {
    setIsTraining(true);
    setStatus('Training in progress... Please wait.');
    setOutputLog(prev => [...prev, 'Training started...']);

    try {
      const trainingParams = {
        jml_hdnunt: parseInt(params.jml_hdnunt),
        batas_MSE: parseFloat(params.batas_MSE),
        batch_size: parseInt(params.batch_size),
        maks_epoch: parseInt(params.maks_epoch),
        elang: parseInt(params.elang),
        iterasi: parseInt(params.iterasi),
      };

      const res = await fetch('https://forecast-gru-hho-production.up.railway.app/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainingParams),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Training failed.');
      }

      const data = await res.json();
      setStatus('Training complete!');
      setOutputLog(prev => [...prev, ...data.training_log, `Training complete! Best MSE: ${data.best_mse}`]);
      setIsTrained(true);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred during training.';
      setStatus(`Error: ${errorMessage}`);
      setOutputLog(prev => [...prev, `Error: ${errorMessage}`]);
      alert(`Error: ${errorMessage}`);
    } finally {
      setIsTraining(false);
    }
  };

  const handleTest = async () => {
    setIsTesting(true);
    setStatus('Testing in progress...');
    setOutputLog(prev => [...prev, 'Testing started...']);

    try {
      const res = await fetch('https://forecast-gru-hho-production.up.railway.app/test');
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Testing failed.');
      }

      const data = await res.json();
      setStatus('Testing complete.');
      setOutputLog(prev => [...prev, `Testing complete! MSE: ${data.mse}`]);

      const testingDataPoints = data.testing_data.dates.map((date: string, index: number) => ({
        x: new Date(date).getTime(),
        y: data.testing_data.values[index],
      }));

      const predictionDataPoints = data.prediction_data.dates.map((date: string, index: number) => ({
        x: new Date(date).getTime(),
        y: data.prediction_data.values[index],
      }));

      setPlotData({
        datasets: [
          {
            label: 'Testing Data',
            data: testingDataPoints,
            borderColor: 'green',
            tension: 0.1,
          },
          {
            label: 'Testing Prediction',
            data: predictionDataPoints,
            borderColor: 'orange',
            tension: 0.1,
          },
        ],
      });

    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred during testing.';
      setStatus(`Error: ${errorMessage}`);
      setOutputLog(prev => [...prev, `Error: ${errorMessage}`]);
      alert(`Error: ${errorMessage}`);
    } finally {
      setIsTesting(false);
    }
  };

  const handlePredict = async () => {
    setIsPredicting(true);
    setStatus('Predicting future values...');
    setOutputLog(prev => [...prev, 'Prediction started...']);

    try {
      const res = await fetch(`https://forecast-gru-hho-production.up.railway.app/predict?n_hari=${params.n_hari}`);
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Prediction failed.');
      }

      const data = await res.json();
      setStatus('Prediction complete.');
      setOutputLog(prev => [...prev, 'Prediction complete.']);

      const futurePredictions: PredictionData[] = data.predictions;

      setPlotData((prevData) => {
        const newDatasets = [...prevData.datasets];
        // Remove old prediction if it exists
        const existingPredIndex = newDatasets.findIndex((ds) => ds.label?.startsWith('Prediction ('));
        if (existingPredIndex > -1) {
          newDatasets.splice(existingPredIndex, 1);
        }

        newDatasets.push({
          label: `Prediction (${params.n_hari} days)`,
          data: futurePredictions.map(p => ({ x: new Date(p.date).getTime(), y: p.value })),
          borderColor: 'red',
          borderDash: [5, 5],
          tension: 0.1,
        });
        return { datasets: newDatasets };
      });

    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred during prediction.';
      setStatus(`Error: ${errorMessage}`);
      setOutputLog(prev => [...prev, `Error: ${errorMessage}`]);
      alert(`Error: ${errorMessage}`);
    } finally {
      setIsPredicting(false);
    }
  };

  // Reusable Components
  const Card = ({ title, children, className }: { title: string, children: React.ReactNode, className?: string }) => (
    <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 ${className}`}>
      <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">{title}</h3>
      {children}
    </div>
  );

  const ParamInput = ({ label, name, value, tooltip }: { label: string, name: string, value: string, tooltip: string }) => (
    <div>
      <label htmlFor={name} className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center mb-1">
        {label}
        <span className="ml-1.5 text-gray-400 dark:text-gray-500 cursor-pointer group relative">
          <InfoIcon />
          <span className="absolute bottom-full mb-2 w-48 p-2 bg-gray-700 text-white text-xs rounded-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
            {tooltip}
          </span>
        </span>
      </label>
      <input
        type="text"
        id={name}
        name={name}
        value={value}
        onChange={handleParamChange}
        className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-gray-50 dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
      />
    </div>
  );

  const chartOptions = useMemo(() => {
    const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    const textColor = theme === 'dark' ? '#E5E7EB' : '#1F2937';

    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' as const, labels: { color: textColor } },
        title: { display: true, text: 'GBP/IDR Forex Prediction using GRU-HHO', color: textColor, font: { size: 16 } },
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
          title: { display: true, text: 'GBP/IDR Forex Price', color: textColor },
          ticks: { color: textColor },
          grid: { color: gridColor },
        },
      },
    };
  }, [theme]);


  return (
    <div className="min-h-screen">
      <header className="bg-white dark:bg-gray-800 shadow-md sticky top-0 z-20">
        <div className="container mx-auto px-6 py-3 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">GRU-HHO Forex Prediction</h1>
          <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')} className="p-2 rounded-full text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700">
            {theme === 'light' ? <MoonIcon /> : <SunIcon />}
          </button>
        </div>
      </header>

      <main className="container mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column */}
          <div className="lg:col-span-1 flex flex-col gap-6">
            <Card title="1. Load Data">
              <div className="flex items-center gap-2">
                <input type="text" readOnly value={fileName} className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-gray-100 dark:bg-gray-700" placeholder="No file selected" />
                <button onClick={() => fileInputRef.current?.click()} className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 font-semibold whitespace-nowrap">Browse</button>
                <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept=".csv" />
              </div>
            </Card>

            <Card title="2. Set Parameters">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <ParamInput label="Hidden Units" name="jml_hdnunt" value={params.jml_hdnunt} tooltip="Number of units in the GRU hidden layer. Must be an integer > 1." />
                <ParamInput label="MSE Threshold" name="batas_MSE" value={params.batas_MSE} tooltip="Mean Squared Error target for stopping training. Must be a float between 0 and 1." />
                <ParamInput label="Batch Size" name="batch_size" value={params.batch_size} tooltip={`Number of samples per gradient update. Must be > 0 and <= ${maxBatchSize || 'N/A'}.`} />
                <ParamInput label="Max Epochs" name="maks_epoch" value={params.maks_epoch} tooltip="Maximum number of training epochs. Must be an integer > 0." />
                <ParamInput label="Hawks (HHO)" name="elang" value={params.elang} tooltip="Number of hawks in the Harris Hawks Optimization. Must be an integer > 1." />
                <ParamInput label="Iterations (HHO)" name="iterasi" value={params.iterasi} tooltip="Number of iterations for the HHO algorithm. Must be an integer > 0." />
                <ParamInput label="Prediction Days" name="n_hari" value={params.n_hari} tooltip="Number of future days to predict. Must be an integer > 0." />
              </div>
            </Card>

            <Card title="3. Run Model">
              <div className="flex flex-col space-y-3">
                <ActionButton onClick={handleTrain} disabled={isTraining || !file} isLoading={isTraining} text="Train Model" />
                <ActionButton onClick={handleTest} disabled={isTesting || !isTrained} isLoading={isTesting} text="Test Model" />
                <ActionButton onClick={handlePredict} disabled={isPredicting || !isTrained} isLoading={isPredicting} text="Predict Future" />
              </div>
              <div className="mt-4 h-8 text-sm text-center italic text-gray-600 dark:text-gray-400">{status}</div>
            </Card>

            <Card title="Output Log" className="lg:max-h-96 flex flex-col">
              <div className="flex-1 overflow-y-auto bg-gray-50 dark:bg-gray-900 p-3 rounded-md font-mono text-xs">
                {outputLog.map((line, index) => (
                  <p key={index} className={line.toLowerCase().includes('error') ? 'text-red-500' : line.includes('complete') ? 'text-green-500' : ''}>
                    <span className="text-gray-400 mr-2">{`[${index + 1}]`}</span>{line}
                  </p>
                ))}
              </div>
            </Card>
          </div>

          {/* Right Column */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            <Card title="Data Preview" className="max-h-[450px] flex flex-col">
              <div className="flex-1 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-lg">
                <table className="w-full text-sm text-left">
                  <thead className="text-xs text-gray-700 dark:text-gray-300 uppercase bg-gray-50 dark:bg-gray-700 sticky top-0">
                    <tr>
                      <th scope="col" className="px-4 py-2">No</th>
                      <th scope="col" className="px-4 py-2">Date</th>
                      <th scope="col" className="px-4 py-2 text-right">Price (IDR)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {tableData.map((row, index) => (
                      <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                        <td className="px-4 py-1.5">{index + 1}</td>
                        <td className="px-4 py-1.5">{row.Tanggal}</td>
                        <td className="px-4 py-1.5 text-right">{row.Terakhir.toLocaleString('id-ID')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card title="Prediction Chart" className="h-[500px]">
              <div className="w-full h-full relative">
                <Line data={plotData} options={chartOptions} />
              </div>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}

// Helper Components & Icons
const ActionButton = ({ onClick, disabled, isLoading, text }: { onClick: () => void, disabled: boolean, isLoading: boolean, text: string }) => (
  <button
    onClick={onClick}
    disabled={disabled || isLoading}
    className="w-full flex justify-center items-center px-4 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 dark:disabled:bg-gray-600 font-bold transition-colors"
  >
    {isLoading ? <SpinnerIcon /> : text}
  </button>
);

const SpinnerIcon = () => (
  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);

const InfoIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const SunIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
  </svg>
);

const MoonIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
  </svg>
);