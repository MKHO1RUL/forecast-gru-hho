"use client";

import { useState, useRef, ChangeEvent } from 'react';
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
      const res = await fetch('https://rh3magz7.up.railway.app/upload', {
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

      const res = await fetch('https://rh3magz7.up.railway.app/train', {
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
      const res = await fetch('https://rh3magz7.up.railway.app/test');
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
      const res = await fetch(`https://rh3magz7.up.railway.app/predict?n_hari=${params.n_hari}`);
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

  const ParamInput = ({ label, name, value, tooltip }: { label: string, name: string, value: string, tooltip: string }) => (
    <div className="grid grid-cols-2 items-center gap-2 mb-2">
      <label htmlFor={name} className="text-sm font-medium flex items-center">
        {label}
        <span className="ml-1 text-blue-500 cursor-pointer" title={tooltip}>(?)</span>
      </label>
      <input
        type="text"
        id={name}
        name={name}
        value={value}
        onChange={handleParamChange}
        className="p-1 border rounded-md text-sm"
      />
    </div>
  );

  return (
    <main className="flex flex-col h-screen bg-gray-100 text-gray-900">
      <div className="flex flex-1 overflow-hidden">
        {/* Left Pane: Data Display */}
        <div className="w-1/4 p-4 flex flex-col border-r bg-white">
          <h2 className="text-lg font-bold text-center mb-2">Input Data</h2>
          <div className="flex-1 overflow-y-auto border rounded-lg">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-gray-700 uppercase bg-gray-50 sticky top-0">
                <tr>
                  <th scope="col" className="px-2 py-2">No</th>
                  <th scope="col" className="px-2 py-2">Date</th>
                  <th scope="col" className="px-2 py-2">Price</th>
                </tr>
              </thead>
              <tbody>
                {tableData.map((row, index) => (
                  <tr key={index} className="bg-white border-b">
                    <td className="px-2 py-1">{index + 1}</td>
                    <td className="px-2 py-1">{row.Tanggal}</td>
                    <td className="px-2 py-1">{row.Terakhir.toLocaleString('id-ID')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Middle Pane: Controls */}
        <div className="w-1/2 p-4 flex flex-col items-center bg-white overflow-y-auto">
          <h1 className="text-2xl font-bold text-center my-4">GRU-HHO Forex Prediction</h1>
          <div className="w-full max-w-md mb-4">
            <div className="flex items-center">
              <label htmlFor="file-upload" className="text-sm font-medium mr-2">CSV File:</label>
              <input
                type="text"
                readOnly
                value={fileName}
                className="p-1 border rounded-md flex-grow text-sm"
                placeholder="No file selected"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="ml-2 px-3 py-1 bg-blue-500 text-white rounded-md hover:bg-blue-600 text-sm font-bold"
              >
                Browse
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                accept=".csv"
              />
            </div>
          </div>

          <div className="border rounded-lg p-4 w-full max-w-md mb-4">
            <h3 className="font-bold mb-2 text-center">Parameters</h3>
            <div className="grid grid-cols-2 gap-x-4">
              <div>
                <ParamInput label="Hidden Units:" name="jml_hdnunt" value={params.jml_hdnunt} tooltip="Must be an integer > 1" />
                <ParamInput label="MSE Threshold:" name="batas_MSE" value={params.batas_MSE} tooltip="Must be a float between 0 and 1" />
                <ParamInput label="Batch Size:" name="batch_size" value={params.batch_size} tooltip={`Must be > 0 and <= ${maxBatchSize || 'N/A'}`} />
                <ParamInput label="Epochs:" name="maks_epoch" value={params.maks_epoch} tooltip="Must be an integer > 0" />
              </div>
              <div>
                <ParamInput label="Hawks:" name="elang" value={params.elang} tooltip="Must be an integer > 1" />
                <ParamInput label="Iterations:" name="iterasi" value={params.iterasi} tooltip="Must be an integer > 0" />
                <ParamInput label="Prediction Days:" name="n_hari" value={params.n_hari} tooltip="Must be an integer > 0" />
              </div>
            </div>
          </div>

          <div className="flex space-x-2 mb-4">
            <button onClick={handleTrain} disabled={isTraining || !file} className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:bg-gray-400 font-bold">Train</button>
            <button onClick={handleTest} disabled={isTesting || !isTrained} className="px-4 py-2 bg-yellow-500 text-white rounded-md hover:bg-yellow-600 disabled:bg-gray-400 font-bold">Test</button>
            <button onClick={handlePredict} disabled={isPredicting || !isTrained} className="px-4 py-2 bg-purple-500 text-white rounded-md hover:bg-purple-600 disabled:bg-gray-400 font-bold">Predict</button>
            <button disabled className="px-4 py-2 bg-gray-500 text-white rounded-md disabled:bg-gray-400 font-bold">Export CSV</button>
          </div>
          <div className="h-6 text-sm italic text-gray-600">{status}</div>
        </div>

        {/* Right Pane: Output Log */}
        <div className="w-1/4 p-4 flex flex-col border-l bg-white">
          <h2 className="text-lg font-bold text-center mb-2">Output Log</h2>
          <div className="flex-1 overflow-y-auto border rounded-lg p-2 bg-gray-50 text-sm">
            {outputLog.map((line, index) => (
              <p key={index} className={line.includes('complete') ? 'text-green-600' : ''}>{line}</p>
            ))}
          </div>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden p-4">
        {/* Bottom Pane: Plot */}
        <div className="w-full p-4 flex flex-col bg-white">
          <h2 className="text-lg font-bold text-center mb-2">Plot</h2>
          <div className="flex-1 relative">
            <Line
              data={plotData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: { position: 'top' },
                  title: { display: true, text: 'GBP/IDR Forex Prediction using GRU-HHO' },
                },
                scales: {
                  x: {
                    type: 'time',
                    time: {
                      unit: 'day',
                      tooltipFormat: 'dd/MM/yyyy',
                    },
                    title: { display: true, text: 'Date' },
                  },
                  y: {
                    title: { display: true, text: 'GBP/IDR Forex Price' },
                  },
                },
              }}
            />
          </div>
        </div>
      </div>
    </main>
  );
}