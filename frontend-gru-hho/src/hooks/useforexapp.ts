import { useReducer, useCallback, useRef, useEffect } from 'react';
import { DataPoint, Theme } from '../types';
import { Chart as ChartJS } from 'chart.js';

export type PlotDataset = {
  label: string;
  data: (number | { x: number; y: number } | null)[];
  // Izinkan properti lain yang mungkin digunakan oleh Chart.js
  [key: string]: unknown;
};

type AppState = {
  isLoadingData: boolean;
  isTraining: boolean;
  isTesting: boolean;
  isPredicting: boolean;
  isTrained: boolean;
  status: string;
  outputLog: string[];
  tableData: DataPoint[];
  maxBatchSize: number;
  plotData: { datasets: PlotDataset[] };
  dateRange: { min: number | null; max: number | null };
  theme: Theme;
  selectedPair: string;
  params: {
    jml_hdnunt: string;
    batas_MSE: string;
    batch_size: string;
    maks_epoch: string;
    elang: string;
    iterasi: string;
    n_hari: string;
  };
};

type Action =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_TRAINING'; payload: boolean }
  | { type: 'SET_TESTING'; payload: boolean }
  | { type: 'SET_PREDICTING'; payload: boolean }
  | { type: 'SET_IS_TRAINED'; payload: boolean }
  | { type: 'SET_STATUS'; payload: string }
  | { type: 'ADD_LOG'; payload: string }
  | { type: 'RESET_LOG' }
  | { type: 'SET_TABLE_DATA'; payload: { data: DataPoint[]; maxBatchSize: number } }
  | { type: 'SET_PLOT_DATA'; payload: PlotDataset[] }
  | { type: 'UPDATE_PLOT_DATA'; payload: (prevDatasets: PlotDataset[]) => PlotDataset[] }
  | { type: 'SET_DATE_RANGE'; payload: { min: number | null; max: number | null } }
  | { type: 'SET_THEME'; payload: Theme }
  | { type: 'SET_SELECTED_PAIR'; payload: string }
  | { type: 'SET_PARAMS'; payload: { name: string; value: string } };

const initialState: AppState = {
  isLoadingData: false,
  isTraining: false,
  isTesting: false,
  isPredicting: false,
  isTrained: false,
  status: 'Please select a forex pair and load the data.',
  outputLog: [],
  tableData: [],
  maxBatchSize: 0,
  plotData: { datasets: [] },
  dateRange: { min: null, max: null },
  theme: 'light',
  selectedPair: 'EURUSD=X',
  params: {
    jml_hdnunt: '',
    batas_MSE: '',
    batch_size: '',
    maks_epoch: '',
    elang: '',
    iterasi: '',
    n_hari: '',
  },
};

function appReducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoadingData: action.payload };
    case 'SET_TRAINING':
      return { ...state, isTraining: action.payload };
    case 'SET_TESTING':
      return { ...state, isTesting: action.payload };
    case 'SET_PREDICTING':
      return { ...state, isPredicting: action.payload };
    case 'SET_IS_TRAINED':
      return { ...state, isTrained: action.payload };
    case 'SET_STATUS':
      return { ...state, status: action.payload };
    case 'ADD_LOG':
      return { ...state, outputLog: [...state.outputLog, action.payload] };
    case 'RESET_LOG':
      return { ...state, outputLog: [] };
    case 'SET_TABLE_DATA':
      return { ...state, tableData: action.payload.data, maxBatchSize: action.payload.maxBatchSize, plotData: { datasets: [] }, isTrained: false };
    case 'SET_PLOT_DATA':
      return { ...state, plotData: { datasets: action.payload } };
    case 'UPDATE_PLOT_DATA':
      return { ...state, plotData: { datasets: action.payload(state.plotData.datasets) } };
    case 'SET_DATE_RANGE':
      return { ...state, dateRange: action.payload };
    case 'SET_THEME':
      return { ...state, theme: action.payload };
    case 'SET_SELECTED_PAIR':
      return { ...state, selectedPair: action.payload };
    case 'SET_PARAMS':
      return { ...state, params: { ...state.params, [action.payload.name]: action.payload.value } };
    default:
      return state;
  }
}

export const useForexApp = () => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // --- Refs ---
  const chartRef = useRef<ChartJS<'line'>>(null);
  const paramContainerRef = useRef<HTMLDivElement>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // --- Effects ---
  // Efek untuk auto-scroll pada output log
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [state.outputLog]);

  // Efek untuk mengatur tema awal berdasarkan preferensi OS dan mendengarkan perubahan
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    dispatch({ type: 'SET_THEME', payload: mediaQuery.matches ? 'dark' : 'light' });
    const handler = (e: MediaQueryListEvent) => dispatch({ type: 'SET_THEME', payload: e.matches ? 'dark' : 'light' });
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  // Efek untuk menerapkan kelas tema ke elemen root
  useEffect(() => {
    const root = window.document.documentElement;
    if (state.theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [state.theme]);

  // --- Callbacks ---
  const handleParamChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    dispatch({ type: 'SET_PARAMS', payload: { name, value } });
  }, []);

  const handleSelectedPairChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    dispatch({ type: 'SET_SELECTED_PAIR', payload: e.target.value });
  }, []);

  const handleParamKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    e.preventDefault();

    const inputs = Array.from(
      paramContainerRef.current?.querySelectorAll('input[type="number"]') ?? []
    ) as HTMLInputElement[];

    const currentIndex = inputs.findIndex(input => input === e.currentTarget);
    if (currentIndex === -1) return;

    let nextIndex;
    if (e.key === 'ArrowDown') {
      nextIndex = (currentIndex + 1) % inputs.length;
    } else { // ArrowUp
      nextIndex = (currentIndex - 1 + inputs.length) % inputs.length;
    }

    const nextInput = inputs[nextIndex];
    if (nextInput) {
      nextInput.focus();
    }
  }, []);

  return {
    state,
    dispatch,
    refs: {
      chartRef,
      paramContainerRef,
      logContainerRef,
    },
    handlers: {
      handleParamChange,
      handleSelectedPairChange,
      handleParamKeyDown,
    },
  };
};
