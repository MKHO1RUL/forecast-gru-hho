import React, { ChangeEvent } from 'react';
import { ParamInput, SpinnerIcon } from './ui/index';

// Define types for props
interface ParamsState {
  jml_hdnunt: string;
  batas_MSE: string;
  batch_size: string;
  maks_epoch: string;
  elang: string;
  iterasi: string;
  n_hari: string;
}

interface ControlsSidebarProps {
  selectedPair: string;
  handleSelectedPairChange: (e: ChangeEvent<HTMLSelectElement>) => void;
  handleLoadData: () => void;
  isLoadingData: boolean;
  isTraining: boolean;
  params: ParamsState;
  handleParamChange: (e: ChangeEvent<HTMLInputElement>) => void;
  handleParamKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  maxBatchSize: number;
  paramContainerRef: React.RefObject<HTMLDivElement | null>;
  forexPairs: string[];
}

export const ControlsSidebar = ({
  selectedPair,
  handleSelectedPairChange,
  handleLoadData,
  isLoadingData,
  isTraining,
  params,
  handleParamChange,
  handleParamKeyDown,
  maxBatchSize,
  paramContainerRef,
  forexPairs,
}: ControlsSidebarProps) => {
  return (
    <aside className="w-full lg:w-72 flex-shrink-0 flex flex-col gap-4 bg-[var(--card-background)] shadow-lg p-4 overflow-y-auto overflow-x-hidden">
      <h3 className="text-xl font-bold text-[var(--foreground)]">Controls</h3>
      
      {/* Data Selection */}
      <div className="flex flex-col gap-2">
        <label className="text-sm font-medium text-[var(--foreground)]">Forex Pair</label>
        <div className="grid grid-cols-[1fr_auto] items-center gap-2">
            <select
              name="selectedPair"
              value={selectedPair}
              onChange={handleSelectedPairChange}
              className="w-full p-2.5 border border-[var(--input-border)] rounded-md text-sm bg-[var(--input-background)] text-[var(--foreground)] focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            >
              {forexPairs.map(pair => (
                <option key={pair} value={pair}>{pair.replace('=X', '')}</option>
              ))}
            </select>
            <button onClick={handleLoadData} disabled={isLoadingData || isTraining} className="px-4 py-2 bg-[var(--primary)] text-white rounded-md hover:bg-[var(--primary-hover)] font-semibold whitespace-nowrap disabled:bg-[var(--secondary)]">
              {isLoadingData ? <SpinnerIcon /> : 'Load'}
            </button>
        </div>
      </div>

      <hr className="border-t border-[var(--input-border)] my-2" />

      {/* Parameters */}
      <div ref={paramContainerRef} className="flex flex-col gap-2">
        <h4 className="text-lg font-semibold text-[var(--foreground)]">Parameters</h4>
        <ParamInput onKeyDown={handleParamKeyDown} onChange={handleParamChange} label="Hidden Units" name="jml_hdnunt" value={params.jml_hdnunt} placeholder="4" tooltip="Number of units in the GRU hidden layer. Must be an integer > 1." />
        <ParamInput onKeyDown={handleParamKeyDown} onChange={handleParamChange} label="MSE Threshold" name="batas_MSE" value={params.batas_MSE} placeholder="0.001" tooltip="Mean Squared Error target for stopping training. Must be a float between 0 and 1." step="0.001" />
        <ParamInput onKeyDown={handleParamKeyDown} onChange={handleParamChange} label="Batch Size" name="batch_size" value={params.batch_size} placeholder="8" tooltip={`Number of samples per gradient update. Must be > 0 and <= ${maxBatchSize || 'N/A'}.`} />
        <ParamInput onKeyDown={handleParamKeyDown} onChange={handleParamChange} label="Max Epochs" name="maks_epoch" value={params.maks_epoch} placeholder="10" tooltip="Maximum number of training epochs. Must be an integer > 0." />
        <ParamInput onKeyDown={handleParamKeyDown} onChange={handleParamChange} label="Hawks (HHO)" name="elang" value={params.elang} placeholder="5" tooltip="Number of hawks in the Harris Hawks Optimization. Must be an integer > 1." />
        <ParamInput onKeyDown={handleParamKeyDown} onChange={handleParamChange} label="Iterations (HHO)" name="iterasi" value={params.iterasi} placeholder="10" tooltip="Number of iterations for the HHO algorithm. Must be an integer > 0." />
        <ParamInput onKeyDown={handleParamKeyDown} onChange={handleParamChange} label="Prediction Days" name="n_hari" value={params.n_hari} placeholder="7" tooltip="Number of future days to predict. Must be an integer > 0." />
      </div>
    </aside>
  );
};
