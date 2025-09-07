import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, ChartData, ChartOptions } from 'chart.js';
import { Card, ActionButton, ZoomButtons } from './ui/index';
import { ScrollingDotMatrix } from './scrollingdotmatrix';


interface MainContentProps {
  handleTrain: () => void;
  handleTest: () => void;
  handlePredict: () => void;
  isTraining: boolean;
  isTesting: boolean;
  isPredicting: boolean;
  isTrained: boolean;
  tableDataLength: number;
  status: string;
  chartRef: React.RefObject<ChartJS<'line'> | null>;
  plotData: ChartData<'line'>;
  chartOptions: ChartOptions<'line'>;
  handleTimeRangeZoom: (range: '1w' | '1m' | '1y' | '5y') => void;
  handleResetZoom: () => void;
}

export const MainContent = ({
  handleTrain,
  handleTest,
  handlePredict,
  isTraining,
  isTesting,
  isPredicting,
  isTrained,
  tableDataLength,
  status,
  chartRef,
  plotData,
  chartOptions,
  handleTimeRangeZoom,
  handleResetZoom,
}: MainContentProps) => {
  const statusColor = status.toLowerCase().includes('error')
    ? '#ff3333' // red
    : status.toLowerCase().includes('complete')
    ? '#33ff33' // green
    : status.toLowerCase().includes('loading') || status.toLowerCase().includes('progress')
    ? '#ffaa33' // orange
    : '#ffff33'; // yellow as default

  return (
    <main className="w-full lg:flex-[5] min-w-0 flex flex-col gap-4">
      <Card className="!p-4">
        <div className="flex flex-col sm:flex-row gap-3">
          <ActionButton onClick={handleTrain} disabled={isTraining || tableDataLength === 0} isLoading={isTraining} text="Train Model" />
          <ActionButton onClick={handleTest} disabled={isTraining || isTesting || !isTrained} isLoading={isTesting} text="Test Model" />
          <ActionButton onClick={handlePredict} disabled={isTraining || isPredicting || !isTrained} isLoading={isPredicting} text="Predict Future" />
        </div>
        <div className="mt-3 p-2 bg-black rounded-md font-mono text-center shadow-inner shadow-black/50 flex items-center justify-center h-10 border border-black/20 overflow-hidden">
          <ScrollingDotMatrix text={(status || '—').toUpperCase()} color={statusColor} pixelsPerSecond={75} className="text-sm font-bold" />
        </div>
      </Card>
      <Card className="flex-1 flex flex-col min-h-0">
        <div className="flex flex-col h-full">
          <div className="flex justify-end mb-2">
            <ZoomButtons onZoom={handleTimeRangeZoom} onReset={handleResetZoom} disabled={tableDataLength === 0} />
          </div>
          <div className="flex-1 relative">
            <Line ref={chartRef} data={plotData} options={chartOptions} />
          </div>
        </div>
      </Card>
    </main>
  );
};
