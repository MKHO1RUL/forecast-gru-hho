import React from 'react';
import { Card } from './ui/index';
import { DotMatrixText } from './dotmatrixtext';
import { DataPoint } from '../types';

interface InfoSidebarProps {
  tableData: DataPoint[];
  outputLog: string[];
  logContainerRef: React.RefObject<HTMLDivElement | null>;
}

export const InfoSidebar = ({ tableData, outputLog, logContainerRef }: InfoSidebarProps) => {
  return (
    <aside className="w-full lg:flex-[3] min-w-0 flex flex-col gap-4">
      <Card title="Data Preview" className="flex flex-col min-h-0 h-80 lg:flex-1 lg:h-auto">
        <div className="flex-1 overflow-auto border border-[var(--secondary)] rounded-lg min-h-0">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-[var(--foreground)] uppercase bg-[var(--secondary)] sticky top-0">
              <tr>
                <th scope="col" className="px-4 py-2">No</th>
                <th scope="col" className="px-4 py-2">Date</th>
                <th scope="col" className="px-4 py-2 text-right">Price</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--secondary)]">
              {tableData.map((row, index) => (
                <tr key={`${row.Tanggal}-${index}`} className="hover:bg-[var(--secondary)]">
                  <td className="px-4 py-1.5">{index + 1}</td>
                  <td className="px-4 py-1.5">{row.Tanggal}</td>
                  <td className="px-4 py-1.5 text-right">{row.Terakhir.toLocaleString('id-ID')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <Card title="Output Log" className="flex flex-col min-h-0 h-80 lg:flex-1 lg:h-auto">
        <div ref={logContainerRef} className="flex-1 overflow-y-auto bg-gray-900 dark:bg-black/70 p-3 rounded-md font-mono text-xs min-h-0 shadow-inner shadow-black/50 border border-black/20">
          {outputLog.map((line, index) => (
            <p key={index}>
              <span className="text-[var(--text-muted)] mr-2">{`[${index + 1}]`}</span>
              <DotMatrixText
                text={line}
                color={
                  line.toLowerCase().includes('error')
                    ? '#ff3333'
                    : line.includes('complete')
                    ? '#33ff33'
                    : '#FFFFFF'
                }
                className="text-xs"
              />
            </p>
          ))}
        </div>
      </Card>
    </aside>
  );
};
