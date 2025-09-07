import React, { ChangeEvent } from 'react';

// --- Icons ---
export const SpinnerIcon = () => (
  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);

export const InfoIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

export const SunIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
  </svg>
);

export const MoonIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
  </svg>
);

// --- UI Components ---
export const Card = ({ title, children, className }: { title?: string, children: React.ReactNode, className?: string }) => (
  <div className={`bg-[var(--card-background)] rounded-xl shadow-lg p-4 ${className}`}>
    {title && <h3 className="text-lg font-semibold mb-2 text-[var(--foreground)]">{title}</h3>}
    {children}
  </div>
);

export const ParamInput = ({ label, name, value, tooltip, type = 'number', step, onChange, onKeyDown, placeholder }: {
  label: string;
  name: string;
  value: string;
  tooltip: string;
  type?: string;
  step?: string;
  placeholder?: string;
  onChange: (e: ChangeEvent<HTMLInputElement>) => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
}) => (
  <div>
    <label htmlFor={name} className="text-xs font-medium text-[var(--foreground)] flex items-center mb-1">
      {label}
      <span className="ml-1.5 text-[var(--text-muted)] cursor-pointer group relative">
        <InfoIcon />
        <span className="absolute bottom-full mb-2 w-48 p-2 bg-gray-900 text-white dark:bg-gray-700 text-xs rounded-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
          {tooltip}
        </span>
      </span>
    </label>
    <input
      type={type}
      id={name}
      name={name}
      value={value}
      onChange={onChange}
      step={step}
      onKeyDown={onKeyDown}
      placeholder={placeholder}
      className="w-full py-1.5 px-2 border border-[var(--input-border)] rounded-md text-xs bg-[var(--input-background)] text-[var(--foreground)] focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
    />
  </div>
);

export const ActionButton = ({ onClick, disabled, isLoading, text }: { onClick: () => void, disabled: boolean, isLoading: boolean, text: string }) => (
  <button
    onClick={onClick}
    disabled={disabled || isLoading}
    className="w-full flex justify-center items-center px-4 py-2.5 bg-[var(--primary)] text-white rounded-lg hover:bg-[var(--primary-hover)] disabled:bg-[var(--secondary)] font-bold transition-colors"
  >
    {isLoading ? <SpinnerIcon /> : text}
  </button>
);

export const ZoomButtons = ({ onZoom, onReset, disabled }: {
  onZoom: (range: '1w' | '1m' | '1y' | '5y') => void;
  onReset: () => void;
  disabled: boolean;
}) => {
  const ranges: { label: string; value: '1w' | '1m' | '1y' | '5y' }[] = [
    { label: '1W', value: '1w' },
    { label: '1M', value: '1m' },
    { label: '1Y', value: '1y' },
    { label: '5Y', value: '5y' },
  ];

  return (
    <div className="flex items-center gap-1">
      {ranges.map(range => (
        <button key={range.value} onClick={() => onZoom(range.value)} disabled={disabled} className="px-3 py-1 text-xs font-semibold rounded-md transition-colors bg-[var(--secondary)] text-[var(--foreground)] hover:bg-[var(--secondary-hover)] disabled:opacity-50 disabled:cursor-not-allowed">
          {range.label}
        </button>
      ))}
      <button onClick={onReset} disabled={disabled} className="px-3 py-1 text-xs font-semibold rounded-md transition-colors bg-[var(--secondary)] text-[var(--foreground)] hover:bg-[var(--secondary-hover)] disabled:opacity-50 disabled:cursor-not-allowed">
        Reset Zoom
      </button>
    </div>
  );
};

