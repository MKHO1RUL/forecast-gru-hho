import React from 'react';
import { SunIcon, MoonIcon } from './ui/index';
import { Theme } from '../types';

export const Header = ({ theme, setTheme }: { theme: Theme, setTheme: (theme: Theme) => void }) => {
  return (
    <header className="bg-[var(--card-background)] shadow-md sticky top-0 z-20 flex-shrink-0">
      <div className="container mx-auto px-6 py-2 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-[var(--foreground)]">GRU-HHO Forex Prediction</h1>
        <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')} className="p-2 rounded-full text-[var(--text-muted)] hover:bg-[var(--secondary)]">
          {theme === 'light' ? <MoonIcon /> : <SunIcon />}
        </button>
      </div>
    </header>
  );
};
