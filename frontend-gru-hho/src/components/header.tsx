import React from 'react';
import { SunIcon, MoonIcon } from './ui/index';
import { Theme } from '../types';

const MenuIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
  </svg>
);

export const Header = ({ theme, setTheme, onMenuClick }: { theme: Theme, setTheme: (theme: Theme) => void, onMenuClick: () => void }) => {
  return (
    <header className="bg-[var(--card-background)] shadow-md sticky top-0 z-20 flex-shrink-0">
      <div className="container mx-auto px-6 py-2 flex justify-between items-center">
        <div className="flex items-center gap-4">
          <button onClick={onMenuClick} className="p-2 rounded-full text-[var(--text-muted)] hover:bg-[var(--secondary)] lg:hidden">
            <MenuIcon />
          </button>
          <h1 className="text-xl sm:text-2xl font-bold text-[var(--foreground)]">GRU-HHO Forex Prediction</h1>
        </div>
        <div className="flex items-center">
          <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')} className="p-2 rounded-full text-[var(--text-muted)] hover:bg-[var(--secondary)]">
            {theme === 'light' ? <MoonIcon /> : <SunIcon />}
          </button>
        </div>
      </div>
    </header>
  );
};
