// components/DotMatrixText.tsx
import React from 'react';

interface DotMatrixTextProps {
  text: string;
  color?: string;
  glowColor?: string;
  className?: string;
}

export const DotMatrixText: React.FC<DotMatrixTextProps> = ({
  text,
  color = '#99ff33',
  glowColor,
  className = '',
}) => {
  // Gunakan versi warna yang semi-transparan untuk efek cahaya
  const finalGlowColor = glowColor || `${color}4D`; // 80 dalam hex setara dengan ~50% opacity

  const style: React.CSSProperties = {
    color: finalGlowColor, // Ini akan menjadi cahaya di antara titik-titik
    textShadow: `0 0 1px ${finalGlowColor}`, // Mengurangi glow agar titik lebih tajam
    backgroundImage: `radial-gradient(${color} 0.6px, transparent 0.6px)`, // Ukuran titik lebih kecil
    backgroundSize: '2px 2px', // Jarak antar titik lebih rapat
    WebkitBackgroundClip: 'text',
    backgroundClip: 'text',
  };

  return (
    <span className={`dot-matrix-text ${className}`} style={style}>
      {text}
    </span>
  );
};
