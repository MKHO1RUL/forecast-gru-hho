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
  const finalGlowColor = glowColor || `${color}4D`;

  const style: React.CSSProperties = {
    color: finalGlowColor,
    textShadow: `0 0 1px ${finalGlowColor}`,
    backgroundImage: `radial-gradient(${color} 0.6px, transparent 0.6px)`,
    backgroundSize: '2px 2px',
    WebkitBackgroundClip: 'text',
    backgroundClip: 'text',
  };

  return (
    <span className={`dot-matrix-text ${className}`} style={style}>
      {text}
    </span>
  );
};
