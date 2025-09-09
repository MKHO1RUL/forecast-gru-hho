import React, { useRef, useLayoutEffect } from 'react';
import { DotMatrixText } from './dotmatrixtext';

interface ScrollingDotMatrixProps {
  text: string;
  color?: string;
  pixelsPerSecond?: number;
  className?: string;
}

const ScrollItem = React.memo(({ text, color, className, fwdRef }: {
  text: string;
  color?: string;
  className?: string;
  fwdRef?: React.Ref<HTMLDivElement>;
}) => (
  <div className="dot-scroll-item" ref={fwdRef}>
    <DotMatrixText text={text} color={color} className={className} />
  </div>
));
ScrollItem.displayName = 'ScrollItem';

export const ScrollingDotMatrix: React.FC<ScrollingDotMatrixProps> = React.memo(({
  text,
  color,
  pixelsPerSecond = 50,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const itemRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    if (itemRef.current && containerRef.current) {
      const width = itemRef.current.scrollWidth;
      if (width > 0) {
        const duration = width / pixelsPerSecond;
        containerRef.current.style.setProperty('--duration', `${duration}s`);
      }
    }
  }, [text, pixelsPerSecond]);

  return (
    <div ref={containerRef} className="dot-scroll-container w-full">
      <ScrollItem text={text} color={color} className={className} fwdRef={itemRef} />
      <ScrollItem text={text} color={color} className={className} />
      <ScrollItem text={text} color={color} className={className} />
    </div>
  );
});

ScrollingDotMatrix.displayName = 'ScrollingDotMatrix';
