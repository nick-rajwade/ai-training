import React from 'react';
import katex from 'katex';

const MathRenderer = ({ math, displayMode = false }) => {
  const html = katex.renderToString(math, {
    displayMode: displayMode,
    throwOnError: false,
    errorColor: '#cc0000',
    strict: 'warn',
    trust: false,
    macros: {
      '\\RR': '\\mathbb{R}',
      '\\mathbb{R}': '\\mathbb{R}',
      '\\times': '\\times',
      '\\ll': '\\ll',
      '\\approx': '\\approx',
      '\\Delta': '\\Delta'
    }
  });

  return (
    <span 
      className={displayMode ? "math-display" : "math-inline"}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};

export default MathRenderer;
