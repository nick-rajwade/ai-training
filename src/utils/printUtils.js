export const printModule = (moduleTitle, moduleContent) => {
  const printWindow = window.open('', '_blank', 'width=800,height=600');
  
  const printHTML = `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>${moduleTitle} - Capgemini AI Training</title>
      <style>
        body {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          line-height: 1.6;
          color: #333;
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
        }
        
        h1, h2, h3, h4, h5, h6 {
          color: #0070AD;
          margin-top: 2rem;
          margin-bottom: 1rem;
        }
        
        h1 {
          border-bottom: 3px solid #0070AD;
          padding-bottom: 0.5rem;
        }
        
        h2 {
          border-bottom: 1px solid #E5E7EB;
          padding-bottom: 0.25rem;
        }
        
        p {
          margin-bottom: 1rem;
          text-align: justify;
        }
        
        code {
          background-color: #F3F4F6;
          padding: 0.25rem 0.5rem;
          border-radius: 0.25rem;
          font-family: 'Courier New', monospace;
          font-size: 0.9em;
        }
        
        pre {
          background-color: #F9FAFB;
          border: 1px solid #E5E7EB;
          border-radius: 0.5rem;
          padding: 1rem;
          overflow-x: auto;
          margin: 1rem 0;
        }
        
        blockquote {
          border-left: 4px solid #0070AD;
          padding-left: 1rem;
          margin: 1rem 0;
          background-color: #F0F8FC;
          padding: 1rem;
          border-radius: 0.25rem;
        }
        
        ul, ol {
          margin-bottom: 1rem;
          padding-left: 2rem;
        }
        
        li {
          margin-bottom: 0.5rem;
        }
        
        .header {
          text-align: center;
          margin-bottom: 2rem;
          padding-bottom: 1rem;
          border-bottom: 2px solid #0070AD;
        }
        
        .footer {
          margin-top: 2rem;
          padding-top: 1rem;
          border-top: 1px solid #E5E7EB;
          text-align: center;
          color: #6B7280;
          font-size: 0.9rem;
        }
        
        @media print {
          body {
            margin: 0;
            padding: 15mm;
          }
          
          .header {
            margin-bottom: 15mm;
          }
          
          h1 {
            page-break-after: avoid;
          }
          
          h2, h3, h4, h5, h6 {
            page-break-after: avoid;
            page-break-inside: avoid;
          }
          
          p, blockquote {
            page-break-inside: avoid;
            orphans: 3;
            widows: 3;
          }
          
          pre, code {
            page-break-inside: avoid;
          }
        }
      </style>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.25/dist/katex.min.css">
    </head>
    <body>
      <div class="header">
        <h1>${moduleTitle}</h1>
        <p><strong>Capgemini AI Training Platform</strong></p>
        <p>Generated on ${new Date().toLocaleDateString()}</p>
      </div>
      
      <div class="content">
        ${moduleContent}
      </div>
      
      <div class="footer">
        <p>© ${new Date().getFullYear()} Capgemini. AI Training Platform.</p>
        <p>This document was generated from the Capgemini AI Training Platform.</p>
      </div>
    </body>
    </html>
  `;
  
  printWindow.document.open();
  printWindow.document.write(printHTML);
  printWindow.document.close();
  
  // Wait for content to load then print
  printWindow.onload = () => {
    setTimeout(() => {
      printWindow.print();
    }, 500);
  };
};

export const exportModuleAsPDF = async (moduleTitle, moduleContent) => {
  // This would integrate with a PDF generation library in a real implementation
  // For now, we'll use the browser's print to PDF functionality
  printModule(moduleTitle, moduleContent);
};
