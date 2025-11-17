import React, { useState, useEffect } from 'react';

const TableOfContents = ({ content }) => {
  const [headings, setHeadings] = useState([]);
  const [activeId, setActiveId] = useState('');
  const [isOpen, setIsOpen] = useState(true);

  useEffect(() => {
    // Extract headings from content
    const extractHeadings = () => {
      const lines = content.split('\n');
      const extractedHeadings = [];
      
      lines.forEach((line, index) => {
        // Match markdown headers (# ## ### ####) - ensure they start at the beginning
        const trimmedLine = line.trim();
        
        // Helper function to clean heading text
        const cleanHeadingText = (text) => {
          return text
            .replace(/\*\*/g, '') // Remove bold markdown
            .replace(/\*/g, '')   // Remove italic markdown
            .replace(/`/g, '')    // Remove code backticks
            .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1') // Remove markdown links, keep text
            .replace(/^[\d.]+\s+/, '') // Remove leading numbers (e.g., "1.1 ")
            .trim();
        };
        
        let match;
        // Only extract H2 headers that start with a number (main numbered sections)
        if ((match = trimmedLine.match(/^##\s+(\d+\..+)$/))) {
          extractedHeadings.push({
            id: `heading-${index}`,
            text: cleanHeadingText(match[1]),
            level: 2
          });
        }
      });
      
      setHeadings(extractedHeadings);
    };

    extractHeadings();
  }, [content]);

  useEffect(() => {
    // Observe which heading is currently in view
    const observer = new IntersectionObserver(
      (entries) => {
        // Find the entry that is most visible
        const visibleEntries = entries.filter(entry => entry.isIntersecting);
        if (visibleEntries.length > 0) {
          // Get the heading that's most visible (closest to top of viewport)
          const topMostEntry = visibleEntries.reduce((closest, entry) => {
            return entry.boundingClientRect.top < closest.boundingClientRect.top ? entry : closest;
          });
          setActiveId(topMostEntry.target.id);
        }
      },
      {
        rootMargin: '-120px 0px -70% 0px', // Adjusted for better detection
        threshold: [0, 0.1, 0.5, 1.0] // Multiple thresholds for better accuracy
      }
    );

    // Wait a bit for content to render, then observe heading elements
    const timer = setTimeout(() => {
      const headingElements = document.querySelectorAll('h1[id^="heading-"], h2[id^="heading-"], h3[id^="heading-"], h4[id^="heading-"]');
      headingElements.forEach((element) => observer.observe(element));
    }, 100);

    return () => {
      clearTimeout(timer);
      const headingElements = document.querySelectorAll('h1[id^="heading-"], h2[id^="heading-"], h3[id^="heading-"], h4[id^="heading-"]');
      headingElements.forEach((element) => observer.unobserve(element));
    };
  }, [headings]);

  const scrollToHeading = (headingId) => {
    const element = document.getElementById(headingId);
    if (element) {
      const yOffset = -120; // Offset for fixed header and some padding
      const y = element.getBoundingClientRect().top + window.pageYOffset + yOffset;
      window.scrollTo({ top: y, behavior: 'smooth' });
      
      // Update active section immediately for better UX
      setActiveId(headingId);
    }
  };

  const getPaddingClass = (level) => {
    switch (level) {
      case 1:
        return 'pl-2';
      case 2:
        return 'pl-4';
      case 3:
        return 'pl-6';
      case 4:
        return 'pl-8';
      default:
        return 'pl-2';
    }
  };

  const getFontSizeClass = (level) => {
    switch (level) {
      case 1:
        return 'text-sm font-semibold';
      case 2:
        return 'text-sm';
      case 3:
        return 'text-xs';
      case 4:
        return 'text-xs';
      default:
        return 'text-sm';
    }
  };

  if (headings.length === 0) return null;

  return (
    <div className={`sticky top-0 h-screen bg-white border-r-2 border-[#E8F4F9] flex-shrink-0 transition-all duration-300 ${isOpen ? 'w-80' : 'w-16'} hidden lg:flex flex-col shadow-lg`}>
      {/* Header */}
      <div 
        className="flex items-center justify-between p-5 border-b-2 border-[#0070AD] cursor-pointer bg-gradient-to-r from-[#F0F8FC] to-[#E8F4F9] hover:from-[#E8F4F9] hover:to-[#D0E9F5] transition-all duration-300"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? (
          <>
            <h3 className="font-bold text-gray-900 flex items-center text-lg">
              <svg 
                className="w-6 h-6 mr-3 text-[#0070AD]" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2.5} 
                  d="M4 6h16M4 12h16M4 18h16" 
                />
              </svg>
              Table of Contents
            </h3>
            <button 
              className="text-[#0070AD] hover:text-[#005A8C] transition-colors duration-300"
            >
              <svg 
                className="w-6 h-6" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2.5} 
                  d="M11 19l-7-7 7-7m8 14l-7-7 7-7" 
                />
              </svg>
            </button>
          </>
        ) : (
          <button 
            className="text-[#0070AD] hover:text-[#005A8C] mx-auto transition-colors duration-300"
          >
            <svg 
              className="w-6 h-6" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2.5} 
                d="M13 5l7 7-7 7M5 5l7 7-7 7" 
              />
            </svg>
          </button>
        )}
      </div>

      {/* Content */}
      {isOpen && (
        <div className="overflow-y-auto flex-1 p-5 bg-gradient-to-b from-white to-[#F8FAFB]">
          <nav>
            <ul className="space-y-2">
              {headings.map((heading) => (
                <li key={heading.id}>
                  <button
                    onClick={() => scrollToHeading(heading.id)}
                    className={`
                      w-full text-left py-3 px-4 rounded-lg transition-all duration-300
                      ${getFontSizeClass(heading.level)}
                      ${activeId === heading.id 
                        ? 'bg-gradient-to-r from-[#0070AD] to-[#00A1DE] text-white border-l-4 border-[#12239E] font-bold shadow-lg scale-105' 
                        : 'text-gray-700 hover:bg-[#E8F4F9] border-l-4 border-transparent hover:border-[#0070AD] hover:text-[#0070AD] font-medium hover:scale-102'
                      }
                    `}
                  >
                    {heading.text}
                  </button>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      )}

      {/* Footer - Hint */}
      {isOpen && (
        <div className="border-t-2 border-[#E8F4F9] px-5 py-4 bg-gradient-to-r from-[#F0F8FC] to-[#E8F4F9]">
          <p className="text-xs text-gray-600 text-center font-semibold">
            💡 Click to navigate • Collapse with →
          </p>
        </div>
      )}
    </div>
  );
};

export default TableOfContents;
