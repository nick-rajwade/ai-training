import React, { useState, useCallback, useMemo, Suspense, lazy } from 'react';
import { trainingContent } from './data/content';
import katex from 'katex';
import './App.css';
import ScrollToTop from './components/ScrollToTop';
import TableOfContents from './components/TableOfContents';
import CapgeminiLogo from './components/CapgeminiLogo';
import LoadingSpinner from './components/LoadingSpinner';
import SearchModal from './components/SearchModal';
import NotesPanel from './components/NotesPanel';
import { useBookmarks } from './contexts/BookmarkContext';
import { printModule } from './utils/printUtils';
import Footer from './components/Footer';

function App() {
  const [currentView, setCurrentView] = useState('overview');
  const [selectedModule, setSelectedModule] = useState(null);
  const [completedModules, setCompletedModules] = useState(() => {
    const saved = localStorage.getItem('ai-training-completed');
    return saved ? new Set(JSON.parse(saved)) : new Set();
  });
  const [showHeader, setShowHeader] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isNotesOpen, setIsNotesOpen] = useState(false);
  const [readingProgress, setReadingProgress] = useState(0);
  const [searchHighlight, setSearchHighlight] = useState(null);
  
  const { addBookmark, removeBookmark, isBookmarked, addNote, getNote } = useBookmarks();

  // Handle scroll to show/hide header
  React.useEffect(() => {
    const handleScroll = () => {
      const heroHeight = 400; // Approximate height of hero section
      setShowHeader(window.scrollY > heroHeight);
    };

    if (currentView === 'overview') {
      window.addEventListener('scroll', handleScroll);
      return () => window.removeEventListener('scroll', handleScroll);
    }
  }, [currentView]);

  // Save completed modules to localStorage
  React.useEffect(() => {
    localStorage.setItem('ai-training-completed', JSON.stringify([...completedModules]));
  }, [completedModules]);

  // Add keyboard shortcuts
  React.useEffect(() => {
    const handleKeyDown = (e) => {
      // Ctrl/Cmd + K for search
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setIsSearchOpen(true);
      }
      // Ctrl/Cmd + N for notes (only in module view)
      if ((e.ctrlKey || e.metaKey) && e.key === 'n' && currentView === 'module') {
        e.preventDefault();
        e.stopPropagation();
        setIsNotesOpen(!isNotesOpen);
      }
      // Escape to go back, close panels, or clear search highlights
      if (e.key === 'Escape') {
        if (isNotesOpen) {
          setIsNotesOpen(false);
        } else if (searchHighlight) {
          setSearchHighlight(null); // Clear search highlights
        } else if (currentView === 'module') {
          handleBackToOverview();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [currentView, isNotesOpen, searchHighlight]);

  // Track reading progress in module view
  React.useEffect(() => {
    if (currentView !== 'module') return;

    const handleScroll = () => {
      const scrollTop = window.pageYOffset;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = (scrollTop / docHeight) * 100;
      setReadingProgress(Math.min(Math.max(progress, 0), 100));
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [currentView]);

  // Handle scrolling to search results
  React.useEffect(() => {
    if (currentView === 'module' && selectedModule) {
      const searchResult = sessionStorage.getItem('searchResult');
      if (searchResult) {
        try {
          const { contentId, searchTerm, lineNumber } = JSON.parse(searchResult);
          setSearchHighlight({ searchTerm, lineNumber });
          
          // Function to find and scroll to element with retry logic
          const findAndScrollToElement = (retryCount = 0) => {
            // First try to find element by search ID
            let element = document.getElementById(contentId);
            
            // If not found, try to find by data-search-id attribute (for headings)
            if (!element) {
              element = document.querySelector(`[data-search-id="${contentId}"]`);
            }
            
            if (element) {
              const yOffset = -120; // Offset for fixed header
              const y = element.getBoundingClientRect().top + window.pageYOffset + yOffset;
              window.scrollTo({ top: y, behavior: 'smooth' });
              
              // Add a pulsing highlight effect
              element.classList.add('search-highlight-pulse');
              setTimeout(() => {
                element.classList.remove('search-highlight-pulse');
              }, 3000);
            } else if (retryCount < 10) {
              // Retry up to 10 times with increasing delays
              setTimeout(() => findAndScrollToElement(retryCount + 1), 200 * (retryCount + 1));
            }
          };
          
          // Start the search with initial delay
          setTimeout(() => findAndScrollToElement(), 500);
          
          // Clear the search result from session storage after a longer delay
          setTimeout(() => {
            sessionStorage.removeItem('searchResult');
          }, 5000);
        } catch (error) {
          console.error('Error parsing search result:', error);
        }
      }
    }
  }, [currentView, selectedModule]);

  const handleModuleClick = useCallback(async (moduleId) => {
    setIsLoading(true);
    // Clear any previous search highlights when navigating to a different module
    if (selectedModule?.id !== moduleId) {
      setSearchHighlight(null);
    }
    
    // Simulate a small delay for smooth UX
    await new Promise(resolve => setTimeout(resolve, 150));
    
    const module = trainingContent.find(m => m.id === moduleId);
    setSelectedModule(module);
    setCurrentView('module');
    setIsLoading(false);
    
    // Scroll to top when entering module (unless we have a search result to scroll to)
    if (!sessionStorage.getItem('searchResult')) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [selectedModule]);

  const handleBackToOverview = useCallback(() => {
    setCurrentView('overview');
    setSelectedModule(null);
    setSearchHighlight(null); // Clear search highlighting
    sessionStorage.removeItem('searchResult'); // Clean up any remaining search data
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  const markModuleComplete = useCallback((moduleId) => {
    setCompletedModules(prev => {
      const newSet = new Set([...prev, moduleId]);
      // Show celebration effect
      if (!prev.has(moduleId)) {
        // Could add confetti or other celebration here
        console.log(`🎉 Module ${moduleId} completed!`);
      }
      return newSet;
    });
  }, []);

  const handleSearchResult = useCallback((result) => {
    if (result.type === 'module' || result.type === 'topic') {
      handleModuleClick(result.moduleId);
    } else if (result.type === 'content') {
      // Store search result data for scrolling and highlighting
      const searchData = {
        contentId: result.contentId,
        searchTerm: result.searchTerm,
        lineNumber: result.lineNumber
      };
      sessionStorage.setItem('searchResult', JSON.stringify(searchData));
      handleModuleClick(result.moduleId);
    }
  }, [handleModuleClick]);

  const renderMath = (mathExpression, displayMode = false) => {
    try {
      // Clean up the math expression
      let processedMath = mathExpression.trim();
      
      // Remove any extra whitespace and normalize
      processedMath = processedMath.replace(/\s+/g, ' ');
      
      // Fix common issues with text commands
      processedMath = processedMath.replace(/\\text\{([^}]*)\}/g, '\\mathrm{$1}');
      
      // Ensure proper spacing around operators
      processedMath = processedMath.replace(/([a-zA-Z0-9}])\s*=\s*([a-zA-Z0-9\\])/g, '$1 = $2');
      processedMath = processedMath.replace(/([a-zA-Z0-9}])\s*\+\s*([a-zA-Z0-9\\])/g, '$1 + $2');
      processedMath = processedMath.replace(/([a-zA-Z0-9}])\s*-\s*([a-zA-Z0-9\\])/g, '$1 - $2');
      
      // Fix common subscript/superscript spacing issues
      processedMath = processedMath.replace(/([A-Za-z])_([A-Za-z]+)/g, '$1_{$2}');
      processedMath = processedMath.replace(/([A-Za-z])\^([A-Za-z]+)/g, '$1^{$2}');
      
      return katex.renderToString(processedMath, {
        displayMode: displayMode,
        throwOnError: false,
        errorColor: '#cc0000',
        strict: false, // More lenient parsing
        trust: true,
        macros: {
          '\\RR': '\\mathbb{R}',
          '\\R': '\\mathbb{R}',
          '\\leq': '\\leq',
          '\\geq': '\\geq',
          '\\infty': '\\infty',
          '\\in': '\\in',
          '\\times': '\\times',
          '\\cdot': '\\cdot',
          '\\ll': '\\ll',
          '\\approx': '\\approx',
          '\\min': '\\operatorname{min}',
          '\\max': '\\operatorname{max}',
          '\\sum': '\\sum',
          '\\exp': '\\exp',
          '\\log': '\\log',
          '\\sin': '\\sin',
          '\\cos': '\\cos',
          '\\tan': '\\tan',
          '\\softmax': '\\operatorname{softmax}',
          '\\argmax': '\\operatorname{argmax}',
          '\\argmin': '\\operatorname{argmin}',
          '\\Attention': '\\operatorname{Attention}',
          '\\MultiHead': '\\operatorname{MultiHead}',
          '\\CrossAttention': '\\operatorname{CrossAttention}',
          '\\Concat': '\\operatorname{Concat}',
          '\\head': '\\operatorname{head}'
        }
      });
    } catch (error) {
      console.error('KaTeX rendering error for:', mathExpression, error);
      // Return a more user-friendly error message without red styling
      return `<span style="color: #666; font-style: italic;">[${mathExpression}]</span>`;
    }
  };

  const formatContent = (content) => {
    // Split content into lines for processing
    const lines = content.split('\n');
    const processedLines = [];
    let inCodeBlock = false;
    
    for (let i = 0; i < lines.length; i++) {
      let line = lines[i];
      
      // Store original line for ID generation
      const originalLine = line;
      
      // Handle code blocks
      if (line.trim().startsWith('```')) {
        if (!inCodeBlock) {
          processedLines.push(`<div class="bg-gradient-to-br from-[#F8FAFB] to-[#F0F8FC] p-6 rounded-lg my-6 font-mono text-sm overflow-x-auto border-2 border-[#E8F4F9] shadow-md">`);
          inCodeBlock = true;
        } else {
          processedLines.push(`</div>`);
          inCodeBlock = false;
        }
        continue;
      }
      
      // If we're in a code block, don't process formatting
      if (inCodeBlock) {
        const contentId = `search-result-${selectedModule?.id}-${i}`;
        processedLines.push(`<div id="${contentId}" class="text-gray-800 leading-relaxed">${line}</div>`);
        continue;
      }
      
      // Handle headers with IDs for table of contents
      if (line.startsWith('# ')) {
        const headingText = line.substring(2);
        const headingId = `heading-${i}`;
        const searchId = `search-result-${selectedModule?.id}-${i}`;
        processedLines.push(`<h1 id="${headingId}" data-search-id="${searchId}" class="text-4xl font-bold mt-10 mb-6 bg-gradient-to-r from-[#0070AD] to-[#12239E] bg-clip-text text-transparent border-b-4 border-[#0070AD] pb-3">${headingText}</h1>`);
        continue;
      }
      if (line.startsWith('## ')) {
        const headingText = line.substring(3);
        const headingId = `heading-${i}`;
        const searchId = `search-result-${selectedModule?.id}-${i}`;
        processedLines.push(`<h2 id="${headingId}" data-search-id="${searchId}" class="text-3xl font-bold mt-8 mb-5 text-[#0070AD] border-l-4 border-[#0070AD] pl-4">${headingText}</h2>`);
        continue;
      }
      if (line.startsWith('### ')) {
        const headingText = line.substring(4);
        const headingId = `heading-${i}`;
        const searchId = `search-result-${selectedModule?.id}-${i}`;
        processedLines.push(`<h3 id="${headingId}" data-search-id="${searchId}" class="text-2xl font-semibold mt-6 mb-4 text-[#00A1DE]">${headingText}</h3>`);
        continue;
      }
      if (line.startsWith('#### ')) {
        const headingText = line.substring(5);
        const headingId = `heading-${i}`;
        const searchId = `search-result-${selectedModule?.id}-${i}`;
        processedLines.push(`<h4 id="${headingId}" data-search-id="${searchId}" class="text-xl font-semibold mt-5 mb-3 text-gray-800">${headingText}</h4>`);
        continue;
      }
      
      // Handle bullet points
      if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
        const bulletText = line.trim().substring(2);
        // Process formatting in bullet text
        const formattedBullet = processInlineFormatting(bulletText);
        const contentId = `search-result-${selectedModule?.id}-${i}`;
        processedLines.push(`<li id="${contentId}" class="ml-6 mb-2 text-gray-800 list-disc marker:text-[#0070AD]">${formattedBullet}</li>`);
        continue;
      }
      
      // Handle numbered lists
      if (/^\d+\.\s/.test(line.trim())) {
        const listText = line.trim().replace(/^\d+\.\s/, '');
        const formattedList = processInlineFormatting(listText);
        const contentId = `search-result-${selectedModule?.id}-${i}`;
        processedLines.push(`<li id="${contentId}" class="ml-6 mb-2 text-gray-800 list-decimal marker:text-[#0070AD] marker:font-bold">${formattedList}</li>`);
        continue;
      }
      
      // Handle display math expressions that are on their own line
      if (line.trim().match(/^\$\$.*\$\$$/) && line.trim().length > 4) {
        const mathExpression = line.trim().slice(2, -2);
        if (mathExpression.trim()) {
          const rendered = renderMath(mathExpression.trim(), true);
          const contentId = `search-result-${selectedModule?.id}-${i}`;
          processedLines.push(`<div id="${contentId}" class="math-display my-6 text-center">${rendered}</div>`);
          continue;
        }
      }
      
      // Handle single $ display math on its own line (less common but possible)
      if (line.trim().startsWith('$') && line.trim().endsWith('$') && line.trim().length > 2 && !line.includes('$$')) {
        const mathExpression = line.trim().slice(1, -1);
        if (mathExpression.trim()) {
          const rendered = renderMath(mathExpression.trim(), true);
          const contentId = `search-result-${selectedModule?.id}-${i}`;
          processedLines.push(`<div id="${contentId}" class="math-display my-6 text-center">${rendered}</div>`);
          continue;
        }
      }
      
      // Handle empty lines
      if (line.trim() === '') {
        processedLines.push(`<br />`);
        continue;
      }
      
      // Process regular paragraphs with inline formatting
      const formattedLine = processInlineFormatting(line);
      const contentId = `search-result-${selectedModule?.id}-${i}`;
      processedLines.push(`<p id="${contentId}" class="mb-3 text-gray-700 leading-relaxed">${formattedLine}</p>`);
    }
    
    return processedLines.join('');
  };
  
  const processInlineFormatting = (text) => {
    // Handle LaTeX math expressions first (before other formatting)
    // Handle display math expressions ($$...$$)
    text = text.replace(/\$\$([^$]+?)\$\$/g, (match, mathExpression) => {
      const cleanExpression = mathExpression.trim();
      if (cleanExpression) {
        return `<div class="math-display my-4 text-center">${renderMath(cleanExpression, true)}</div>`;
      }
      return match;
    });
    
    // Handle inline math expressions ($...$ but not $$...$$)
    text = text.replace(/(?<!\$)\$([^$\n]+?)\$(?!\$)/g, (match, mathExpression) => {
      const cleanExpression = mathExpression.trim();
      if (cleanExpression) {
        return `<span class="math-inline">${renderMath(cleanExpression, false)}</span>`;
      }
      return match;
    });
    
    // Handle mathematical expressions (bold formatting for math)
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong class="font-bold text-[#0070AD]">$1</strong>');
    
    // Handle italic text
    text = text.replace(/\*([^*]+)\*/g, '<em class="italic text-gray-700">$1</em>');
    
    // Handle inline code
    text = text.replace(/`([^`]+)`/g, '<code class="bg-[#F0F8FC] px-3 py-1 rounded-md text-sm font-mono border border-[#E8F4F9] text-[#0070AD]">$1</code>');
    
    // Apply search highlighting if we have a search term
    if (searchHighlight && searchHighlight.searchTerm && text.toLowerCase().includes(searchHighlight.searchTerm.toLowerCase())) {
      const regex = new RegExp(`(${searchHighlight.searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
      text = text.replace(regex, '<mark class="bg-yellow-300 px-1 rounded font-semibold">$1</mark>');
    }
    
    return text;
  };

  const renderOverview = () => (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Conditional Header */}
      {showHeader && (
        <header className="bg-white shadow-md border-b-2 border-[#0070AD] fixed top-0 left-0 right-0 z-50 transform transition-transform duration-300">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-center items-center py-4 relative">
              <div className="flex items-center text-2xl font-bold bg-gradient-to-r from-[#0070AD] to-[#12239E] bg-clip-text text-transparent justify-center">
                <CapgeminiLogo className="mr-3" width="32" height="30" />
                AI Training Platform
              </div>
              <div className="flex items-center space-x-2 absolute right-0">
                <button
                  onClick={() => setIsSearchOpen(true)}
                  className="p-3 rounded-lg transition-all duration-300 hover:scale-110 focus:outline-none focus:ring-2 focus:ring-[#0070AD] focus:ring-offset-2 hover:bg-gray-100"
                  aria-label="Search content"
                  title="Search content (Ctrl+K)"
                >
                  <svg className="w-6 h-6 text-gray-600 hover:text-[#0070AD]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </header>
      )}

      {/* Hero Section */}
      <div className={`capgemini-gradient text-white shadow-xl ${showHeader ? 'pt-20' : ''}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <div className="text-center mb-8">
              <h1 className="text-4xl md:text-5xl font-bold drop-shadow-lg">
                AI Training Excellence
              </h1>
            </div>
            <p className="text-xl mb-4 font-light">Capgemini's Comprehensive Guide to AI/ML</p>
            <p className="text-lg opacity-95 mb-8 max-w-3xl mx-auto leading-relaxed">
              Master AI and Machine Learning concepts with business applications, 
              real-world insights, and hands-on implementation strategies designed for consultants.
            </p>
            
            {/* Search Action */}
            <div className="flex justify-center mb-12">
              <button
                onClick={() => setIsSearchOpen(true)}
                className="bg-white/20 hover:bg-white/30 px-8 py-4 rounded-xl transition-all duration-300 hover:scale-105 backdrop-blur-sm border border-white/30 flex items-center space-x-3 text-lg font-medium"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <span>Search All Training Content</span>
                <span className="text-sm bg-white/20 px-3 py-1 rounded-lg">Ctrl+K</span>
              </button>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto text-sm">
              <div className="bg-white/15 backdrop-blur-sm px-4 py-3 rounded-lg border border-white/20">
                <div className="font-semibold">8 Modules</div>
                <div className="text-xs opacity-90">Comprehensive Coverage</div>
              </div>
              <div className="bg-white/15 backdrop-blur-sm px-4 py-3 rounded-lg border border-white/20">
                <div className="font-semibold">Interactive</div>
                <div className="text-xs opacity-90">Hands-on Learning</div>
              </div>
              <div className="bg-white/15 backdrop-blur-sm px-4 py-3 rounded-lg border border-white/20">
                <div className="font-semibold">Business-Focused</div>
                <div className="text-xs opacity-90">Real Applications</div>
              </div>
              <div className="bg-white/15 backdrop-blur-sm px-4 py-3 rounded-lg border border-white/20">
                <div className="font-semibold">Industry-Ready</div>
                <div className="text-xs opacity-90">Practical Skills</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Progress Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="bg-white rounded-xl shadow-lg p-8 mb-12 border-l-4 border-[#0070AD]">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Learning Progress</h2>
            <p className="text-gray-600">Track your journey through the AI training modules</p>
          </div>
          
          <div className="mb-8">
            <div className="flex justify-between text-sm text-gray-700 mb-3 font-medium">
              <span>Modules Completed</span>
              <span className="text-[#0070AD] font-semibold">{completedModules.size} of {trainingContent.length}</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div 
                className="bg-gradient-to-r from-[#0070AD] to-[#00A1DE] h-3 rounded-full transition-all duration-500"
                style={{ width: `${(completedModules.size / trainingContent.length) * 100}%` }}
              ></div>
            </div>
            <div className="mt-2 text-right text-xs text-gray-500">
              {Math.round((completedModules.size / trainingContent.length) * 100)}% Complete
            </div>
          </div>

          <div className="grid grid-cols-4 md:grid-cols-8 gap-3">
            {trainingContent.map((module) => (
              <div
                key={module.id}
                className={`text-center p-3 rounded-lg text-sm cursor-pointer transition-all duration-200 font-medium ${
                  completedModules.has(module.id)
                    ? 'bg-green-500 text-white shadow-sm'
                    : 'bg-gray-100 text-gray-600 hover:bg-[#0070AD] hover:text-white'
                }`}
                onClick={() => handleModuleClick(module.id)}
                title={`Module ${module.id}`}
              >
                {completedModules.has(module.id) ? '✓' : module.id}
              </div>
            ))}
          </div>
        </div>

        {/* Training Modules */}
        <div className="mb-12">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              <span className="bg-gradient-to-r from-[#0070AD] to-[#12239E] bg-clip-text text-transparent">
                Training Modules
              </span>
            </h2>
            <p className="text-gray-600 text-lg max-w-2xl mx-auto">
              Comprehensive AI and Machine Learning curriculum designed for business consultants
            </p>
          </div>
          
          <div className="grid gap-6">
            {trainingContent.map((module) => (
              <div
                key={module.id}
                className="bg-white rounded-lg shadow-md border border-gray-200 hover:shadow-lg hover:border-[#0070AD] transition-all duration-300 cursor-pointer group"
                onClick={() => handleModuleClick(module.id)}
              >
                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="flex items-center mb-2">
                        <div className="bg-[#0070AD] text-white text-sm font-bold px-3 py-1 rounded-full mr-4">
                          {module.id}
                        </div>
                        <h3 className="text-xl font-bold text-gray-900 group-hover:text-[#0070AD] transition-colors duration-300">
                          {module.title}
                        </h3>
                      </div>
                      <p className="text-sm text-gray-600 mb-4 leading-relaxed">{module.subtitle}</p>
                      
                      <div className="flex flex-wrap gap-2 mb-4">
                        {module.topics.slice(0, 4).map((topic, index) => (
                          <span key={index} className="bg-gray-100 text-gray-700 text-xs px-3 py-1 rounded-full">
                            {topic}
                          </span>
                        ))}
                        {module.topics.length > 4 && (
                          <span className="bg-[#0070AD] text-white text-xs px-3 py-1 rounded-full">
                            +{module.topics.length - 4} more
                          </span>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex flex-col items-end ml-6">
                      {completedModules.has(module.id) && (
                        <div className="text-green-600 bg-green-100 rounded-full w-8 h-8 flex items-center justify-center mb-2">
                          ✓
                        </div>
                      )}
                      <button className="bg-[#0070AD] text-white px-6 py-2 rounded-md text-sm font-medium hover:bg-[#005A8C] transition-colors duration-200">
                        {completedModules.has(module.id) ? 'Review' : 'Start'}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <Footer />
    </div>
  );

  const renderModule = () => {
    if (!selectedModule) return null;

    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 flex">
        {/* Table of Contents Sidebar */}
        <TableOfContents content={selectedModule.content} />
        
        {/* Notes Panel */}
        <NotesPanel 
          moduleId={selectedModule.id}
          isOpen={isNotesOpen}
          onClose={() => setIsNotesOpen(false)}
        />
        
        {/* Main Content Area */}
        <div className={`flex-1 flex flex-col min-w-0 transition-all duration-300 ${isNotesOpen ? 'pr-96' : ''}`}>
          {/* Reading Progress Bar */}
          <div className="fixed top-0 left-0 right-0 h-1 bg-gray-200 z-50">
            <div 
              className="h-full bg-gradient-to-r from-[#0070AD] to-[#00A1DE] transition-all duration-300"
              style={{ width: `${readingProgress}%` }}
            ></div>
          </div>

          {/* Header */}
          <header className="bg-white shadow-md border-b-2 border-[#0070AD] sticky top-0 z-40">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between items-center py-6">
                <div className="flex items-center flex-1 min-w-0">
                  <button
                    onClick={handleBackToOverview}
                    className="text-[#0070AD] hover:text-[#005A8C] mr-6 flex items-center font-semibold transition-colors duration-300 hover:scale-105 flex-shrink-0"
                    title="Back to Overview (Esc)"
                  >
                    <span className="text-xl mr-2">←</span> Back
                  </button>
                  <div className="text-xl md:text-2xl font-bold bg-gradient-to-r from-[#0070AD] to-[#12239E] bg-clip-text text-transparent flex items-center min-w-0">
                    <CapgeminiLogo className="mr-3 flex-shrink-0" width="32" height="30" />
                    <span className="truncate">{selectedModule.title}</span>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3 flex-shrink-0">
                  {/* Clear Search Highlights */}
                  {searchHighlight && (
                    <button
                      onClick={() => setSearchHighlight(null)}
                      className="px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white rounded-lg transition-colors font-medium text-sm flex items-center space-x-2"
                      title="Clear search highlights (Esc)"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                      <span>Clear Highlights</span>
                    </button>
                  )}

                  {/* Notes Toggle */}
                  <button
                    onClick={() => setIsNotesOpen(!isNotesOpen)}
                    className={`p-3 rounded-lg transition-colors ${isNotesOpen ? 'bg-[#0070AD] text-white' : 'hover:bg-gray-100 text-gray-600'}`}
                    title="Toggle notes (Ctrl+N)"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                  </button>

                  {/* Search Button */}
                  <button
                    onClick={() => setIsSearchOpen(true)}
                    className="p-3 rounded-lg hover:bg-gray-100 transition-colors text-gray-600"
                    title="Search content (Ctrl+K)"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </button>

                  {/* Complete Button */}
                  <button
                    onClick={() => markModuleComplete(selectedModule.id)}
                    className={`px-6 py-3 rounded-lg transition-all duration-300 font-semibold shadow-md hover:shadow-lg ${
                      completedModules.has(selectedModule.id)
                        ? 'bg-gradient-to-r from-green-500 to-green-600 text-white'
                        : 'bg-gradient-to-r from-[#0070AD] to-[#00A1DE] text-white hover:scale-105'
                    }`}
                  >
                    {completedModules.has(selectedModule.id) ? '✓ Completed' : 'Mark Complete'}
                  </button>
                </div>
              </div>
            </div>
          </header>

          {/* Module Content */}
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
              <div className="bg-white rounded-xl shadow-xl p-10 border-t-4 border-[#0070AD]">
                <div className="mb-8">
                  <div className="flex items-center mb-4">
                    <CapgeminiLogo className="mr-4" width="60" height="56" />
                    <div>
                      <h1 className="text-4xl font-bold bg-gradient-to-r from-[#0070AD] to-[#12239E] bg-clip-text text-transparent mb-2">
                        {selectedModule.title}
                      </h1>
                      <p className="text-xl text-gray-700 mb-4 font-medium">{selectedModule.subtitle}</p>
                    </div>
                  </div>
                  <p className="text-gray-700 leading-relaxed text-lg border-l-4 border-[#0070AD] pl-6 py-2 bg-[#F0F8FC]">
                    {selectedModule.description}
                  </p>
                </div>

                {/* Topics Overview */}
                <div className="mb-10 bg-gradient-to-br from-[#F0F8FC] to-white rounded-lg p-6 border border-[#E8F4F9]">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                    <span className="text-[#0070AD] mr-3">📋</span> Topics Covered
                  </h2>
                  <div className="grid md:grid-cols-2 gap-4">
                    {selectedModule.topics.map((topic, index) => (
                      <div key={index} className="flex items-start p-4 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow duration-300 border border-gray-100">
                        <div className="text-[#0070AD] mr-3 mt-0.5 font-bold">✓</div>
                        <div className="text-gray-800 font-medium">{topic}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Content */}
                <div className="prose max-w-none">
                  <div 
                    className="text-gray-800 leading-relaxed"
                    dangerouslySetInnerHTML={{ 
                      __html: formatContent(selectedModule.content)
                    }}
                  />
                </div>

                {/* Navigation */}
                <div className="mt-10 pt-8 border-t-2 border-gray-200 flex justify-between">
                  <button
                    onClick={handleBackToOverview}
                    className="bg-gradient-to-r from-gray-500 to-gray-600 text-white px-8 py-3 rounded-lg hover:from-gray-600 hover:to-gray-700 transition-all duration-300 font-semibold shadow-md hover:shadow-lg hover:scale-105"
                  >
                    ← Back to Overview
                  </button>
                  <button
                    onClick={() => markModuleComplete(selectedModule.id)}
                    className="bg-gradient-to-r from-green-500 to-green-600 text-white px-8 py-3 rounded-lg hover:from-green-600 hover:to-green-700 transition-all duration-300 font-semibold shadow-md hover:shadow-lg hover:scale-105"
                  >
                    {completedModules.has(selectedModule.id) ? 'Completed ✓' : 'Complete Module ✓'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="App">
      {isLoading && (
        <div className="fixed inset-0 bg-white/80 backdrop-blur-sm z-50 flex items-center justify-center">
          <LoadingSpinner size="lg" text="Loading module..." />
        </div>
      )}
      
      <SearchModal
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        content={trainingContent}
        onSelectResult={handleSearchResult}
      />
      
      {currentView === 'overview' ? renderOverview() : renderModule()}
      <ScrollToTop />
    </div>
  );
}

export default App;

