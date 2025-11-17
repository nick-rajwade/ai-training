import React, { useState, useEffect, useCallback } from 'react';

const SearchModal = ({ isOpen, onClose, content, onSelectResult }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(0);

  const searchContent = useCallback((query) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    const results = [];
    const queryLower = query.toLowerCase();

    content.forEach(module => {
      // Search in module title and description
      if (module.title.toLowerCase().includes(queryLower) || 
          module.description.toLowerCase().includes(queryLower)) {
        results.push({
          type: 'module',
          moduleId: module.id,
          title: module.title,
          description: module.description,
          icon: module.icon,
          relevance: 10
        });
      }

      // Search in topics
      module.topics.forEach(topic => {
        if (topic.toLowerCase().includes(queryLower)) {
          results.push({
            type: 'topic',
            moduleId: module.id,
            moduleTitle: module.title,
            title: topic,
            icon: module.icon,
            relevance: 8
          });
        }
      });

      // Search in content (limited to avoid overwhelming results)
      const contentLines = module.content.split('\n');
      contentLines.forEach((line, index) => {
        if (line.toLowerCase().includes(queryLower) && line.trim().length > 20) {
          const words = line.split(' ');
          const queryIndex = words.findIndex(word => 
            word.toLowerCase().includes(queryLower)
          );
          
          if (queryIndex !== -1) {
            const start = Math.max(0, queryIndex - 5);
            const end = Math.min(words.length, queryIndex + 15);
            const snippet = words.slice(start, end).join(' ');
            
            // Generate a unique ID for this content piece for scrolling
            const contentId = `search-result-${module.id}-${index}`;
            
            // Determine content type for better context
            let contentType = 'paragraph';
            if (line.startsWith('#')) {
              contentType = 'heading';
            } else if (line.startsWith('- ') || line.startsWith('* ')) {
              contentType = 'list item';
            } else if (line.includes('$$') || line.includes('$')) {
              contentType = 'equation';
            } else if (line.includes('```')) {
              contentType = 'code';
            }
            
            results.push({
              type: 'content',
              moduleId: module.id,
              moduleTitle: module.title,
              title: snippet,
              lineNumber: index,
              contentId: contentId,
              fullLine: line,
              searchTerm: query,
              contentType: contentType,
              icon: module.icon,
              relevance: 5
            });
          }
        }
      });
    });

    // Sort by relevance and limit results
    const sortedResults = results
      .sort((a, b) => b.relevance - a.relevance)
      .slice(0, 50);

    setSearchResults(sortedResults);
    setSelectedIndex(0);
  }, [content]);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      searchContent(searchQuery);
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery, searchContent]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!isOpen) return;

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(prev => 
            prev < searchResults.length - 1 ? prev + 1 : prev
          );
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(prev => prev > 0 ? prev - 1 : 0);
          break;
        case 'Enter':
          e.preventDefault();
          if (searchResults[selectedIndex]) {
            handleSelectResult(searchResults[selectedIndex]);
          }
          break;
        case 'Escape':
          onClose();
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, searchResults, selectedIndex, onClose]);

  const handleSelectResult = (result) => {
    onSelectResult(result);
    onClose();
    setSearchQuery('');
  };

  const highlightText = (text, query) => {
    if (!query.trim()) return text;
    
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    return text.replace(regex, '<mark class="bg-yellow-200 dark:bg-yellow-600 px-1 rounded">$1</mark>');
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-gray-900 bg-opacity-40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-3xl max-h-[85vh] flex flex-col border border-gray-100">
        {/* Search Input */}
        <div className="p-8 border-b border-gray-200">
          <div className="text-center mb-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Search Training Content</h2>
            <p className="text-gray-600">Find modules, topics, and specific content across all training materials</p>
          </div>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <svg className="h-6 w-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <input
              type="text"
              className="block w-full pl-12 pr-4 py-4 border-2 border-gray-200 rounded-xl leading-5 bg-gray-50 text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#0070AD] focus:border-[#0070AD] focus:bg-white text-lg transition-all duration-200"
              placeholder="Search modules, topics, and content..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              autoFocus
            />
          </div>
        </div>

        {/* Search Results */}
        <div className="flex-1 overflow-y-auto">
          {searchQuery.trim() && searchResults.length === 0 && (
            <div className="p-8 text-center text-gray-500">
              <svg className="mx-auto h-12 w-12 text-gray-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.5-.902-6.122-2.377M15 17.085A7.962 7.962 0 0112 15c-2.34 0-4.5-.902-6.122-2.377" />
              </svg>
              <p className="text-lg font-medium">No results found</p>
              <p className="text-sm">Try adjusting your search terms</p>
            </div>
          )}

          {searchResults.length > 0 && (
            <div className="py-2">
              {searchResults.map((result, index) => (
                <button
                  key={`${result.type}-${result.moduleId}-${index}`}
                  className={`w-full text-left px-6 py-4 hover:bg-gray-50 transition-colors duration-150 ${
                    index === selectedIndex ? 'bg-[#F0F8FC] border-r-4 border-[#0070AD]' : ''
                  }`}
                  onClick={() => handleSelectResult(result)}
                >
                  <div className="flex items-start space-x-3">
                    <span className="text-2xl flex-shrink-0 mt-1">{result.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                          result.type === 'module' ? 'bg-blue-100 text-blue-800' :
                          result.type === 'topic' ? 'bg-green-100 text-green-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {result.type}
                        </span>
                        {result.contentType && (
                          <span className="text-xs px-2 py-1 rounded-full bg-purple-100 text-purple-800 font-medium">
                            {result.contentType}
                          </span>
                        )}
                        {result.moduleTitle && (
                          <span className="text-xs text-gray-500">
                            in {result.moduleTitle}
                          </span>
                        )}
                      </div>
                      <div 
                        className="font-medium text-gray-900 line-clamp-2"
                        dangerouslySetInnerHTML={{ 
                          __html: highlightText(result.title, searchQuery) 
                        }}
                      />
                      {result.type === 'module' && (
                        <p className="text-sm text-gray-500 mt-1 line-clamp-2">
                          {result.description}
                        </p>
                      )}
                      {result.type === 'content' && (
                        <p className="text-xs text-gray-400 mt-1">
                          Line {result.lineNumber + 1} • Click to scroll to this content
                        </p>
                      )}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-8 py-4 border-t border-gray-200 bg-gray-50 rounded-b-xl">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-1">
                <kbd className="px-2 py-1 bg-white rounded border text-xs">↑↓</kbd>
                <span>Navigate</span>
              </div>
              <div className="flex items-center space-x-1">
                <kbd className="px-2 py-1 bg-white rounded border text-xs">↵</kbd>
                <span>Select</span>
              </div>
              <div className="flex items-center space-x-1">
                <kbd className="px-2 py-1 bg-white rounded border text-xs">Esc</kbd>
                <span>Close</span>
              </div>
            </div>
            {searchResults.length > 0 && (
              <span className="font-medium text-[#0070AD]">{searchResults.length} result{searchResults.length !== 1 ? 's' : ''}</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SearchModal;
