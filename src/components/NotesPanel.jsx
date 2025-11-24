import React, { useState, useEffect } from 'react';
import { useBookmarks } from '../contexts/BookmarkContext';

const NotesPanel = ({ moduleId, isOpen, onClose }) => {
  const { getNote, addNote } = useBookmarks();
  const [noteContent, setNoteContent] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setNoteContent(getNote(moduleId));
    }
  }, [isOpen, moduleId, getNote]);

  const handleSave = async () => {
    setIsSaving(true);
    await new Promise(resolve => setTimeout(resolve, 300)); // Simulate save delay
    addNote(moduleId, noteContent);
    setIsSaving(false);
  };

  const handleAutoSave = () => {
    addNote(moduleId, noteContent);
  };

  // Auto-save after 2 seconds of inactivity
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (noteContent !== getNote(moduleId)) {
        handleAutoSave();
      }
    }, 2000);

    return () => clearTimeout(timeoutId);
  }, [noteContent, moduleId, getNote]);

  if (!isOpen) return null;

  return (
    <div className="fixed right-0 top-0 h-full w-96 bg-white shadow-2xl z-40 border-l border-gray-200 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-gray-200">
        <div className="flex items-center space-x-2">
          <svg className="w-5 h-5 text-[#0070AD]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
          </svg>
          <h3 className="text-lg font-semibold text-gray-900">Module Notes</h3>
        </div>
        <button
          onClick={onClose}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          aria-label="Close notes"
        >
          <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 flex flex-col p-6">
        <textarea
          value={noteContent}
          onChange={(e) => setNoteContent(e.target.value)}
          placeholder="Take notes about this module..."
          className="flex-1 w-full p-4 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-[#0070AD] focus:border-transparent bg-white text-gray-900 placeholder-gray-500"
        />
        
        <div className="mt-4 flex items-center justify-between">
          <div className="text-sm text-gray-500">
            {isSaving ? 'Saving...' : 'Auto-saved'}
          </div>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="bg-[#0070AD] text-white px-4 py-2 rounded-lg hover:bg-[#005A8C] transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSaving ? 'Saving...' : 'Save Notes'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default NotesPanel;
