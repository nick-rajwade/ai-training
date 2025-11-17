import React, { createContext, useContext, useState, useEffect } from 'react';

const BookmarkContext = createContext();

export const useBookmarks = () => {
  const context = useContext(BookmarkContext);
  if (!context) {
    throw new Error('useBookmarks must be used within a BookmarkProvider');
  }
  return context;
};

export const BookmarkProvider = ({ children }) => {
  const [bookmarks, setBookmarks] = useState(() => {
    const saved = localStorage.getItem('ai-training-bookmarks');
    return saved ? JSON.parse(saved) : [];
  });
  
  const [notes, setNotes] = useState(() => {
    const saved = localStorage.getItem('ai-training-notes');
    return saved ? JSON.parse(saved) : {};
  });

  useEffect(() => {
    localStorage.setItem('ai-training-bookmarks', JSON.stringify(bookmarks));
  }, [bookmarks]);

  useEffect(() => {
    localStorage.setItem('ai-training-notes', JSON.stringify(notes));
  }, [notes]);

  const addBookmark = (moduleId, section = null) => {
    const bookmark = {
      id: Date.now(),
      moduleId,
      section,
      timestamp: new Date().toISOString(),
    };
    setBookmarks(prev => [...prev, bookmark]);
  };

  const removeBookmark = (bookmarkId) => {
    setBookmarks(prev => prev.filter(b => b.id !== bookmarkId));
  };

  const isBookmarked = (moduleId, section = null) => {
    return bookmarks.some(b => 
      b.moduleId === moduleId && 
      (section ? b.section === section : !b.section)
    );
  };

  const addNote = (moduleId, content) => {
    setNotes(prev => ({
      ...prev,
      [moduleId]: content
    }));
  };

  const getNote = (moduleId) => {
    return notes[moduleId] || '';
  };

  return (
    <BookmarkContext.Provider value={{
      bookmarks,
      notes,
      addBookmark,
      removeBookmark,
      isBookmarked,
      addNote,
      getNote
    }}>
      {children}
    </BookmarkContext.Provider>
  );
};
