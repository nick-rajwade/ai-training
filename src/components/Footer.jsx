import React from 'react';
import CapgeminiLogo from './CapgeminiLogo';

const Footer = () => {
  return (
    <footer className="bg-white border-t border-gray-200 mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center mb-4 md:mb-0">
            <CapgeminiLogo className="mr-3" width="32" height="30" />
            <span className="text-lg font-bold bg-gradient-to-r from-[#0070AD] to-[#12239E] bg-clip-text text-transparent">
              AI Training Platform
            </span>
          </div>
          
          <div className="text-sm text-gray-600">
            © {new Date().getFullYear()} Capgemini. All rights reserved.
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
