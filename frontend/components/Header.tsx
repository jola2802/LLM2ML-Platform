
import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="bg-gray-800 shadow-md">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <img 
              src="/idpm.png" 
              alt="IDPM Logo" 
              className="h-8 w-8 object-contain"
            />
            <h1 className="text-2xl font-bold text-white ml-3">No-Code ML Platform</h1>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
