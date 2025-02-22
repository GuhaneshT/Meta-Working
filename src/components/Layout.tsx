import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { Upload, Settings, BarChart2 } from 'lucide-react';

const Layout = () => {
  const navItems = [
    { path: '/upload', icon: Upload, label: 'Upload Dataset' },
    { path: '/preprocess', icon: Settings, label: 'Preprocess' },
    { path: '/visualize', icon: BarChart2, label: 'Visualize' },
  ];

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-gray-200">
        <div className="p-6">
          <h1 className="text-2xl font-bold text-gray-800">Data Viz</h1>
        </div>
        <nav className="px-4">
          {navItems.map(({ path, icon: Icon, label }) => (
            <NavLink
              key={path}
              to={path}
              className={({ isActive }) =>
                `flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                  isActive
                    ? 'bg-purple-50 text-purple-700'
                    : 'text-gray-600 hover:bg-gray-50'
                }`
              }
            >
              <Icon size={20} />
              <span className="font-medium">{label}</span>
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;