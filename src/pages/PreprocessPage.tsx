import React, { useState } from 'react';
import { Settings, Check } from 'lucide-react';
import toast from 'react-hot-toast';
import axios from 'axios';

interface PreprocessingOptions {
  handleMissing: 'drop' | 'mean' | 'median' | 'mode';
  filterColumn?: string;
  filterValue?: string;
  transformations: string[];
}

const PreprocessPage = () => {
  const [options, setOptions] = useState<PreprocessingOptions>({
    handleMissing: 'drop',
    transformations: [],
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8080/preprocess', options);
      toast.success('Data preprocessed successfully!');
      // Handle the preprocessed data
    } catch (error) {
      console.error('Preprocessing error:', error);
      toast.error('Failed to preprocess data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Data Preprocessing
        </h2>
        <p className="text-gray-600">
          Configure preprocessing options for your dataset.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Missing Values */}
        <div className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Handle Missing Values
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {['drop', 'mean', 'median', 'mode'].map((method) => (
              <label
                key={method}
                className={`flex items-center p-4 border rounded-lg cursor-pointer transition-colors ${
                  options.handleMissing === method
                    ? 'border-purple-500 bg-purple-50'
                    : 'border-gray-200 hover:border-purple-200'
                }`}
              >
                <input
                  type="radio"
                  name="handleMissing"
                  value={method}
                  checked={options.handleMissing === method}
                  onChange={(e) =>
                    setOptions({ ...options, handleMissing: e.target.value as any })
                  }
                  className="hidden"
                />
                <div className="flex items-center justify-between w-full">
                  <span className="font-medium text-gray-700 capitalize">
                    {method}
                  </span>
                  {options.handleMissing === method && (
                    <Check size={20} className="text-purple-500" />
                  )}
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Transformations */}
        <div className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Data Transformations
          </h3>
          <div className="space-y-3">
            {['normalize', 'standardize', 'log', 'square_root'].map((transform) => (
              <label
                key={transform}
                className="flex items-center space-x-3 p-3 border rounded-lg cursor-pointer hover:bg-gray-50"
              >
                <input
                  type="checkbox"
                  checked={options.transformations.includes(transform)}
                  onChange={(e) => {
                    const newTransformations = e.target.checked
                      ? [...options.transformations, transform]
                      : options.transformations.filter((t) => t !== transform);
                    setOptions({ ...options, transformations: newTransformations });
                  }}
                  className="h-4 w-4 text-purple-600 rounded border-gray-300 focus:ring-purple-500"
                />
                <span className="font-medium text-gray-700 capitalize">
                  {transform.replace('_', ' ')}
                </span>
              </label>
            ))}
          </div>
        </div>

        {/* Filter Options */}
        <div className="bg-white p-6 rounded-lg shadow-sm">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Filter Data
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Column
              </label>
              <input
                type="text"
                value={options.filterColumn || ''}
                onChange={(e) =>
                  setOptions({ ...options, filterColumn: e.target.value })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="Enter column name"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Value
              </label>
              <input
                type="text"
                value={options.filterValue || ''}
                onChange={(e) =>
                  setOptions({ ...options, filterValue: e.target.value })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="Enter filter value"
              />
            </div>
          </div>
        </div>

        <div className="flex justify-end">
          <button
            type="submit"
            disabled={loading}
            className="flex items-center space-x-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                <span>Processing...</span>
              </>
            ) : (
              <>
                <Settings size={20} />
                <span>Apply Preprocessing</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default PreprocessPage;