import React, { useState, useEffect } from 'react';
import {
  BarChart2,
  LineChart,
  PieChart,
  Activity,
  RefreshCw,
} from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line, Bar, Pie } from 'react-chartjs-2';
import toast from 'react-hot-toast';
import axios from 'axios';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

type ChartType = 'line' | 'bar' | 'pie' | 'scatter';

interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string;
    borderWidth?: number;
  }[];
}

const VisualizePage = () => {
  const [chartType, setChartType] = useState<ChartType>('bar');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ChartData | null>(null);
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:5000/visualize', {
        params: { columns: selectedColumns, type: chartType },
      });
      setData(response.data);
    } catch (error) {
      console.error('Visualization error:', error);
      toast.error('Failed to fetch visualization data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const fetchColumns = async () => {
      try {
        const response = await axios.get('http://localhost:5000/columns');
        setAvailableColumns(response.data);
      } catch (error) {
        console.error('Error fetching columns:', error);
        toast.error('Failed to fetch available columns');
      }
    };

    fetchColumns();
  }, []);

  const chartTypes = [
    { type: 'bar', icon: BarChart2, label: 'Bar Chart' },
    { type: 'line', icon: LineChart, label: 'Line Chart' },
    { type: 'pie', icon: PieChart, label: 'Pie Chart' },
    { type: 'scatter', icon: Activity, label: 'Scatter Plot' },
  ];

  const renderChart = () => {
    if (!data) return null;

    const chartProps = {
      data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top' as const,
          },
          title: {
            display: true,
            text: 'Data Visualization',
          },
        },
      },
    };

    switch (chartType) {
      case 'line':
        return <Line {...chartProps} />;
      case 'bar':
        return <Bar {...chartProps} />;
      case 'pie':
        return <Pie {...chartProps} />;
      default:
        return null;
    }
  };

  return (
    <div className="p-6">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Data Visualization
        </h2>
        <p className="text-gray-600">
          Create interactive visualizations from your processed data.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar Controls */}
        <div className="space-y-6">
          {/* Chart Type Selection */}
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              Chart Type
            </h3>
            <div className="space-y-2">
              {chartTypes.map(({ type, icon: Icon, label }) => (
                <button
                  key={type}
                  onClick={() => setChartType(type as ChartType)}
                  className={`flex items-center space-x-3 w-full p-3 rounded-lg transition-colors ${
                    chartType === type
                      ? 'bg-purple-50 text-purple-700'
                      : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  <Icon size={20} />
                  <span className="font-medium">{label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Column Selection */}
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              Select Columns
            </h3>
            <div className="space-y-2">
              {availableColumns.map((column) => (
                <label
                  key={column}
                  className="flex items-center space-x-3 p-3 border rounded-lg cursor-pointer hover:bg-gray-50"
                >
                  <input
                    type="checkbox"
                    checked={selectedColumns.includes(column)}
                    onChange={(e) => {
                      const newColumns = e.target.checked
                        ? [...selectedColumns, column]
                        : selectedColumns.filter((c) => c !== column);
                      setSelectedColumns(newColumns);
                    }}
                    className="h-4 w-4 text-purple-600 rounded border-gray-300 focus:ring-purple-500"
                  />
                  <span className="font-medium text-gray-700">{column}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Update Button */}
          <button
            onClick={fetchData}
            disabled={loading || selectedColumns.length === 0}
            className="w-full flex items-center justify-center space-x-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                <span>Loading...</span>
              </>
            ) : (
              <>
                <RefreshCw size={20} />
                <span>Update Visualization</span>
              </>
            )}
          </button>
        </div>

        {/* Chart Area */}
        <div className="lg:col-span-3 bg-white p-6 rounded-lg shadow-sm">
          <div className="h-[600px] flex items-center justify-center">
            {loading ? (
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto"></div>
                <p className="mt-4 text-gray-600">Loading visualization...</p>
              </div>
            ) : data ? (
              renderChart()
            ) : (
              <div className="text-center text-gray-500">
                <BarChart2 size={48} className="mx-auto mb-4 opacity-50" />
                <p>Select columns and click Update to visualize your data</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VisualizePage;