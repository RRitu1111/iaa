/**
 * Backend Test Component
 * 
 * This component tests the connection to the IAA backend
 * and provides a simple interface to test authentication.
 */

import React, { useState, useEffect } from 'react';
import apiService from '../api/apiService';
import authService from '../api/authService';
import { useAuth } from '../contexts/AuthContext';

const BackendTest = () => {
  const [healthStatus, setHealthStatus] = useState(null);
  const [dbInfo, setDbInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [testResults, setTestResults] = useState([]);
  const [loginForm, setLoginForm] = useState({
    email: '',
    password: ''
  });
  const [registerForm, setRegisterForm] = useState({
    email: '',
    password: '',
    firstName: '',
    lastName: '',
    role: 'trainee',
    departmentId: null
  });

  const { user, login, logout, isAuthenticated } = useAuth();

  // Test backend health on component mount
  useEffect(() => {
    testBackendHealth();
  }, []);

  const addTestResult = (test, success, message, data = null) => {
    const result = {
      id: Date.now(),
      test,
      success,
      message,
      data,
      timestamp: new Date().toLocaleTimeString()
    };
    setTestResults(prev => [result, ...prev.slice(0, 9)]); // Keep last 10 results
  };

  const testBackendHealth = async () => {
    setLoading(true);
    try {
      const health = await apiService.checkHealth();
      setHealthStatus(health);
      addTestResult('Health Check', health.success !== false, 
        health.message || 'Backend is healthy', health);
      
      const db = await apiService.getDatabaseInfo();
      setDbInfo(db);
      addTestResult('Database Info', db.success !== false, 
        'Database connection verified', db);
    } catch (error) {
      addTestResult('Health Check', false, 'Failed to connect to backend', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const result = await login(loginForm);
      addTestResult('Login Test', result.success, result.message, result.user);
      
      if (result.success) {
        setLoginForm({ email: '', password: '' });
      }
    } catch (error) {
      addTestResult('Login Test', false, 'Login failed', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const result = await authService.register(registerForm);
      addTestResult('Register Test', result.success, result.message, result.user);
      
      if (result.success) {
        setRegisterForm({
          email: '',
          password: '',
          firstName: '',
          lastName: '',
          role: 'trainee',
          departmentId: null
        });
      }
    } catch (error) {
      addTestResult('Register Test', false, 'Registration failed', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    addTestResult('Logout', true, 'Successfully logged out');
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-white">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          🔧 IAA Backend Connection Test
        </h1>
        <p className="text-gray-600">
          Test the connection between React frontend and Rust backend
        </p>
      </div>

      {/* Backend Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
          <h2 className="text-xl font-semibold text-blue-800 mb-4">
            🏥 Backend Health
          </h2>
          {healthStatus ? (
            <div className="space-y-2">
              <div className={`flex items-center ${healthStatus.success !== false ? 'text-green-600' : 'text-red-600'}`}>
                <span className="mr-2">
                  {healthStatus.success !== false ? '✅' : '❌'}
                </span>
                Status: {healthStatus.status || 'Connected'}
              </div>
              <div className="text-sm text-gray-600">
                Service: {healthStatus.service || 'IAA Backend API'}
              </div>
              <div className="text-sm text-gray-600">
                Database: {healthStatus.database || 'Connected'}
              </div>
            </div>
          ) : (
            <div className="text-gray-500">Loading...</div>
          )}
        </div>

        <div className="bg-green-50 p-6 rounded-lg border border-green-200">
          <h2 className="text-xl font-semibold text-green-800 mb-4">
            🗄️ Database Info
          </h2>
          {dbInfo ? (
            <div className="space-y-2">
              <div className="text-sm text-gray-600">
                Database: {dbInfo.database || 'PostgreSQL'}
              </div>
              <div className="text-sm text-gray-600">
                Provider: {dbInfo.provider || 'Supabase'}
              </div>
              <div className="text-sm text-gray-600">
                Host: {dbInfo.host || 'Connected'}
              </div>
            </div>
          ) : (
            <div className="text-gray-500">Loading...</div>
          )}
        </div>
      </div>

      {/* Authentication Status */}
      <div className="bg-yellow-50 p-6 rounded-lg border border-yellow-200 mb-8">
        <h2 className="text-xl font-semibold text-yellow-800 mb-4">
          🔐 Authentication Status
        </h2>
        {isAuthenticated && user ? (
          <div className="space-y-2">
            <div className="flex items-center text-green-600">
              <span className="mr-2">✅</span>
              Logged in as: {user.first_name} {user.last_name}
            </div>
            <div className="text-sm text-gray-600">
              Email: {user.email}
            </div>
            <div className="text-sm text-gray-600">
              Role: {user.role}
            </div>
            <div className="text-sm text-gray-600">
              Department ID: {user.department_id || 'None'}
            </div>
            <button
              onClick={handleLogout}
              className="mt-2 px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
            >
              Logout
            </button>
          </div>
        ) : (
          <div className="text-gray-500">Not authenticated</div>
        )}
      </div>

      {/* Test Forms */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* Login Test */}
        <div className="bg-gray-50 p-6 rounded-lg border">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            🔑 Login Test
          </h3>
          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Email
              </label>
              <input
                type="email"
                value={loginForm.email}
                onChange={(e) => setLoginForm({...loginForm, email: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="user@iaa.edu.in"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Password
              </label>
              <input
                type="password"
                value={loginForm.password}
                onChange={(e) => setLoginForm({...loginForm, password: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="password"
                required
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50"
            >
              {loading ? 'Testing...' : 'Test Login'}
            </button>
          </form>
        </div>

        {/* Register Test */}
        <div className="bg-gray-50 p-6 rounded-lg border">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            📝 Register Test
          </h3>
          <form onSubmit={handleRegister} className="space-y-4">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  First Name
                </label>
                <input
                  type="text"
                  value={registerForm.firstName}
                  onChange={(e) => setRegisterForm({...registerForm, firstName: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Last Name
                </label>
                <input
                  type="text"
                  value={registerForm.lastName}
                  onChange={(e) => setRegisterForm({...registerForm, lastName: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  required
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Email
              </label>
              <input
                type="email"
                value={registerForm.email}
                onChange={(e) => setRegisterForm({...registerForm, email: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Password
              </label>
              <input
                type="password"
                value={registerForm.password}
                onChange={(e) => setRegisterForm({...registerForm, password: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Role
              </label>
              <select
                value={registerForm.role}
                onChange={(e) => setRegisterForm({...registerForm, role: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="trainee">Trainee</option>
                <option value="trainer">Trainer</option>
                <option value="admin">Admin</option>
              </select>
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50"
            >
              {loading ? 'Testing...' : 'Test Register'}
            </button>
          </form>
        </div>
      </div>

      {/* Test Results */}
      <div className="bg-gray-50 p-6 rounded-lg border">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-800">
            📊 Test Results
          </h3>
          <button
            onClick={testBackendHealth}
            disabled={loading}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
          >
            {loading ? 'Testing...' : 'Refresh Tests'}
          </button>
        </div>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {testResults.length > 0 ? (
            testResults.map((result) => (
              <div
                key={result.id}
                className={`p-3 rounded border-l-4 ${
                  result.success 
                    ? 'bg-green-50 border-green-400' 
                    : 'bg-red-50 border-red-400'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <div className="font-medium">
                      {result.success ? '✅' : '❌'} {result.test}
                    </div>
                    <div className="text-sm text-gray-600">
                      {result.message}
                    </div>
                  </div>
                  <div className="text-xs text-gray-500">
                    {result.timestamp}
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="text-gray-500 text-center py-4">
              No test results yet
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BackendTest;
