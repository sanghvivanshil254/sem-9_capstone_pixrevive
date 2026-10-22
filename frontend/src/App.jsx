
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';

import Navbar from './components/Navbar';
import Footer from './components/Footer';

import Home from './pages/Home';
import Documentation from './pages/Documentation';
import DevWeights from './pages/DevWeights';

import PublicHome from './pages/PublicHome';
import { Login, Register } from './pages/AuthPages';
import AppHome from './pages/AppHome';
import Dashboard from './pages/Dashboard';
import Profile from './pages/Profile';

import './styles/account-pages.css';

// Protect pages that require a logged-in user
function RequireAccount({ children }) {
  const location = useLocation();
  const user = localStorage.getItem('pixrevive-user');

  return user ? (
    children
  ) : (
    <Navigate
      to="/login"
      replace
      state={{ from: location.pathname }}
    />
  );
}

function App() {
  return (
    <Routes>
      {/* Public home page */}
      <Route
        path="/"
        element={<PublicHome />}
      />

      {/* Authentication pages */}
      <Route
        path="/login"
        element={<Login />}
      />
      <Route
        path="/register"
        element={<Register />}
      />

      {/* Protected account pages */}
      <Route
        path="/app"
        element={
          <RequireAccount>
            <AppHome />
          </RequireAccount>
        }
      />

      <Route
        path="/dashboard"
        element={
          <RequireAccount>
            <Dashboard />
          </RequireAccount>
        }
      />

      <Route
        path="/profile"
        element={
          <RequireAccount>
            <Profile />
          </RequireAccount>
        }
      />

      {/* Existing pages — original layout preserved */}
      <Route
        path="/workspace"
        element={
          <div className="min-h-screen flex flex-col bg-[#faf9f6] text-stone-800 font-sans selection:bg-indigo-100 selection:text-indigo-900 w-full overflow-x-hidden">
            <Navbar />
            <div className="flex-1 flex flex-col w-full max-w-[1920px] mx-auto">
              <Home />
            </div>
            <Footer />
          </div>
        }
      />

      <Route
        path="/documentation"
        element={
          <div className="min-h-screen flex flex-col bg-[#faf9f6] text-stone-800 font-sans w-full overflow-x-hidden">
            <Navbar />
            <div className="flex-1 flex flex-col w-full max-w-[1920px] mx-auto">
              <Documentation />
            </div>
            <Footer />
          </div>
        }
      />

      <Route
        path="/developer/weights"
        element={
          <div className="min-h-screen flex flex-col bg-[#faf9f6] text-stone-800 font-sans w-full overflow-x-hidden">
            <Navbar />
            <div className="flex-1 flex flex-col w-full max-w-[1920px] mx-auto">
              <DevWeights />
            </div>
            <Footer />
          </div>
        }
      />

      {/* Fallback */}
      <Route
        path="*"
        element={<Navigate to="/" replace />}
      />
    </Routes>
  );
}

export default App;
