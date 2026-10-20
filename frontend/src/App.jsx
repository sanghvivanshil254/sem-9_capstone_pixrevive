import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Documentation from './pages/Documentation';
import DevWeights from './pages/DevWeights';

function App() {
  return (
    <div className="min-h-screen min-h-[100dvh] flex flex-col bg-[#faf9f6] text-stone-800 font-sans selection:bg-indigo-100 selection:text-indigo-900 w-full overflow-x-hidden">
      <Navbar />
      <div className="flex-1 flex flex-col w-full max-w-[1920px] mx-auto">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/developer/weights" element={<DevWeights />} />
          <Route path="/documentation" element={<Documentation />} />
        </Routes>
      </div>
      <Footer />
    </div>
  );
}

export default App;
