import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import WeightsAdmin from './pages/WeightsAdmin';
import Documentation from './pages/Documentation';

function App() {
  return (
    <div className="min-h-screen flex flex-col bg-[#faf9f6] text-stone-800 font-sans">
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/developer/weights" element={<WeightsAdmin />} />
        <Route path="/documentation" element={<Documentation />} />
      </Routes>
      <Footer />
    </div>
  );
}

export default App;
