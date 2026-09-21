import React, { useState, useCallback, useEffect } from 'react';
import axios from 'axios';
import {
  UploadCloud, CheckCircle, AlertCircle, Cpu, ArrowLeft,
  TerminalSquare, Plus, Trash2,
  Image as ImageIcon, ChevronLeft, ChevronRight, FileArchive, RefreshCw, Search
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import './WeightsAdmin.css';

export default function WeightsAdmin() {
  const [restorerStatus, setRestorerStatus] = useState('idle');
  const [zipStatus, setZipStatus] = useState('idle');
  const [msg, setMsg] = useState('');
  const navigate = useNavigate();

  // Training Data State
  const [pairs, setPairs] = useState([]);
  const [totalPairs, setTotalPairs] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newBwFile, setNewBwFile] = useState(null);
  const [newColorFile, setNewColorFile] = useState(null);
  const [addStatus, setAddStatus] = useState('idle');

  // New features state
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIds, setSelectedIds] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState('');

  // Weights status & Live training pipeline states
  const [weightsStatus, setWeightsStatus] = useState(null);
  const [trainingState, setTrainingState] = useState({
    status: 'idle',
    progress: 0,
    message: 'No active training pipeline.',
    epochs: [],
    results: null
  });
  const [isTraining, setIsTraining] = useState(false);
  const [terminalLogs, setTerminalLogs] = useState([]);

  // Fetch weights active status from backend
  const fetchWeightsStatus = useCallback(async () => {
    try {
      const res = await axios.get('http://localhost:8000/api/admin/weights/status');
      setWeightsStatus(res.data);
    } catch (err) {
      console.error("Failed to fetch weights status:", err);
    }
  }, []);

  // Fetch training status and update logs
  const fetchTrainingStatus = useCallback(async () => {
    try {
      const res = await axios.get('http://localhost:8000/api/admin/training/status');
      const data = res.data;
      setTrainingState(data);
      
      // Build terminal log messages based on trainingState
      const logs = [];
      if (data.status !== 'idle') {
        logs.push(`[SYSTEM] Initializing training pipeline...`);
        logs.push(`[SYSTEM] ${data.message}`);
        
        if (data.epochs && data.epochs.length > 0) {
          logs.push(`[DATA] Classification complete. Training data split: 80% train, 20% test.`);
          data.epochs.forEach(e => {
            logs.push(`[EPOCH ${e.epoch}/10] Restorer Loss: ${e.restorer_loss.toFixed(4)} | Colorizer Loss: ${e.colorizer_loss.toFixed(4)} | Restorer Acc: ${e.restorer_accuracy.toFixed(1)}% | Colorizer F1: ${e.colorizer_f1.toFixed(3)} | Combined Acc: ${e.combined_accuracy.toFixed(1)}%`);
          });
        }
        
        if (data.status === 'completed' && data.results) {
          logs.push(`[SUCCESS] Training pipeline completed successfully!`);
          logs.push(`[RESULTS] -----------------------------------------`);
          logs.push(`[RESULTS] Restorer Accuracy: ${data.results.restorer_accuracy}% (PSNR: ${data.results.restorer_psnr} dB)`);
          logs.push(`[RESULTS] Colorizer Accuracy: ${data.results.colorizer_accuracy}% (F1-score: ${data.results.colorizer_f1})`);
          logs.push(`[RESULTS] Pipeline Combined Accuracy: ${data.results.combined_accuracy}%`);
          logs.push(`[RESULTS] -----------------------------------------`);
        } else if (data.status === 'failed') {
          logs.push(`[ERROR] Training pipeline failed: ${data.message}`);
        }
      }
      setTerminalLogs(logs);

      if (data.status === 'running') {
        setIsTraining(true);
        return true;
      } else {
        setIsTraining(false);
        // Refresh weights status if training just finished to reflect new models
        if (data.status === 'completed') {
          fetchWeightsStatus();
        }
        return false;
      }
    } catch (err) {
      console.error("Failed to fetch training status:", err);
      return false;
    }
  }, [fetchWeightsStatus]);

  // Initial loads on mount
  useEffect(() => {
    fetchWeightsStatus();
    fetchTrainingStatus();
  }, [fetchWeightsStatus, fetchTrainingStatus]);

  // Poll training status when running
  useEffect(() => {
    let intervalId;
    if (isTraining) {
      intervalId = setInterval(async () => {
        const shouldContinue = await fetchTrainingStatus();
        if (!shouldContinue) {
          clearInterval(intervalId);
        }
      }, 1000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isTraining, fetchTrainingStatus]);

  const handleStartTraining = async () => {
    if (totalPairs === 0) {
      alert("Cannot start training. Dataset is empty. Please upload some training data pairs first!");
      return;
    }
    
    try {
      setMsg('');
      await axios.post('http://localhost:8000/api/admin/training/start');
      setIsTraining(true);
      fetchTrainingStatus();
    } catch (err) {
      console.error("Failed to start training:", err);
      setMsg(err.response?.data?.detail || "Failed to start training pipeline.");
    }
  };

  const fetchPairs = useCallback(async (page, query = '') => {
    try {
      const res = await axios.get(`http://localhost:8000/api/admin/training-data/pairs?page=${page}&limit=6&q=${encodeURIComponent(query)}`);
      setPairs(res.data.pairs);
      setTotalPages(res.data.total_pages);
      setTotalPairs(res.data.total);
    } catch (err) {
      console.error("Failed to fetch training pairs:", err);
    }
  }, []);

  // Search input debouncer
  useEffect(() => {
    const delayDebounceFn = setTimeout(() => {
      setCurrentPage(1);
      fetchPairs(1, searchQuery);
    }, 350);
    return () => clearTimeout(delayDebounceFn);
  }, [searchQuery, fetchPairs]);

  const handleDrop = useCallback(async (e, type) => {
    e.preventDefault();
    const files = e.dataTransfer ? e.dataTransfer.files : e.target.files;
    if (!files || files.length === 0) return;
    const file = files[0];
    
    if (!file.name.endsWith('.pth')) {
      setMsg('Invalid file format. Please upload a .pth PyTorch weights file.');
      return;
    }

    const setStatus = setRestorerStatus;
    setStatus('uploading');
    setMsg('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', type);

    try {
      const response = await axios.post('http://localhost:8000/api/admin/weights/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setStatus('success');
      setMsg(response.data.message);
      fetchWeightsStatus();
      
      setTimeout(() => {
        setStatus('idle');
      }, 3000);
    } catch (err) {
      console.error(err);
      setStatus('error');
      setMsg(err.response?.data?.detail || 'Failed to upload weights. Ensure the backend is running.');
      
      setTimeout(() => {
        setStatus('idle');
      }, 4000);
    }
  }, [fetchWeightsStatus]);

  const renderLossChart = () => {
    if (!trainingState.epochs || trainingState.epochs.length === 0) {
      return (
        <div className="flex items-center justify-center h-48 text-stone-400 text-sm border border-stone-200 rounded-xl bg-stone-50">
          No training metrics logged yet.
        </div>
      );
    }

    const width = 500;
    const height = 260;
    const paddingLeft = 50;
    const paddingRight = 20;
    const paddingTop = 20;
    const paddingBottom = 40;

    const chartWidth = width - paddingLeft - paddingRight;
    const chartHeight = height - paddingTop - paddingBottom;

    const maxRestorerLoss = Math.max(...trainingState.epochs.map(e => e.restorer_loss));
    const maxColorizerLoss = Math.max(...trainingState.epochs.map(e => e.colorizer_loss));
    const maxLossVal = Math.max(maxRestorerLoss, maxColorizerLoss, 0.16);
    const maxLoss = Math.ceil(maxLossVal * 10) / 10;

    const getX = (index) => paddingLeft + (index / 9) * chartWidth;
    const getY = (val) => height - paddingBottom - (val / maxLoss) * chartHeight;

    let restorerPoints = "";
    let colorizerPoints = "";

    trainingState.epochs.forEach((e, idx) => {
      const x = getX(idx);
      const yRestorer = getY(e.restorer_loss);
      const yColorizer = getY(e.colorizer_loss);

      if (idx === 0) {
        restorerPoints += `M ${x} ${yRestorer}`;
        colorizerPoints += `M ${x} ${yColorizer}`;
      } else {
        restorerPoints += ` L ${x} ${yRestorer}`;
        colorizerPoints += ` L ${x} ${yColorizer}`;
      }
    });

    return (
      <div className="flex flex-col w-full">
        <div className="flex items-center justify-center gap-4 text-xs mb-3 font-semibold">
          <span className="flex items-center gap-1.5 text-indigo-600"><span className="w-2.5 h-2.5 rounded-full bg-indigo-500"></span>Restorer Loss</span>
          <span className="flex items-center gap-1.5 text-amber-500"><span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span>Colorizer Loss</span>
        </div>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto bg-stone-50/50 border border-stone-200 rounded-xl p-2">
          {[0, 0.25, 0.5, 0.75, 1.0].map((ratio, i) => {
            const yVal = ratio * maxLoss;
            const y = getY(yVal);
            return (
              <g key={i} className="opacity-40">
                <line x1={paddingLeft} y1={y} x2={width - paddingRight} y2={y} stroke="#d6d3d1" strokeDasharray="3 3" />
                <text x={paddingLeft - 8} y={y + 4} textAnchor="end" className="text-[10px] font-mono fill-stone-500">{yVal.toFixed(3)}</text>
              </g>
            );
          })}

          {trainingState.epochs.map((e, idx) => {
            const x = getX(idx);
            return (
              <g key={idx}>
                {idx > 0 && idx < 9 && (
                  <line x1={x} y1={paddingTop} x2={x} y2={height - paddingBottom} stroke="#e7e5e4" strokeDasharray="3 3" className="opacity-20" />
                )}
                <text x={x} y={height - paddingBottom + 16} textAnchor="middle" className="text-[10px] font-mono fill-stone-500">E{e.epoch}</text>
              </g>
            );
          })}

          <line x1={paddingLeft} y1={height - paddingBottom} x2={width - paddingRight} y2={height - paddingBottom} stroke="#78716c" strokeWidth="1" />
          <line x1={paddingLeft} y1={paddingTop} x2={paddingLeft} y2={height - paddingBottom} stroke="#78716c" strokeWidth="1" />

          <path d={restorerPoints} fill="none" stroke="#6366f1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={colorizerPoints} fill="none" stroke="#f59e0b" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />

          {trainingState.epochs.map((e, idx) => (
            <g key={idx}>
              <circle cx={getX(idx)} cy={getY(e.restorer_loss)} r="3" fill="#6366f1" stroke="#fff" strokeWidth="1" />
              <circle cx={getX(idx)} cy={getY(e.colorizer_loss)} r="3" fill="#f59e0b" stroke="#fff" strokeWidth="1" />
            </g>
          ))}
        </svg>
      </div>
    );
  };

  const renderAccuracyChart = () => {
    if (!trainingState.epochs || trainingState.epochs.length === 0) {
      return (
        <div className="flex items-center justify-center h-48 text-stone-400 text-sm border border-stone-200 rounded-xl bg-stone-50">
          No training metrics logged yet.
        </div>
      );
    }

    const width = 500;
    const height = 260;
    const paddingLeft = 50;
    const paddingRight = 20;
    const paddingTop = 20;
    const paddingBottom = 40;

    const chartWidth = width - paddingLeft - paddingRight;
    const chartHeight = height - paddingTop - paddingBottom;

    const minY = 50;
    const maxY = 100;

    const getX = (index) => paddingLeft + (index / 9) * chartWidth;
    const getY = (val) => height - paddingBottom - ((val - minY) / (maxY - minY)) * chartHeight;

    let restorerPoints = "";
    let colorizerPoints = "";
    let combinedPoints = "";

    trainingState.epochs.forEach((e, idx) => {
      const x = getX(idx);
      const yRestorer = getY(e.restorer_accuracy);
      const yColorizer = getY(e.colorizer_f1 * 100);
      const yCombined = getY(e.combined_accuracy);

      if (idx === 0) {
        restorerPoints += `M ${x} ${yRestorer}`;
        colorizerPoints += `M ${x} ${yColorizer}`;
        combinedPoints += `M ${x} ${yCombined}`;
      } else {
        restorerPoints += ` L ${x} ${yRestorer}`;
        colorizerPoints += ` L ${x} ${yColorizer}`;
        combinedPoints += ` L ${x} ${yCombined}`;
      }
    });

    return (
      <div className="flex flex-col w-full">
        <div className="flex items-center justify-center gap-4 text-xs mb-3 font-semibold">
          <span className="flex items-center gap-1.5 text-indigo-600"><span className="w-2.5 h-2.5 rounded-full bg-indigo-500"></span>Restorer Acc</span>
          <span className="flex items-center gap-1.5 text-amber-500"><span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span>Colorizer Acc</span>
          <span className="flex items-center gap-1.5 text-emerald-500"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>Combined Acc</span>
        </div>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto bg-stone-50/50 border border-stone-200 rounded-xl p-2">
          {[50, 62.5, 75, 87.5, 100].map((yVal, i) => {
            const y = getY(yVal);
            return (
              <g key={i} className="opacity-40">
                <line x1={paddingLeft} y1={y} x2={width - paddingRight} y2={y} stroke="#d6d3d1" strokeDasharray="3 3" />
                <text x={paddingLeft - 8} y={y + 4} textAnchor="end" className="text-[10px] font-mono fill-stone-500">{yVal.toFixed(0)}%</text>
              </g>
            );
          })}

          {trainingState.epochs.map((e, idx) => {
            const x = getX(idx);
            return (
              <g key={idx}>
                {idx > 0 && idx < 9 && (
                  <line x1={x} y1={paddingTop} x2={x} y2={height - paddingBottom} stroke="#e7e5e4" strokeDasharray="3 3" className="opacity-20" />
                )}
                <text x={x} y={height - paddingBottom + 16} textAnchor="middle" className="text-[10px] font-mono fill-stone-500">E{e.epoch}</text>
              </g>
            );
          })}

          <line x1={paddingLeft} y1={height - paddingBottom} x2={width - paddingRight} y2={height - paddingBottom} stroke="#78716c" strokeWidth="1" />
          <line x1={paddingLeft} y1={paddingTop} x2={paddingLeft} y2={height - paddingBottom} stroke="#78716c" strokeWidth="1" />

          <path d={restorerPoints} fill="none" stroke="#6366f1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={colorizerPoints} fill="none" stroke="#f59e0b" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={combinedPoints} fill="none" stroke="#10b981" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />

          {trainingState.epochs.map((e, idx) => (
            <g key={idx}>
              <circle cx={getX(idx)} cy={getY(e.restorer_accuracy)} r="3" fill="#6366f1" stroke="#fff" strokeWidth="1" />
              <circle cx={getX(idx)} cy={getY(e.colorizer_f1 * 100)} r="3" fill="#f59e0b" stroke="#fff" strokeWidth="1" />
              <circle cx={getX(idx)} cy={getY(e.combined_accuracy)} r="3" fill="#10b981" stroke="#fff" strokeWidth="1" />
            </g>
          ))}
        </svg>
      </div>
    );
  };

  const startPollingProgress = useCallback((id) => {
    setIsProcessing(true);
    setUploadProgress(5);
    setProgressMsg('Initiating upload analysis...');
    
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`http://localhost:8000/api/admin/training-data/upload-progress?upload_id=${id}`);
        const { status, progress, message } = res.data;
        
        setUploadProgress(progress);
        setProgressMsg(message);
        
        if (status === 'completed') {
          clearInterval(interval);
          setMsg(message);
          setZipStatus('success');
          setIsProcessing(false);
          setSelectedIds([]);
          setCurrentPage(1);
          fetchPairs(1, searchQuery);
          setTimeout(() => setZipStatus('idle'), 4000);
        } else if (status === 'failed') {
          clearInterval(interval);
          setMsg(message);
          setZipStatus('error');
          setIsProcessing(false);
          setTimeout(() => setZipStatus('idle'), 5000);
        }
      } catch (err) {
        console.error("Polling progress error:", err);
        clearInterval(interval);
        setZipStatus('error');
        setMsg('Lost connection to progress tracker during extraction.');
        setIsProcessing(false);
        setTimeout(() => setZipStatus('idle'), 4000);
      }
    }, 450);
  }, [fetchPairs, searchQuery]);

  const handleZipDrop = useCallback(async (e) => {
    e.preventDefault();
    const files = e.dataTransfer ? e.dataTransfer.files : e.target.files;
    if (!files || files.length === 0) return;
    const file = files[0];
    
    if (!file.name.endsWith('.zip')) {
      setMsg('Invalid file format. Please upload a .zip image dataset archive.');
      return;
    }

    setZipStatus('uploading');
    setMsg('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/admin/training-data/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      const { upload_id } = response.data;
      startPollingProgress(upload_id);
    } catch (err) {
      console.error(err);
      setZipStatus('error');
      setMsg(err.response?.data?.detail || 'Failed to upload training dataset ZIP.');
      setTimeout(() => {
        setZipStatus('idle');
      }, 4000);
    }
  }, [startPollingProgress]);

  const handleAddPair = async (e) => {
    e.preventDefault();
    if (!newBwFile || !newColorFile) {
      setMsg('Please select both a Black & White image and a Color target image.');
      return;
    }

    setAddStatus('submitting');
    setMsg('');

    const formData = new FormData();
    formData.append('bw_file', newBwFile);
    formData.append('color_file', newColorFile);

    try {
      await axios.post('http://localhost:8000/api/admin/training-data/pairs', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setAddStatus('success');
      setMsg('Successfully added a new training pair!');
      setNewBwFile(null);
      setNewColorFile(null);
      setShowAddForm(false);
      fetchPairs(currentPage, searchQuery);
      
      setTimeout(() => {
        setAddStatus('idle');
      }, 3000);
    } catch (err) {
      console.error(err);
      setAddStatus('error');
      setMsg(err.response?.data?.detail || 'Failed to create new image pair.');
      
      setTimeout(() => {
        setAddStatus('idle');
      }, 4000);
    }
  };

  const handleUpdateImage = async (pairId, type, file) => {
    if (!file) return;

    const formData = new FormData();
    if (type === 'bw') {
      formData.append('bw_file', file);
    } else {
      formData.append('color_file', file);
    }

    try {
      setMsg(`Updating ${type === 'bw' ? 'Black & White' : 'Color'} image...`);
      await axios.put(`http://localhost:8000/api/admin/training-data/pairs/${pairId}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setMsg(`Successfully updated ${type === 'bw' ? 'B&W' : 'Color'} image.`);
      fetchPairs(currentPage, searchQuery);
    } catch (err) {
      console.error(err);
      setMsg(err.response?.data?.detail || `Failed to update ${type === 'bw' ? 'B&W' : 'Color'} image.`);
    }
  };

  const handleDeletePair = async (pairId) => {
    if (!window.confirm('Are you sure you want to permanently delete this training image pair?')) {
      return;
    }

    try {
      await axios.delete(`http://localhost:8000/api/admin/training-data/pairs/${pairId}`);
      setMsg('Successfully deleted training pair.');
      
      setSelectedIds(prev => prev.filter(id => id !== pairId));
      
      // Handle page decrement if last element on last page was deleted
      if (pairs.length === 1 && currentPage > 1) {
        setCurrentPage(prev => prev - 1);
      } else {
        fetchPairs(currentPage, searchQuery);
      }
    } catch (err) {
      console.error(err);
      setMsg(err.response?.data?.detail || 'Failed to delete image pair.');
    }
  };

  const handleBatchDelete = async () => {
    if (!window.confirm(`Are you sure you want to permanently delete the ${selectedIds.length} selected image pairs?`)) {
      return;
    }
    
    try {
      setMsg(`Deleting ${selectedIds.length} pairs...`);
      await axios.post('http://localhost:8000/api/admin/training-data/pairs/batch-delete', { ids: selectedIds });
      setMsg(`Successfully deleted ${selectedIds.length} image pairs.`);
      setSelectedIds([]);
      
      // Reset page calculation if needed
      const remainingTotal = totalPairs - selectedIds.length;
      const expectedTotalPages = Math.max(1, Math.ceil(remainingTotal / 6));
      const targetPage = currentPage > expectedTotalPages ? expectedTotalPages : currentPage;
      
      setCurrentPage(targetPage);
      fetchPairs(targetPage, searchQuery);
    } catch (err) {
      console.error(err);
      setMsg(err.response?.data?.detail || 'Failed to delete selected pairs.');
    }
  };

  const handleDeleteAll = async () => {
    const confirm1 = window.confirm("WARNING: This will permanently delete ALL training image pairs in this dataset. This action CANNOT be undone. Are you absolutely sure?");
    if (!confirm1) return;
    
    const confirm2 = window.prompt("To confirm dataset wipe, please type 'DELETE ALL' below:");
    if (confirm2 !== 'DELETE ALL') {
      alert("Wipe cancelled. Confirmation text did not match.");
      return;
    }
    
    try {
      setMsg("Wiping entire dataset...");
      await axios.delete('http://localhost:8000/api/admin/training-data/pairs/all');
      setMsg("Successfully cleared training dataset.");
      setSelectedIds([]);
      setCurrentPage(1);
      fetchPairs(1, searchQuery);
    } catch (err) {
      console.error(err);
      setMsg(err.response?.data?.detail || 'Failed to clear dataset.');
    }
  };

  const onDragOver = (e) => {
    e.preventDefault();
  };

  return (
    <div className="admin-page-container flex flex-col pt-4 w-full">
      <div className="w-full px-4 md:px-12 lg:px-24 py-8 flex-grow">
        
        {/* Header */}
        <button 
          onClick={() => navigate('/')}
          className="flex items-center gap-2 text-stone-500 hover:text-stone-900 transition-colors mb-8 font-medium text-sm w-fit"
        >
          <ArrowLeft className="w-4 h-4" /> Back to Application
        </button>

        <div className="mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full dev-badge text-xs font-mono mb-4">
            <TerminalSquare className="w-3 h-3" />
            ADMINISTRATION_ACCESS
          </div>
          <h1 className="text-4xl md:text-5xl font-serif text-stone-900 mb-4 tracking-tight">
            Weights & Dataset Administration
          </h1>
          <p className="text-stone-600 leading-relaxed text-lg w-full max-w-none">
            Hot-swap the offline PyTorch <code>.pth</code> models or manage custom image training datasets to power the AI colorization and restoration engines.
          </p>
        </div>

        {/* Global Message Alert */}
        {msg && (
          <div className={`mb-8 p-4 rounded-xl border flex items-center gap-3 ${msg.includes('Failed') || msg.includes('Invalid') || msg.includes('Error') ? 'bg-red-50 border-red-200 text-red-700' : 'bg-emerald-50 border-emerald-200 text-emerald-700'}`}>
            {msg.includes('Failed') || msg.includes('Invalid') || msg.includes('Error') ? <AlertCircle className="w-5 h-5 shrink-0" /> : <CheckCircle className="w-5 h-5 shrink-0" />}
            <p className="font-medium">{msg}</p>
          </div>
        )}

        {/* Weights Swap Section */}
        <div className="mb-16">
          <h2 className="text-2xl font-serif text-stone-900 tracking-wide mb-6 pb-2 border-b border-stone-200">
            Model Weights Hot-Swap
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            
            {/* RealESRGAN Zone */}
            <div className="cream-card p-8 flex flex-col">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-serif text-stone-900 tracking-wide">Restoration</h2>
                <span className="bg-stone-100 text-stone-600 text-xs px-3 py-1 rounded-full font-mono border border-stone-200">RealESRGAN_x4plus.pth</span>
              </div>
              
              <div className="architecture-details p-5 mb-8 flex-grow">
                <ul className="text-sm text-stone-600 space-y-3">
                  <li className="flex justify-between items-center border-b border-stone-200 pb-2">
                    <span>Architecture</span> <strong className="text-stone-800">RealESRGAN (RRDB x23)</strong>
                  </li>
                  <li className="flex justify-between items-center border-b border-stone-200 pb-2">
                    <span>Target Task</span> <strong className="text-stone-800">Denoising & 4x Upscaling</strong>
                  </li>
                  <li className="flex justify-between items-center border-b border-stone-200 pb-2">
                    <span>Weight File</span>
                    {weightsStatus?.restorer?.file_exists ? (
                      <span className="text-emerald-700 font-semibold flex items-center gap-1"><CheckCircle className="w-3.5 h-3.5" /> Present</span>
                    ) : (
                      <span className="text-red-600 font-semibold flex items-center gap-1"><AlertCircle className="w-3.5 h-3.5" /> Missing</span>
                    )}
                  </li>
                  <li className="flex justify-between items-center">
                    <span>Memory Status</span>
                    {weightsStatus?.restorer?.loaded ? (
                      <span className="relative flex items-center gap-2 text-emerald-700 font-semibold">
                        <span className="flex h-2 w-2 relative">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        Active ({weightsStatus.restorer.device})
                      </span>
                    ) : (
                      <span className="text-red-600 font-semibold flex items-center gap-1"><AlertCircle className="w-3.5 h-3.5" /> Not Active</span>
                    )}
                  </li>
                </ul>
              </div>
              
              <div 
                onDragOver={onDragOver} 
                onDrop={(e) => handleDrop(e, 'realesrgan')}
                className={`drop-zone p-8 flex flex-col items-center justify-center text-center cursor-pointer ${restorerStatus}`}
              >
                <input type="file" id="realesrgan-upload" className="hidden" accept=".pth" onChange={(e) => handleDrop(e, 'realesrgan')} />
                <label htmlFor="realesrgan-upload" className="cursor-pointer flex flex-col items-center w-full h-full">
                  {restorerStatus === 'uploading' ? (
                    <div className="w-10 h-10 border-4 border-stone-200 border-t-indigo-500 rounded-full animate-spin mb-4"></div>
                  ) : restorerStatus === 'success' ? (
                    <CheckCircle className="w-10 h-10 text-emerald-500 mb-4 scale-110 transition-transform" />
                  ) : (
                    <UploadCloud className={`w-10 h-10 mb-4 ${restorerStatus === 'error' ? 'text-red-400' : 'text-stone-400'}`} />
                  )}
                  <span className="font-semibold text-stone-800 text-base mb-1">{restorerStatus === 'uploading' ? 'Injecting Weights...' : 'Drop RealESRGAN_x4plus.pth'}</span>
                  <span className="text-xs text-stone-500">or click to browse local files</span>
                </label>
              </div>
            </div>

            {/* ECCV16 colorizer status */}
            <div className="cream-card p-8 flex flex-col">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-serif text-stone-900 tracking-wide">Colorization</h2>
                <span className="bg-stone-100 text-stone-600 text-xs px-3 py-1 rounded-full font-mono border border-stone-200">ECCV16</span>
              </div>
              
              <div className="architecture-details p-5 mb-8 flex-grow">
                <ul className="text-sm text-stone-600 space-y-3">
                  <li className="flex justify-between items-center border-b border-stone-200 pb-2">
                    <span>Architecture</span> <strong className="text-stone-800">ECCV16 Caffe DNN</strong>
                  </li>
                  <li className="flex justify-between items-center border-b border-stone-200 pb-2">
                    <span>Target Task</span> <strong className="text-stone-800">B&W to Natural Color</strong>
                  </li>
                  <li className="flex justify-between items-center border-b border-stone-200 pb-2">
                    <span>Weight File</span>
                    {weightsStatus?.colorizer?.file_exists ? (
                      <span className="text-emerald-700 font-semibold flex items-center gap-1"><CheckCircle className="w-3.5 h-3.5" /> Present</span>
                    ) : (
                      <span className="text-red-600 font-semibold flex items-center gap-1"><AlertCircle className="w-3.5 h-3.5" /> Missing</span>
                    )}
                  </li>
                  <li className="flex justify-between items-center">
                    <span>Memory Status</span>
                    {weightsStatus?.colorizer?.loaded ? (
                      <span className="relative flex items-center gap-2 text-emerald-700 font-semibold">
                        <span className="flex h-2 w-2 relative">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        Active ({weightsStatus.colorizer.device})
                      </span>
                    ) : (
                      <span className="text-red-600 font-semibold flex items-center gap-1"><AlertCircle className="w-3.5 h-3.5" /> Not Active</span>
                    )}
                  </li>
                </ul>
              </div>
              
              <div className="p-6 text-center text-sm text-stone-500 bg-stone-50 rounded-xl border border-stone-200">
                <Cpu className="w-8 h-8 mx-auto mb-3 text-stone-400" />
                <p className="font-medium text-stone-700 mb-1">Auto-loaded Model</p>
                <p>DDColor weights load automatically from <code className="text-xs bg-stone-100 px-1 py-0.5 rounded">ai/weights/ddcolor/</code>.</p>
                <p className="mt-1">Run <code className="text-xs bg-stone-100 px-1 py-0.5 rounded">python backend/ai/install_ddcolor.py</code> to download (~870 MB).</p>
              </div>
            </div>

          </div>
        </div>

        {/* Automatic Model Training Dashboard */}
        <div className="mb-16">
          <h2 className="text-2xl font-serif text-stone-900 tracking-wide mb-6 pb-2 border-b border-stone-200 flex items-center justify-between">
            <span>Automatic Model Training Dashboard</span>
            <span className="text-xs bg-stone-100 border border-stone-200 text-stone-500 font-mono px-2 py-0.5 rounded uppercase">
              Pipeline: {trainingState.status}
            </span>
          </h2>
          
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-8 mb-8">
            
            {/* Control Panel Card */}
            <div className="cream-card p-6 flex flex-col justify-between h-full">
              <div>
                <h3 className="text-lg font-serif text-stone-800 mb-3 flex items-center gap-2">
                  <Cpu className="w-5 h-5 text-indigo-500" /> Pipeline Controls
                </h3>
                <p className="text-xs text-stone-500 mb-6">
                  Run the RealESRGAN benchmark pipeline. The system measures real PSNR on sample patches from the H5 dataset and reports how much quality gain RealESRGAN achieves over bicubic upscaling.
                </p>
                
                {/* Active Dataset Stats */}
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="bg-stone-50 border border-stone-200 rounded-lg p-3 text-center">
                    <span className="block text-[10px] uppercase tracking-wider text-stone-400 font-semibold mb-1">Dataset Size</span>
                    <strong className="text-xl text-stone-800 font-serif">{totalPairs} Pairs</strong>
                  </div>
                  <div className="bg-stone-50 border border-stone-200 rounded-lg p-3 text-center">
                    <span className="block text-[10px] uppercase tracking-wider text-stone-400 font-semibold mb-1">Status</span>
                    <strong className="text-sm text-stone-700 capitalize flex items-center justify-center gap-1.5 mt-1">
                      {trainingState.status === 'running' ? (
                        <>
                          <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
                          </span>
                          Training...
                        </>
                      ) : trainingState.status === 'completed' ? (
                        <>
                          <CheckCircle className="w-4 h-4 text-emerald-500" />
                          Ready
                        </>
                      ) : (
                        'Idle'
                      )}
                    </strong>
                  </div>
                </div>
              </div>
              
              {/* Progress/Trigger area */}
              <div className="mt-4 space-y-4">
                {trainingState.status === 'running' && (
                  <div className="space-y-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-stone-500 font-medium truncate max-w-[200px]" title={trainingState.message}>{trainingState.message}</span>
                      <span className="text-indigo-600 font-bold">{trainingState.progress}%</span>
                    </div>
                    <div className="w-full bg-stone-200 rounded-full h-2.5 overflow-hidden shadow-inner">
                      <div 
                        className="bg-indigo-600 h-full rounded-full transition-all duration-500 ease-out"
                        style={{ width: `${trainingState.progress}%` }}
                      />
                    </div>
                  </div>
                )}
                
                <button
                  onClick={handleStartTraining}
                  disabled={isTraining || totalPairs === 0}
                  className="w-full bg-stone-950 hover:bg-stone-800 disabled:bg-stone-100 disabled:text-stone-400 border border-transparent disabled:border-stone-200 text-stone-100 font-semibold py-3 px-4 rounded-xl transition-all shadow-md hover:shadow-lg flex items-center justify-center gap-2 text-sm cursor-pointer"
                >
                  {isTraining ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" /> Running Pipeline...
                    </>
                  ) : (
                    <>
                      <Cpu className="w-4 h-4" /> Start Training Pipeline
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Terminal Live Output logs */}
            <div className="xl:col-span-2 cream-card p-6 flex flex-col h-full min-h-[300px]">
              <div className="flex items-center justify-between mb-3 border-b border-stone-200 pb-2.5">
                <h3 className="text-lg font-serif text-stone-800 flex items-center gap-2">
                  <TerminalSquare className="w-5 h-5 text-stone-600" /> Training Terminal Output
                </h3>
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" title="System online"></span>
              </div>
              
              <div className="training-terminal flex-grow p-4 text-xs text-stone-300 font-mono overflow-y-auto max-h-[220px] rounded-lg space-y-1.5 scrollbar-thin">
                {terminalLogs.length === 0 ? (
                  <div className="text-stone-500 italic text-center py-12">
                    Terminal idle. Click "Start Training Pipeline" to spawn training logs.
                  </div>
                ) : (
                  terminalLogs.map((log, i) => {
                    let colorClass = "text-stone-300";
                    if (log.startsWith("[SYSTEM]")) colorClass = "text-stone-400 font-medium";
                    else if (log.startsWith("[DATA]")) colorClass = "text-indigo-400";
                    else if (log.startsWith("[TRAIN]")) colorClass = "text-amber-400";
                    else if (log.startsWith("[EPOCH")) colorClass = "text-stone-300";
                    else if (log.startsWith("[SUCCESS]")) colorClass = "text-emerald-400 font-bold";
                    else if (log.startsWith("[RESULTS]")) colorClass = "text-emerald-300 font-semibold";
                    else if (log.startsWith("[ERROR]")) colorClass = "text-red-400 font-bold";
                    
                    return (
                      <div key={i} className={`training-terminal-line ${colorClass}`}>
                        {log}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>

          {/* Results dashboard & SVG Graphs */}
          {trainingState.epochs.length > 0 && (
            <div className="space-y-8 mt-8">
              
              {/* Metrics Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                {/* Restorer Card */}
                <div className="cream-card p-6 border-l-4 border-l-indigo-500">
                  <h4 className="font-serif text-lg text-stone-800 mb-4">Restorer Model (RealESRGAN)</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center border-b border-stone-100 pb-1.5">
                      <span className="text-xs text-stone-500 font-medium">Evaluation Accuracy</span>
                      <strong className="text-stone-800">{trainingState.results?.restorer_accuracy ? `${trainingState.results.restorer_accuracy}%` : 'Evaluating...'}</strong>
                    </div>
                    <div className="flex justify-between items-center border-b border-stone-100 pb-1.5">
                      <span className="text-xs text-stone-500 font-medium">Peak SNR (PSNR)</span>
                      <strong className="text-stone-800">{trainingState.results?.restorer_psnr ? `${trainingState.results.restorer_psnr} dB` : 'Calculating...'}</strong>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-stone-500 font-medium">MSE Training Loss</span>
                      <strong className="text-stone-800">{trainingState.results?.restorer_mse ? trainingState.results.restorer_mse.toFixed(4) : 'Running...'}</strong>
                    </div>
                  </div>
                </div>

                {/* Colorizer Card */}
                <div className="cream-card p-6 border-l-4 border-l-amber-500">
                  <h4 className="font-serif text-lg text-stone-800 mb-4">Colorizer Model (ECCV16)</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center border-b border-stone-100 pb-1.5">
                      <span className="text-xs text-stone-500 font-medium">Evaluation Accuracy</span>
                      <strong className="text-stone-800">{trainingState.results?.colorizer_accuracy ? `${trainingState.results.colorizer_accuracy}%` : 'Evaluating...'}</strong>
                    </div>
                    <div className="flex justify-between items-center border-b border-stone-100 pb-1.5">
                      <span className="text-xs text-stone-500 font-medium">F1 Score (ab Channels)</span>
                      <strong className="text-stone-800">{trainingState.results?.colorizer_f1 ? trainingState.results.colorizer_f1.toFixed(3) : 'Calculating...'}</strong>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-stone-500 font-medium">Cross-Entropy Loss</span>
                      <strong className="text-stone-800">{trainingState.results?.colorizer_loss ? trainingState.results.colorizer_loss.toFixed(4) : 'Running...'}</strong>
                    </div>
                  </div>
                </div>

                {/* Combined Pipeline Card */}
                <div className="cream-card p-6 border-l-4 border-l-emerald-500 bg-emerald-50/5">
                  <h4 className="font-serif text-lg text-stone-800 mb-4">Combined AI Engine</h4>
                  <div className="space-y-4">
                    <div className="text-center py-2 bg-stone-50 border border-stone-200 rounded-lg">
                      <span className="block text-[10px] uppercase font-semibold tracking-wider text-stone-400">Total Combined Accuracy</span>
                      <strong className="text-3xl text-emerald-600 font-serif font-bold mt-1 block">
                        {trainingState.results?.combined_accuracy ? `${trainingState.results.combined_accuracy}%` : 'Calculating...'}
                      </strong>
                    </div>
                    <p className="text-[10px] text-stone-400 text-center leading-relaxed">
                      Weighted metrics: 55% Restorer resolution + 45% Colorizer chromatic mapping.
                    </p>
                  </div>
                </div>

              </div>

              {/* Loss and Accuracy Graphs Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                
                {/* Loss Curves */}
                <div className="cream-card p-6">
                  <h3 className="text-lg font-serif text-stone-800 mb-4">Training Loss Convergence Curves</h3>
                  {renderLossChart()}
                </div>

                {/* Accuracy Curves */}
                <div className="cream-card p-6">
                  <h3 className="text-lg font-serif text-stone-800 mb-4">Validation Accuracy Progression</h3>
                  {renderAccuracyChart()}
                </div>

              </div>

            </div>
          )}
        </div>

        {/* Training Dataset Manager Section */}
        <div className="mb-16">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6 pb-2 border-b border-stone-200">
            <div>
              <h2 className="text-2xl font-serif text-stone-900 tracking-wide">
                Training Dataset Manager
              </h2>
              <p className="text-sm text-stone-500 mt-1">
                Upload image archives or add individual paired datasets to train and tune restoration models.
              </p>
            </div>
            
            <button 
              onClick={() => setShowAddForm(!showAddForm)}
              className="flex items-center gap-2 bg-stone-900 hover:bg-stone-800 text-stone-100 px-4 py-2 rounded-lg font-medium text-sm transition-colors self-start md:self-center"
            >
              <Plus className="w-4 h-4" /> {showAddForm ? 'Cancel Direct Upload' : 'Add Single Pair'}
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            
            {/* ZIP Drag and Drop Uploader */}
            <div className="lg:col-span-2 cream-card p-6 flex flex-col justify-between">
              <div>
                <h3 className="text-lg font-serif text-stone-800 mb-2 flex items-center gap-2">
                  <FileArchive className="w-5 h-5 text-stone-600" /> Upload Dataset ZIP
                </h3>
                <p className="text-xs text-stone-500 mb-4">
                  Archive structure: Put Black & White (degraded) images inside a <code>bw/</code> or <code>input/</code> folder, and matching Color (clean) targets inside a <code>color/</code> or <code>target/</code> folder with the same filenames.
                </p>
              </div>
              
              {isProcessing ? (
                <div className="p-8 border border-dashed border-indigo-300 bg-indigo-50/30 rounded-2xl flex flex-col items-center justify-center text-center">
                  <RefreshCw className="w-8 h-8 text-indigo-600 animate-spin mb-3" />
                  <span className="font-semibold text-stone-800 mb-2">{progressMsg}</span>
                  
                  {/* Progress Bar */}
                  <div className="w-full max-w-md bg-stone-200 rounded-full h-3 mb-2 overflow-hidden shadow-inner">
                    <div 
                      className="bg-indigo-600 h-full rounded-full transition-all duration-300 ease-out" 
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono font-bold text-indigo-600">{uploadProgress}%</span>
                </div>
              ) : (
                <div 
                  onDragOver={onDragOver} 
                  onDrop={handleZipDrop}
                  className={`drop-zone p-8 flex flex-col items-center justify-center text-center cursor-pointer ${zipStatus}`}
                >
                  <input type="file" id="zip-upload" className="hidden" accept=".zip" onChange={handleZipDrop} />
                  <label htmlFor="zip-upload" className="cursor-pointer flex flex-col items-center w-full h-full">
                    {zipStatus === 'uploading' ? (
                      <div className="w-10 h-10 border-4 border-stone-200 border-t-indigo-500 rounded-full animate-spin mb-4"></div>
                    ) : zipStatus === 'success' ? (
                      <CheckCircle className="w-10 h-10 text-emerald-500 mb-4 scale-110 transition-transform" />
                    ) : (
                      <UploadCloud className={`w-10 h-10 mb-4 ${zipStatus === 'error' ? 'text-red-400' : 'text-stone-400'}`} />
                    )}
                    <span className="font-semibold text-stone-800 text-base mb-1">
                      {zipStatus === 'uploading' ? 'Extracting ZIP & Pairing Images...' : 'Drop Dataset ZIP File'}
                    </span>
                    <span className="text-xs text-stone-500">or click to browse local files (.zip)</span>
                  </label>
                </div>
              )}
            </div>

            {/* Direct Upload Form */}
            <div className={`cream-card p-6 flex flex-col transition-all duration-300 ${showAddForm ? 'opacity-100 scale-100' : 'opacity-60 pointer-events-none'}`}>
              <h3 className="text-lg font-serif text-stone-800 mb-4 flex items-center gap-2">
                <Plus className="w-5 h-5 text-stone-600" /> Direct Pair Upload
              </h3>
              
              <form onSubmit={handleAddPair} className="space-y-4 flex-grow flex flex-col justify-between">
                <div className="space-y-3">
                  <div>
                    <label className="block text-xs font-semibold text-stone-700 uppercase tracking-wider mb-1">
                      Black & White (Source)
                    </label>
                    <input 
                      type="file" 
                      accept="image/*"
                      disabled={!showAddForm}
                      onChange={(e) => setNewBwFile(e.target.files?.[0] || null)}
                      className="w-full text-xs text-stone-600 file:mr-2 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-stone-100 file:text-stone-700 hover:file:bg-stone-200"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-stone-700 uppercase tracking-wider mb-1">
                      Color (Target / Clean)
                    </label>
                    <input 
                      type="file" 
                      accept="image/*"
                      disabled={!showAddForm}
                      onChange={(e) => setNewColorFile(e.target.files?.[0] || null)}
                      className="w-full text-xs text-stone-600 file:mr-2 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-stone-100 file:text-stone-700 hover:file:bg-stone-200"
                    />
                  </div>
                </div>

                <button 
                  type="submit"
                  disabled={!showAddForm || addStatus === 'submitting' || !newBwFile || !newColorFile}
                  className="w-full mt-4 bg-stone-800 hover:bg-stone-700 disabled:bg-stone-200 disabled:text-stone-400 text-stone-100 font-semibold py-2 rounded-lg text-sm transition-colors flex items-center justify-center gap-2"
                >
                  {addStatus === 'submitting' ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" /> Uploading Pair...
                    </>
                  ) : (
                    'Upload Pair'
                  )}
                </button>
              </form>
            </div>

          </div>

          {/* Paginated Pair Display */}
          <div className="cream-card p-6">
            
            {/* Search and Control Bar */}
            <div className="flex flex-col xl:flex-row xl:items-center justify-between pb-6 mb-6 border-b border-stone-200 gap-4">
              <div>
                <h3 className="text-xl font-serif text-stone-800">
                  Dataset Pairs Directory ({totalPairs} total pairs)
                </h3>
                <p className="text-xs text-stone-500 mt-0.5">
                  Showing in chunks of 6. Hover over any image to update it in-place.
                </p>
              </div>
              
              <div className="flex flex-wrap items-center gap-3">
                {/* Search Input */}
                <div className="relative min-w-[240px] flex-grow sm:flex-grow-0">
                  <Search className="w-4 h-4 text-stone-400 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input 
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search by filename..."
                    className="w-full pl-9 pr-4 py-2 border border-stone-300 rounded-lg text-sm bg-stone-50/50 focus:bg-white focus:ring-1 focus:ring-stone-400 focus:outline-none transition-colors"
                  />
                </div>

                {/* Select All on Page */}
                {pairs.length > 0 && (
                  <label className="flex items-center gap-2 border border-stone-300 rounded-lg px-3 py-2 text-xs font-semibold text-stone-600 bg-stone-50 hover:bg-stone-100 transition-colors cursor-pointer select-none">
                    <input 
                      type="checkbox"
                      checked={pairs.every(p => selectedIds.includes(p.id))}
                      onChange={(e) => {
                        if (e.target.checked) {
                          const newIds = [...selectedIds];
                          pairs.forEach(p => {
                            if (!newIds.includes(p.id)) newIds.push(p.id);
                          });
                          setSelectedIds(newIds);
                        } else {
                          const pageIds = pairs.map(p => p.id);
                          setSelectedIds(prev => prev.filter(id => !pageIds.includes(id)));
                        }
                      }}
                      className="w-3.5 h-3.5 rounded text-indigo-600 border-stone-300 focus:ring-indigo-500 cursor-pointer"
                    />
                    Select Page
                  </label>
                )}

                {/* Batch Actions */}
                {selectedIds.length > 0 && (
                  <button
                    onClick={handleBatchDelete}
                    className="flex items-center gap-1.5 bg-red-50 hover:bg-red-100 text-red-700 border border-red-200 px-3 py-2 rounded-lg font-semibold text-xs transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" /> Delete Selected ({selectedIds.length})
                  </button>
                )}

                {/* Delete All Wiping Database */}
                {totalPairs > 0 && (
                  <button
                    onClick={handleDeleteAll}
                    className="flex items-center gap-1.5 bg-stone-100 hover:bg-red-50 text-stone-600 hover:text-red-700 border border-stone-200 hover:border-red-200 px-3 py-2 rounded-lg font-semibold text-xs transition-colors ml-auto sm:ml-0"
                  >
                    <Trash2 className="w-3.5 h-3.5" /> Wipe Entire Dataset
                  </button>
                )}
              </div>
            </div>

            {/* Chunk/Page Selector Top */}
            {totalPages > 1 && (
              <div className="flex justify-end items-center gap-2 mb-6">
                <button 
                  disabled={currentPage === 1}
                  onClick={() => {
                    const prev = Math.max(1, currentPage - 1);
                    setCurrentPage(prev);
                    fetchPairs(prev, searchQuery);
                  }}
                  className="p-1.5 rounded-lg border border-stone-300 hover:bg-stone-100 disabled:opacity-40 disabled:hover:bg-transparent transition-colors"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <span className="text-sm font-mono text-stone-600">
                  Chunk {currentPage} of {totalPages}
                </span>
                <button 
                  disabled={currentPage === totalPages}
                  onClick={() => {
                    const next = Math.min(totalPages, currentPage + 1);
                    setCurrentPage(next);
                    fetchPairs(next, searchQuery);
                  }}
                  className="p-1.5 rounded-lg border border-stone-300 hover:bg-stone-100 disabled:opacity-40 disabled:hover:bg-transparent transition-colors"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            )}

            {pairs.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-center text-stone-400">
                <ImageIcon className="w-12 h-12 mb-3 stroke-[1.5]" />
                <p className="text-lg font-serif mb-1">No training data found</p>
                <p className="text-sm">
                  {searchQuery ? "No matches found for your search query." : "Upload a dataset ZIP or add a single image pair to get started."}
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                {pairs.map((pair) => {
                  const isChecked = selectedIds.includes(pair.id);
                  return (
                    <div 
                      key={pair.id} 
                      className={`pair-card p-4 rounded-xl border flex flex-col justify-between relative transition-all ${isChecked ? 'bg-indigo-50/20 border-indigo-300 ring-1 ring-indigo-300' : 'bg-stone-50 border-stone-200'}`}
                    >
                      
                      {/* Checkbox Overlay */}
                      <div className="absolute top-2 left-2 z-10 bg-white/90 backdrop-blur rounded p-1 border border-stone-200 shadow-sm flex items-center justify-center">
                        <input 
                          type="checkbox"
                          checked={isChecked}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedIds(prev => [...prev, pair.id]);
                            } else {
                              setSelectedIds(prev => prev.filter(id => id !== pair.id));
                            }
                          }}
                          className="w-4 h-4 rounded text-indigo-600 border-stone-300 focus:ring-indigo-500 cursor-pointer"
                        />
                      </div>
                      
                      {/* Images Container */}
                      <div className="grid grid-cols-2 gap-3 mb-4 mt-4">
                        
                        {/* B&W Side */}
                        <div className="relative group overflow-hidden rounded-lg border border-stone-300 aspect-square bg-stone-200">
                          <img 
                            src={`http://localhost:8000${pair.bw_url}`} 
                            alt="Black & White" 
                            className="w-full h-full object-cover" 
                          />
                          <div className="absolute inset-0 bg-stone-900/60 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center text-stone-100 p-2 text-center">
                            <input 
                              type="file" 
                              accept="image/*"
                              id={`update-bw-${pair.id}`} 
                              className="hidden" 
                              onChange={(e) => handleUpdateImage(pair.id, 'bw', e.target.files?.[0])}
                            />
                            <label 
                              htmlFor={`update-bw-${pair.id}`} 
                              className="cursor-pointer text-xs font-semibold px-2 py-1 rounded bg-stone-800 hover:bg-stone-700 transition-colors flex items-center gap-1 shadow-sm"
                            >
                              <RefreshCw className="w-3 h-3" /> Change B&W
                            </label>
                          </div>
                          <span className="absolute bottom-1 left-1 bg-stone-900/75 text-stone-100 font-mono text-[9px] px-1.5 py-0.5 rounded uppercase tracking-wider">
                            B&W / Degraded
                          </span>
                        </div>

                        {/* Color Side */}
                        <div className="relative group overflow-hidden rounded-lg border border-stone-300 aspect-square bg-stone-200">
                          <img 
                            src={`http://localhost:8000${pair.color_url}`} 
                            alt="Color Target" 
                            className="w-full h-full object-cover" 
                          />
                          <div className="absolute inset-0 bg-stone-900/60 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center text-stone-100 p-2 text-center">
                            <input 
                              type="file" 
                              accept="image/*"
                              id={`update-color-${pair.id}`} 
                              className="hidden" 
                              onChange={(e) => handleUpdateImage(pair.id, 'color', e.target.files?.[0])}
                            />
                            <label 
                              htmlFor={`update-color-${pair.id}`} 
                              className="cursor-pointer text-xs font-semibold px-2 py-1 rounded bg-stone-800 hover:bg-stone-700 transition-colors flex items-center gap-1 shadow-sm"
                            >
                              <RefreshCw className="w-3 h-3" /> Change Color
                            </label>
                          </div>
                          <span className="absolute bottom-1 left-1 bg-stone-900/75 text-stone-100 font-mono text-[9px] px-1.5 py-0.5 rounded uppercase tracking-wider">
                            Color / Target
                          </span>
                        </div>

                      </div>

                      {/* Metadata & Controls */}
                      <div className="flex items-center justify-between gap-2 border-t border-stone-200 pt-3">
                        <div className="min-w-0">
                          <p className="text-xs font-semibold text-stone-800 truncate" title={pair.original_name}>
                            {pair.original_name}
                          </p>
                          <p className="text-[10px] text-stone-500 font-mono">
                            ID: {pair.id}
                          </p>
                        </div>

                        <button 
                          onClick={() => handleDeletePair(pair.id)}
                          className="text-stone-400 hover:text-red-600 p-1.5 hover:bg-red-50 rounded-lg transition-colors flex-shrink-0"
                          title="Delete Pair"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>

                    </div>
                  );
                })}
              </div>
            )}
            
            {/* Pagination Controls at Bottom */}
            {totalPages > 1 && (
              <div className="flex justify-center items-center gap-4 mt-8 pt-4 border-t border-stone-200">
                <button 
                  disabled={currentPage === 1}
                  onClick={() => {
                    const prev = Math.max(1, currentPage - 1);
                    setCurrentPage(prev);
                    fetchPairs(prev, searchQuery);
                  }}
                  className="flex items-center gap-1 text-sm font-medium text-stone-600 hover:text-stone-900 disabled:opacity-40 disabled:hover:text-stone-600 transition-colors"
                >
                  <ChevronLeft className="w-4 h-4" /> Prev Chunk
                </button>
                <span className="text-sm font-mono text-stone-500">
                  {currentPage} / {totalPages}
                </span>
                <button 
                  disabled={currentPage === totalPages}
                  onClick={() => {
                    const next = Math.min(totalPages, currentPage + 1);
                    setCurrentPage(next);
                    fetchPairs(next, searchQuery);
                  }}
                  className="flex items-center gap-1 text-sm font-medium text-stone-600 hover:text-stone-900 disabled:opacity-40 disabled:hover:text-stone-600 transition-colors"
                >
                  Next Chunk <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            )}

          </div>
        </div>

      </div>
    </div>
  );
}
