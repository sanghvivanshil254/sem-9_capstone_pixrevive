import { ArrowLeft, Cpu, Layers, Workflow, CheckCircle, BookOpen } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Documentation() {
  return (
    <div className="max-w-4xl mx-auto px-4 md:px-8 py-12">
      {/* Back to Home Link */}
      <div className="mb-8">
        <Link to="/" className="inline-flex items-center gap-2 text-stone-500 hover:text-stone-900 transition-colors text-sm font-medium">
          <ArrowLeft className="w-4 h-4" /> Back to Workspace
        </Link>
      </div>

      {/* Page Header */}
      <div className="mb-12 border-b border-stone-200 pb-8">
        <h1 className="text-4xl md:text-5xl font-serif text-stone-900 tracking-tight mb-4">
          PixRevive Engine Documentation
        </h1>
        <p className="text-lg text-stone-600 font-light max-w-2xl leading-relaxed">
          Understanding the deep learning pipelines, neural architectures, and data flows behind professional archival photo restoration.
        </p>
      </div>

      {/* Main Flow Overview Section */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <Workflow className="w-6 h-6 text-indigo-500" /> Restoration Pipeline Overview
        </h2>
        <div className="cream-card p-6 mb-6">
          <p className="text-stone-700 leading-relaxed mb-4">
            PixRevive operates as a dual-network sequential engine. When an image is submitted, it passes through two primary stages: 
            <strong> Phase 1: Spatial & Luminance Restoration</strong> followed by <strong>Phase 2: Chromatic Colorization</strong>.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
            <div className="bg-stone-50 border border-stone-200 rounded-lg p-5">
              <span className="text-xs bg-indigo-100 border border-indigo-200 text-indigo-700 font-mono px-2 py-0.5 rounded uppercase font-semibold">Phase 1</span>
              <h3 className="font-serif text-lg text-stone-800 mt-2 mb-1.5">Real-ESRGAN Restorer</h3>
              <p className="text-xs text-stone-500 leading-relaxed">
                Super-resolves and sharpens structural details using state-of-the-art Real-ESRGAN (4× RRDB network) on full RGB channels for maximum quality. Falls back to bicubic 2× upscaling if weights are missing. Eliminates scanner noise, grain, and digital artifacts.
              </p>
            </div>
            <div className="bg-stone-50 border border-stone-200 rounded-lg p-5">
              <span className="text-xs bg-amber-100 border border-amber-200 text-amber-700 font-mono px-2 py-0.5 rounded uppercase font-semibold">Phase 2</span>
              <h3 className="font-serif text-lg text-stone-800 mt-2 mb-1.5">ECCV16 Colorizer</h3>
              <p className="text-xs text-stone-500 leading-relaxed">
                Maps lightness values to predicted chromatic ranges, introducing realistic colorization based on deep contextual priors.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Preprocessing Details */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <Layers className="w-6 h-6 text-indigo-500" /> Phase 0.5: Vector-Guided Crack Removal
        </h2>
        <div className="space-y-6">
          <p className="text-stone-700 leading-relaxed">
            Heavily textured images (like those with canvas cracks or paint damage) overwhelm standard AI restorers. To solve this, PixRevive uses a hybrid mathematical approach: combining <strong>vector-style edge detection</strong> to protect structural outlines, and <strong>Navier-Stokes raster inpainting</strong> to melt away the cracks before the AI takes over.
          </p>

          <div className="bg-stone-900 border border-stone-800 text-stone-300 font-mono text-[11px] p-5 rounded-lg overflow-x-auto leading-relaxed shadow-inner">
            <div className="text-indigo-400 mb-2"># Vector/Raster Crack Inpainting</div>
            <div className="text-amber-400 mt-1"># 1. Isolate cracks (Blackhat) and Vector Outlines (Canny)</div>
            <div>blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)</div>
            <div>edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)</div>
            <div className="text-amber-400 mt-2"># 2. Subtract protected edges from the crack mask</div>
            <div>crack_mask = cv2.bitwise_and(crack_mask, cv2.bitwise_not(edges))</div>
            <div className="text-amber-400 mt-2"># 3. Navier-Stokes fluid dynamics inpainting (Raster)</div>
            <div>img_clean = cv2.inpaint(img_np, crack_mask, 3, cv2.INPAINT_NS)</div>
          </div>
        </div>
      </section>

      {/* Real-ESRGAN Restorer Details */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <Cpu className="w-6 h-6 text-indigo-500" /> Phase 1: Real-ESRGAN Restorer
        </h2>
        <div className="space-y-6">
          <p className="text-stone-700 leading-relaxed">
            Once the structural cracks are inpainted, the image is passed to the state-of-the-art <strong>Real-ESRGAN (RRDBNet)</strong> architecture for unparalleled artifact removal and face restoration on the full RGB image.
          </p>

          <div className="bg-stone-900 border border-stone-800 text-stone-300 font-mono text-[11px] p-5 rounded-lg overflow-x-auto leading-relaxed shadow-inner">
            <div className="text-indigo-400 mb-2"># Restoration Pipeline</div>
            <div className="text-indigo-400 mt-1"># SOTA Real-ESRGAN Pipeline (Full RGB)</div>
            <div className="text-amber-400">img_t = torch.Tensor(img_rgb).permute(2, 0, 1).unsqueeze(0) / 255.0</div>
            <div>out_t = rrdbnet_model(img_t).clamp(0.0, 1.0)</div>
          </div>

          <div className="cream-card p-6">
            <h4 className="font-serif text-lg text-stone-800 mb-3">Key Design Advantages:</h4>
            <ul className="space-y-2 text-sm text-stone-600">
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                <strong>100% Offline Capability:</strong> The RRDBNet architecture is baked directly into the backend source code. This allows the system to load gigabyte-scale state-of-the-art weights locally without relying on any external packages like PyTorch Hub or HuggingFace during runtime.
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                <strong>Uncompromising Quality:</strong> By processing all three color channels (RGB) simultaneously using deep residual dense blocks, it repairs severe artifacts and textures (like faces) far better than lightweight luminance-only models.
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* ECCV16 Colorizer Details */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <Layers className="w-6 h-6 text-amber-500" /> Phase 2: ECCV16 Colorizer
        </h2>
        <div className="space-y-6">
          <p className="text-stone-700 leading-relaxed">
            The Colorizer implements the architecture proposed by Richard Zhang et al. in the ECCV 2016 paper <em>"Colorful Image Colorization"</em>. The model treats colorization as a classification problem, predicting a probability distribution over quantized color bins in the <strong>CIE LAB</strong> color space.
          </p>

          <div className="bg-stone-50 border border-stone-200 rounded-lg p-6">
            <h3 className="font-serif text-stone-800 text-lg mb-3">How CIE LAB Processing Works:</h3>
            <p className="text-sm text-stone-600 leading-relaxed mb-4">
              Unlike the standard RGB color space, LAB separates lightness (<strong>L</strong>) from color information (<strong>a</strong> represents green-red balance, and <strong>b</strong> represents blue-yellow balance). 
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center mt-6">
              <div className="bg-white border border-stone-200 rounded p-3">
                <strong className="block text-stone-900 text-base">L Channel</strong>
                <span className="text-[11px] text-stone-500">Lightness (0 to 100)<br />Input Grayscale Image</span>
              </div>
              <div className="bg-white border border-stone-200 rounded p-3">
                <strong className="block text-stone-900 text-base">a Channel</strong>
                <span className="text-[11px] text-stone-500">Predicted chrominance<br />Green vs Red axis</span>
              </div>
              <div className="bg-white border border-stone-200 rounded p-3">
                <strong className="block text-stone-900 text-base">b Channel</strong>
                <span className="text-[11px] text-stone-500">Predicted chrominance<br />Blue vs Yellow axis</span>
              </div>
            </div>
          </div>

          <h3 className="font-serif text-stone-800 text-xl mb-3">Processing Pipeline:</h3>
          <div className="bg-stone-900 border border-stone-800 text-stone-300 font-mono text-[11px] p-5 rounded-lg overflow-x-auto leading-relaxed shadow-inner">
            <div className="text-indigo-400 mb-2"># 1. Convert source image to LAB space and resize input to model resolution (256x256)</div>
            <div>img_lab = cv2.cvtColor(img_resized.astype("float32") / 255.0, cv2.COLOR_RGB2LAB)</div>
            <div>L_channel = img_lab[:, :, 0]</div>
            <div className="mt-2 text-indigo-400"># 2. Prepare tensor for ECCVGenerator network (shape: 1, 1, 256, 256)</div>
            <div>L_tensor = torch.Tensor(L_channel).unsqueeze(0).unsqueeze(0).to(device)</div>
            <div className="mt-2 text-indigo-400"># 3. Model forward pass outputs normalized ab channels [-1.0, 1.0]</div>
            <div>ab_tensor = colorizer_model(L_tensor)</div>
            <div className="mt-2 text-indigo-400"># 4. Unnormalize ab output by scaling factor (110.0) & resize back to source shape</div>
            <div>ab_pred = ab_tensor.cpu().numpy()[0].transpose((1, 2, 0)) * 110.0</div>
            <div>ab_pred_resized = cv2.resize(ab_pred, (w, h))</div>
            <div className="mt-2 text-indigo-400"># 5. Concatenate original High-Res L channel with upscaled predicted ab channels</div>
            <div>colorized_lab = np.concatenate((L_orig[:, :, np.newaxis], ab_pred_resized), axis=2)</div>
            <div>colorized_rgb = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2RGB)</div>
          </div>

          <div className="cream-card p-6">
            <h4 className="font-serif text-lg text-stone-800 mb-3">Core Algorithmic Fixes Implemented:</h4>
            <ul className="space-y-2.5 text-sm text-stone-600">
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                <strong>Double Normalization Fix:</strong> The network's internal input normalization function <code>normalize_l()</code> automatically subtracts `50` and divides by `100` to convert the L range from `[0, 100]` to `[-0.5, 0.5]`. The external preprocessing layer was updated to prevent subtracting `50` twice, resolving the bias-shift bug.
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                <strong>Ab Chrominance Scaling (110.0):</strong> The raw convolutional logits represent a probability map that is mapped to normalized color bins. Unnormalizing the predicted color maps by multiplying by `110.0` correctly expands the chrominance values to the full range of OpenCV LAB space, reviving the image colors.
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Project References & Pipeline Flow */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <Workflow className="w-6 h-6 text-indigo-500" /> Pipeline Implementation References
        </h2>
        <div className="cream-card p-6">
          <p className="text-stone-700 leading-relaxed mb-4 text-sm">
            The colorization and restoration logic is organized across clean, decoupled files in the backend codebase:
          </p>
          <div className="space-y-4 text-xs font-mono text-stone-600">
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-850 font-semibold mb-1">1. API Entry Point:</strong>
              <code>backend/routers/image_router.py</code> - Receives multipart image uploads, reads colorization boolean and device preference parameters, and returns binary JPEG payloads.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-850 font-semibold mb-1">2. Service Layer Wrapper:</strong>
              <code>backend/services/image_service.py</code> - Acts as the entry layer for image transformations, delegating tasks to the singleton <code>ImageEnhancer</code> instance.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-850 font-semibold mb-1">3. Enhancement Orchestrator:</strong>
              <code>backend/ai_model/inference.py</code> - Drives the sequential execution (Stage 0: denoising → Stage 0.5: generative inpainting → Stage 1: RealESRGAN restorer → Stage 2: ECCV16 colorizer) and handles dynamic model transfers between CPU/GPU.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-850 font-semibold mb-1">4. Chrominance Processing & Post-Scale:</strong>
              <code>backend/ai_model/colorizer/colorizer.py</code> - Implements LAB conversions, extracts lightness, feeds the generator, scales outputs by 110.0, and reconstructs the RGB final image.
            </div>
            <div className="p-3 bg-stone-50 border border-stone-200 rounded-lg">
              <strong className="block text-stone-850 font-semibold mb-1">5. Neural Network Architecture:</strong>
              <code>backend/ai_model/colorizer/model/model.py</code> - Holds the VGG-style <code>ECCVGenerator</code> class layers, weights definitions, and L-channel normalization functions.
            </div>
          </div>
        </div>
      </section>

      {/* Dataset Credits */}
      <section className="mb-16">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <BookOpen className="w-6 h-6 text-indigo-500" /> Training Dataset Credits
        </h2>
        <div className="bg-stone-50 border border-stone-200 rounded-lg p-6">
          <p className="text-stone-700 leading-relaxed mb-4 text-sm">
            Special credits and appreciation are given to <strong>Aayush9753</strong> for publishing the public colorization dataset used during model training and evaluation:
          </p>
          <div className="flex flex-wrap items-center justify-between gap-4 mt-2">
            <a 
              href="https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset" 
              target="_blank" 
              rel="noreferrer" 
              className="text-xs bg-white hover:bg-stone-50 border border-stone-200 text-indigo-600 font-semibold py-2.5 px-4 rounded-lg shadow-sm transition-all flex items-center gap-2 hover:border-stone-300"
            >
              <Layers className="w-4 h-4 text-indigo-500" /> Kaggle: Image Colorization Dataset
            </a>
            <span className="text-[11px] text-stone-400 font-mono">Published under Public Domain / CC0 LICENSE</span>
          </div>
        </div>
      </section>

      {/* Runtime Device Preference Selection */}
      <section className="mb-16 border-t border-stone-200 pt-12">
        <h2 className="text-2xl font-serif text-stone-800 mb-6 flex items-center gap-2">
          <Cpu className="w-6 h-6 text-stone-700" /> Runtime Device Management
        </h2>
        <p className="text-stone-700 leading-relaxed mb-4">
          To ensure deployment versatility, PixRevive supports dynamic device execution routing at the API route level:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm mt-4">
          <div className="bg-stone-50 border border-stone-200 rounded-lg p-4">
            <strong className="block text-stone-800 mb-1">Auto Preference</strong>
            <p className="text-xs text-stone-500 leading-relaxed">
              Queries CUDA availability on launch. Routes execution to the GPU if active, falling back to CPU if unavailable.
            </p>
          </div>
          <div className="bg-stone-50 border border-stone-200 rounded-lg p-4">
            <strong className="block text-stone-800 mb-1">CPU Only</strong>
            <p className="text-xs text-stone-500 leading-relaxed">
              Forces tensors and model weights to PyTorch CPU device. Best for environments without discrete graphics hardware.
            </p>
          </div>
          <div className="bg-stone-50 border border-stone-200 rounded-lg p-4">
            <strong className="block text-stone-800 mb-1">GPU Only</strong>
            <p className="text-xs text-stone-500 leading-relaxed">
              Forces execution to CUDA. Accelerates inference times by parallelizing matrix convolutions across GPU kernels.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
