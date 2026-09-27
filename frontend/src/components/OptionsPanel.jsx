// OptionsPanel — the collapsible "More Options" block (plan 5.6 / A-7):
// Q-8 preset selector, the four pipeline toggles, and device preference.
import { Image as ImageIcon, ShieldCheck, Settings2, ChevronDown, ChevronUp, Wand2, Sparkles } from 'lucide-react';

const DEVICE_CHOICES = [
  { id: 'auto', label: 'Auto' },
  { id: 'cpu', label: 'CPU' },
  { id: 'gpu', label: 'GPU' },
];

export default function OptionsPanel({
  presets,
  activePreset,
  applyPreset,
  activeOptionsCount,
  showMoreOptions,
  setShowMoreOptions,
  enableRestore, enableRepair, enableDenoise, enableColorize, enableReimagine,
  setEnableRestore, setEnableRepair, setEnableDenoise, setEnableColorize, setEnableReimagine,
  selectedDevice, setSelectedDevice,
}) {
  return (
    <div className="flex flex-col items-center gap-3 mt-4">
      {/* Q-8 preset selector */}
      {presets && (
        <div className="flex items-center gap-2 flex-wrap justify-center">
          <Wand2 className="w-4 h-4 text-amber-500" />
          {Object.entries(presets).map(([id, preset]) => (
            <button
              key={id}
              type="button"
              title={preset.description}
              onClick={() => applyPreset(id)}
              className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-all cursor-pointer ${
                activePreset === id
                  ? 'bg-stone-900 text-white border-stone-900 shadow-sm'
                  : 'bg-white text-stone-600 border-stone-200 hover:border-stone-400 hover:text-stone-800'
              }`}
            >
              {preset.label}
            </button>
          ))}
        </div>
      )}

      {/* Auto-applied badge */}
      <div className="flex items-center gap-2 text-sm text-stone-500">
        <ShieldCheck className="w-4 h-4 text-emerald-500" />
        <span>
          <strong className="text-stone-700">{activeOptionsCount} enhancements</strong> will be applied automatically
        </span>
      </div>

      {/* More Options collapsible */}
      <button
        type="button"
        onClick={() => setShowMoreOptions(!showMoreOptions)}
        className="flex items-center gap-1.5 text-sm text-stone-500 hover:text-stone-700 transition-colors cursor-pointer font-medium"
      >
        <Settings2 className="w-3.5 h-3.5" />
        More Options
        {showMoreOptions ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>

      {showMoreOptions && (
        <div className="flex flex-col items-center gap-3 text-sm text-stone-600 bg-stone-50 py-4 px-6 rounded border border-stone-200 w-fit mx-auto animate-in fade-in duration-200">

          <div className="flex items-center gap-2">
            <ImageIcon className="w-4 h-4 text-indigo-600" />
            <input
              type="checkbox"
              id="enableRestore"
              checked={enableRestore}
              onChange={(e) => setEnableRestore(e.target.checked)}
              className="w-4 h-4 text-indigo-600 rounded border-stone-300 focus:ring-indigo-500 cursor-pointer"
            />
            <label htmlFor="enableRestore" className="cursor-pointer font-medium select-none">
              AI Restoration &amp; Upscaling (4× RealESRGAN)
            </label>
          </div>

          <div className="w-full h-px bg-stone-200"></div>

          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-violet-500" />
            <input
              type="checkbox"
              id="enableRepair"
              checked={enableRepair}
              onChange={(e) => setEnableRepair(e.target.checked)}
              className="w-4 h-4 text-violet-500 rounded border-stone-300 focus:ring-violet-500 cursor-pointer"
            />
            <label htmlFor="enableRepair" className="cursor-pointer font-medium select-none">
              Repair Damage (Scratches, White Spots, Cracks)
            </label>
          </div>

          <div className="w-full h-px bg-stone-200"></div>

          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-rose-500" />
            <input
              type="checkbox"
              id="enableDenoise"
              checked={enableDenoise}
              onChange={(e) => setEnableDenoise(e.target.checked)}
              className="w-4 h-4 text-rose-500 rounded border-stone-300 focus:ring-rose-500 cursor-pointer"
            />
            <label htmlFor="enableDenoise" className="cursor-pointer font-medium select-none">
              Remove Dust &amp; Scratches (Denoise)
            </label>
          </div>

          <div className="w-full h-px bg-stone-200"></div>

          <div className="flex items-center gap-2">
            <ImageIcon className="w-4 h-4 text-indigo-500" />
            <input
              type="checkbox"
              id="enableColorize"
              checked={enableColorize}
              onChange={(e) => setEnableColorize(e.target.checked)}
              className="w-4 h-4 text-indigo-500 rounded border-stone-300 focus:ring-indigo-500 cursor-pointer"
            />
            <label htmlFor="enableColorize" className="cursor-pointer font-medium select-none">
              AI Colorization (For B&amp;W photos)
            </label>
          </div>

          {/* Phase H1/H2 — Re-imagine (beta) */}
          <div className="w-full h-px bg-stone-200"></div>
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-500" />
            <input
              type="checkbox"
              id="enableReimagine"
              checked={enableReimagine}
              onChange={(e) => setEnableReimagine(e.target.checked)}
              className="w-4 h-4 text-amber-500 rounded border-stone-300 focus:ring-amber-500 cursor-pointer"
            />
            <label htmlFor="enableReimagine" className="cursor-pointer font-medium select-none">
              Re-imagine lost regions (beta)
            </label>
            <span
              className="text-[11px] text-stone-400 leading-tight"
              title="Opt-in beta. When damage is too extensive to repair (≥35% of the photo or over half a face), AI re-imagines only the destroyed regions while every surviving pixel is preserved exactly and disclosed in the Restoration Report. Adds several minutes per photo and only fires when the damage is truly hopeless."
            >ⓘ</span>
          </div>

          {/* Device Preference */}
          <div className="flex flex-col sm:flex-row items-center gap-3 w-full justify-between pt-1">
            <span className="font-semibold text-xs text-stone-500 uppercase tracking-wider">Device:</span>
            <div className="inline-flex rounded-lg border border-stone-200 bg-stone-100 p-0.5 shadow-sm text-xs font-semibold">
              {DEVICE_CHOICES.map((dev) => (
                <button
                  key={dev.id}
                  type="button"
                  onClick={() => setSelectedDevice(dev.id)}
                  className={`px-3 py-1.5 rounded-md transition-all cursor-pointer ${
                    selectedDevice === dev.id
                      ? 'bg-white text-stone-900 shadow-sm border border-stone-200/50'
                      : 'text-stone-500 hover:text-stone-800'
                  }`}
                >
                  {dev.label}
                </button>
              ))}
            </div>
          </div>

        </div>
      )}
    </div>
  );
}
