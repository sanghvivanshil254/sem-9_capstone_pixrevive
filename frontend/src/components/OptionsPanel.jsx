// OptionsPanel — the collapsible "More Options" block (plan 5.6 / A-7):
// Q-8 preset selector, the four pipeline toggles, and device preference.
import {
  Image as ImageIcon,
  ShieldCheck,
  Settings2,
  ChevronDown,
  ChevronUp,
  Wand2,
  Sparkles,
  Cpu
} from 'lucide-react';

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
  enableRestore,
  enableRepair,
  enableDenoise,
  enableColorize,
  enableReimagine,
  setEnableRestore,
  setEnableRepair,
  setEnableDenoise,
  setEnableColorize,
  setEnableReimagine,
  selectedDevice,
  setSelectedDevice,
}) {
  return (
    <div className="flex flex-col items-center gap-3.5 mt-2 sm:mt-4 w-full max-w-2xl mx-auto px-1">
      {/* Q-8 preset selector */}
      {presets && (
        <div className="flex items-center gap-2 flex-wrap justify-center w-full">
          <span className="flex items-center gap-1 text-xs font-semibold uppercase tracking-wider text-stone-500 mr-1 shrink-0">
            <Wand2 className="w-3.5 h-3.5 text-amber-500" /> Presets:
          </span>
          {Object.entries(presets).map(([id, preset]) => (
            <button
              key={id}
              type="button"
              title={preset.description}
              onClick={() => applyPreset(id)}
              className={`px-3.5 py-2 rounded-full text-xs font-semibold border transition-all cursor-pointer min-h-[38px] flex items-center justify-center ${
                activePreset === id
                  ? 'bg-stone-900 text-white border-stone-900 shadow-sm'
                  : 'bg-white text-stone-600 border-stone-200 hover:border-stone-400 hover:text-stone-900 active:bg-stone-100'
              }`}
            >
              {preset.label}
            </button>
          ))}
        </div>
      )}

      {/* Auto-applied badge */}
      <div className="flex items-center gap-2 text-xs sm:text-sm text-stone-500 text-center px-2">
        <ShieldCheck className="w-4 h-4 text-emerald-500 shrink-0" />
        <span>
          <strong className="text-stone-700">{activeOptionsCount} enhancements</strong> will be applied automatically
        </span>
      </div>

      {/* More Options collapsible toggle */}
      <button
        type="button"
        onClick={() => setShowMoreOptions(!showMoreOptions)}
        className="flex items-center gap-1.5 text-xs sm:text-sm text-stone-600 hover:text-stone-900 active:text-stone-950 transition-colors cursor-pointer font-medium min-h-[44px] px-3 rounded-lg hover:bg-stone-100/60"
        aria-expanded={showMoreOptions}
        aria-controls="advanced-restoration-options"
      >
        <Settings2 className="w-4 h-4 text-stone-500" />
        <span>More Options</span>
        {showMoreOptions ? <ChevronUp className="w-4 h-4 text-stone-500" /> : <ChevronDown className="w-4 h-4 text-stone-500" />}
      </button>

      {/* Collapsible Options Container */}
      {showMoreOptions && (
        <div
          id="advanced-restoration-options"
          className="w-full flex flex-col gap-2.5 text-sm text-stone-700 bg-stone-50 p-3.5 sm:p-5 rounded-xl border border-stone-200 animate-in fade-in duration-200"
        >
          {/* AI Restoration & Upscaling */}
          <label
            htmlFor="enableRestore"
            className="flex items-start sm:items-center gap-3 p-2 rounded-lg hover:bg-stone-100/80 active:bg-stone-200/50 transition-colors cursor-pointer min-h-[44px]"
          >
            <input
              type="checkbox"
              id="enableRestore"
              checked={enableRestore}
              onChange={(e) => setEnableRestore(e.target.checked)}
              className="w-4 h-4 sm:w-5 sm:h-5 text-indigo-600 rounded border-stone-300 focus:ring-indigo-500 cursor-pointer mt-0.5 sm:mt-0 shrink-0"
            />
            <div className="flex items-center gap-2 min-w-0">
              <ImageIcon className="w-4 h-4 text-indigo-600 shrink-0" />
              <span className="font-medium text-xs sm:text-sm select-none">
                AI Restoration &amp; Upscaling (4&times; RealESRGAN)
              </span>
            </div>
          </label>

          <div className="w-full h-px bg-stone-200/70" />

          {/* Repair Damage */}
          <label
            htmlFor="enableRepair"
            className="flex items-start sm:items-center gap-3 p-2 rounded-lg hover:bg-stone-100/80 active:bg-stone-200/50 transition-colors cursor-pointer min-h-[44px]"
          >
            <input
              type="checkbox"
              id="enableRepair"
              checked={enableRepair}
              onChange={(e) => setEnableRepair(e.target.checked)}
              className="w-4 h-4 sm:w-5 sm:h-5 text-violet-500 rounded border-stone-300 focus:ring-violet-500 cursor-pointer mt-0.5 sm:mt-0 shrink-0"
            />
            <div className="flex items-center gap-2 min-w-0">
              <ShieldCheck className="w-4 h-4 text-violet-500 shrink-0" />
              <span className="font-medium text-xs sm:text-sm select-none">
                Repair Damage (Scratches, White Spots, Cracks)
              </span>
            </div>
          </label>

          <div className="w-full h-px bg-stone-200/70" />

          {/* Denoise */}
          <label
            htmlFor="enableDenoise"
            className="flex items-start sm:items-center gap-3 p-2 rounded-lg hover:bg-stone-100/80 active:bg-stone-200/50 transition-colors cursor-pointer min-h-[44px]"
          >
            <input
              type="checkbox"
              id="enableDenoise"
              checked={enableDenoise}
              onChange={(e) => setEnableDenoise(e.target.checked)}
              className="w-4 h-4 sm:w-5 sm:h-5 text-rose-500 rounded border-stone-300 focus:ring-rose-500 cursor-pointer mt-0.5 sm:mt-0 shrink-0"
            />
            <div className="flex items-center gap-2 min-w-0">
              <ShieldCheck className="w-4 h-4 text-rose-500 shrink-0" />
              <span className="font-medium text-xs sm:text-sm select-none">
                Remove Dust &amp; Scratches (Denoise)
              </span>
            </div>
          </label>

          <div className="w-full h-px bg-stone-200/70" />

          {/* AI Colorize */}
          <label
            htmlFor="enableColorize"
            className="flex items-start sm:items-center gap-3 p-2 rounded-lg hover:bg-stone-100/80 active:bg-stone-200/50 transition-colors cursor-pointer min-h-[44px]"
          >
            <input
              type="checkbox"
              id="enableColorize"
              checked={enableColorize}
              onChange={(e) => setEnableColorize(e.target.checked)}
              className="w-4 h-4 sm:w-5 sm:h-5 text-indigo-500 rounded border-stone-300 focus:ring-indigo-500 cursor-pointer mt-0.5 sm:mt-0 shrink-0"
            />
            <div className="flex items-center gap-2 min-w-0">
              <ImageIcon className="w-4 h-4 text-indigo-500 shrink-0" />
              <span className="font-medium text-xs sm:text-sm select-none">
                AI Colorization (For B&amp;W photos)
              </span>
            </div>
          </label>

          <div className="w-full h-px bg-stone-200/70" />

          {/* Re-imagine lost regions (beta) */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-2 rounded-lg hover:bg-stone-100/80 active:bg-stone-200/50 transition-colors">
            <label
              htmlFor="enableReimagine"
              className="flex items-start sm:items-center gap-3 cursor-pointer min-h-[36px] flex-1 min-w-0"
            >
              <input
                type="checkbox"
                id="enableReimagine"
                checked={enableReimagine}
                onChange={(e) => setEnableReimagine(e.target.checked)}
                className="w-4 h-4 sm:w-5 sm:h-5 text-amber-500 rounded border-stone-300 focus:ring-amber-500 cursor-pointer mt-0.5 sm:mt-0 shrink-0"
              />
              <div className="flex items-center gap-2 min-w-0">
                <Sparkles className="w-4 h-4 text-amber-500 shrink-0" />
                <span className="font-medium text-xs sm:text-sm select-none">
                  Re-imagine lost regions (beta)
                </span>
              </div>
            </label>
            <span
              className="text-[11px] text-stone-500 bg-amber-50/80 border border-amber-200 px-2 py-1 rounded leading-normal sm:max-w-xs"
              title="Opt-in beta. When damage is too extensive to repair (≥35% of the photo or over half a face), AI re-imagines only the destroyed regions while every surviving pixel is preserved exactly and disclosed in the Restoration Report."
            >
              For extreme damage (&ge;35% destroyed)
            </span>
          </div>

          <div className="w-full h-px bg-stone-200/70" />

          {/* Device Preference */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 w-full justify-between pt-2">
            <span className="font-semibold text-xs text-stone-600 uppercase tracking-wider flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-stone-500" /> Device:
            </span>
            <div className="inline-flex rounded-lg border border-stone-200 bg-stone-100 p-0.5 shadow-2xs text-xs font-semibold w-full sm:w-auto justify-between sm:justify-start">
              {DEVICE_CHOICES.map((dev) => (
                <button
                  key={dev.id}
                  type="button"
                  onClick={() => setSelectedDevice(dev.id)}
                  className={`flex-1 sm:flex-none px-4 py-2 rounded-md transition-all cursor-pointer min-h-[38px] flex items-center justify-center ${
                    selectedDevice === dev.id
                      ? 'bg-white text-stone-900 shadow-xs border border-stone-200/50'
                      : 'text-stone-600 hover:text-stone-900 active:bg-stone-200/60'
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
