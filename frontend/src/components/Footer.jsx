export default function Footer() {
  return (
    <footer className="w-full bg-transparent border-t border-stone-200 text-stone-500 pt-6 sm:pt-8 pb-[calc(1.5rem+env(safe-area-inset-bottom,0px))] px-4 text-center text-xs sm:text-sm">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-stone-500">
        <p>&copy; {new Date().getFullYear()} PixRevive AI. All rights reserved.</p>
        <p className="text-[11px] sm:text-xs text-stone-400">Local inference &bull; Privacy-first restoration</p>
      </div>
    </footer>
  );
}
