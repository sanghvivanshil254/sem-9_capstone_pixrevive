import ImageUploader from '../components/ImageUploader';

export default function Home() {
  return (
    <main className="flex-1 flex flex-col items-center justify-start sm:justify-center px-3 sm:px-6 md:px-8 lg:px-12 py-4 sm:py-8 md:py-12 bg-[#faf9f6] w-full max-w-7xl mx-auto">
      {/* Hero Header */}
      <div className="text-center w-full max-w-3xl mx-auto mt-2 sm:mt-6 mb-6 sm:mb-10 md:mb-14">
        <h1 className="text-2xl sm:text-4xl md:text-5xl font-serif text-stone-900 mb-3 sm:mb-5 leading-tight tracking-tight px-2">
          Professional Image Restoration
        </h1>
        <p className="text-sm sm:text-base md:text-lg text-stone-600 font-light max-w-xl mx-auto leading-relaxed px-4">
          Enhance and restore your archival, formal, or degraded photographs with state-of-the-art artificial intelligence.
        </p>
      </div>

      {/* Main Workspace Uploader */}
      <div className="w-full max-w-5xl mx-auto mb-6 sm:mb-10">
        <ImageUploader />
      </div>
    </main>
  );
}
