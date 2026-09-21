import ImageUploader from '../components/ImageUploader';

export default function Home() {
  return (
    <main className="flex-1 flex flex-col items-center justify-center p-8 bg-[#faf9f6]">
      <div className="text-center max-w-3xl mx-auto mb-16 mt-8">
        <h1 className="text-4xl md:text-5xl font-serif text-stone-900 mb-6 leading-tight">
          Professional Image Restoration
        </h1>
        <p className="text-lg text-stone-600 font-light max-w-xl mx-auto">
          Enhance and restore your archival, formal, or degraded photographs with state-of-the-art artificial intelligence.
        </p>
      </div>
      
      <div className="w-full px-4 md:px-12 lg:px-24 mb-12">
        <ImageUploader />
      </div>
    </main>
  );
}
