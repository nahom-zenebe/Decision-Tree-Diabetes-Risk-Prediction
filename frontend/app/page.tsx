"use client";
import Link from 'next/link';
import { Activity, ShieldCheck, ArrowRight, ChevronRight, HeartPulse } from 'lucide-react';
import { useRouter } from 'next/navigation';

export default function Home() {

   const router = useRouter();

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-blue-100 via-slate-50 to-white">
      {/* Decorative background elements */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-72 h-72 bg-blue-200/30 rounded-full blur-3xl"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-cyan-100/40 rounded-full blur-3xl"></div>
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-6 py-16">
        {/* Header Section */}
        <header className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-50 border border-blue-100 text-blue-600 text-sm font-medium mb-6 animate-fade-in">
            <Activity size={16} />
            <span>AI-Powered Health Insights</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-extrabold text-slate-100 tracking-tight mb-6">
            Diabetes <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-cyan-500">Prediction</span> System
          </h1>
          <p className="text-lg text-slate-300 max-w-2xl mx-auto leading-relaxed">
          The prediction system uses advanced machine learning algorithms to analyze multiple health 
              parameters and calculate your diabetes risk score. The model considers factors like age, BMI, 
              blood glucose levels, and lifestyle habits to provide accurate predictions.
          </p>
        </header>

        
              <button onClick={()=>router.push("/Home")} className="group ml-80 mt-40 w-96 bg-blue-500 hover:bg-blue-400 text-white font-bold py-4 px-8 rounded-2xl flex items-center justify-center gap-2 transition-all transform active:scale-95">
                Start Risk Assessment
                <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
              </button>
           

        {/* Footer Note */}
        <footer className="text-center mt-96 text-slate-400 text-sm">
          <p>© 2024 HealthTech Prediction Systems. For informational purposes only.</p>
        </footer>
      </div>
    </div>
  );
}