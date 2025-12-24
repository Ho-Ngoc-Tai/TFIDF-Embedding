export function HeroSection() {
    return (
        <header className="relative overflow-hidden border-b border-white/10 bg-black/40">
            <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.2),_transparent_55%)]" />
            <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 py-16 text-slate-100">
                <p className="text-sm uppercase tracking-[0.4em] text-emerald-300">Vietnamese NLP Playground</p>
                <h1 className="max-w-3xl text-4xl font-semibold leading-tight md:text-5xl">
                    Trực quan pipeline phân loại văn bản với TF-IDF, Word2Vec và Doc2Vec
                </h1>
                <p className="max-w-2xl text-base leading-relaxed text-slate-300">
                    Ứng dụng này chuyển đổi notebook IM-Final thành trải nghiệm tương tác. Bạn có thể nhập văn bản của mình,
                    kiểm tra kết quả dự đoán của từng pipeline và so sánh hiệu suất đã được huấn luyện trên tập tin tức tiếng Việt.
                </p>
                <div className="flex flex-wrap items-center gap-4 text-sm">
                    <div className="rounded-full border border-emerald-300/60 bg-emerald-400/10 px-4 py-1 text-emerald-200">
                        FastAPI inference
                    </div>
                    <div className="rounded-full border border-sky-300/60 bg-sky-400/10 px-4 py-1 text-sky-200">
                        Next.js App Router
                    </div>
                    <div className="rounded-full border border-violet-300/60 bg-violet-500/10 px-4 py-1 text-violet-200">
                        Tailwind-driven UI
                    </div>
                </div>
            </div>
        </header>
    );
}
