import Link from "next/link";

import { HeroSection } from "@/components/hero-section";
import { Playground } from "@/components/playground";
import { EvaluationShowcase } from "@/components/evaluation-showcase";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      <HeroSection />

      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-16 px-6 py-12">
        <section className="grid gap-8 lg:grid-cols-[1.3fr_1fr]">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-8 shadow-xl backdrop-blur">
            <h2 className="text-sm uppercase tracking-[0.35em] text-emerald-300">Bối cảnh</h2>
            <h3 className="mt-3 text-3xl font-semibold text-white">
              Ba biểu diễn, một bài toán: phân loại văn bản tiếng Việt.
            </h3>
            <p className="mt-4 text-base leading-relaxed text-slate-300">
              Bộ demo này cho phép bạn tương tác với ba cách biểu diễn phổ biến trong NLP – TF-IDF, Word2Vec
              và Doc2Vec – kết hợp các bộ phân loại (Rocchio, KNN, Naive Bayes). Mỗi lựa chọn mang tới góc
              nhìn khác nhau về cấu trúc và ngữ nghĩa của tài liệu.
            </p>
            <ul className="mt-6 space-y-3 text-sm text-slate-200">
              <li className="flex items-start gap-3">
                <span className="mt-1 inline-flex h-6 w-6 items-center justify-center rounded-full bg-emerald-400/20 text-emerald-300">
                  1
                </span>
                <span>
                  <strong>TF-IDF</strong> nhấn mạnh những từ đặc trưng, phù hợp với các mô hình tuyến tính như Naive Bayes hoặc Rocchio.
                </span>
              </li>
              <li className="flex items-start gap-3">
                <span className="mt-1 inline-flex h-6 w-6 items-center justify-center rounded-full bg-sky-400/20 text-sky-200">
                  2
                </span>
                <span>
                  <strong>Word2Vec</strong> bắt ngữ nghĩa từ mức từ vựng, giúp đo tương đồng khi dùng KNN hay centroid-based.
                </span>
              </li>
              <li className="flex items-start gap-3">
                <span className="mt-1 inline-flex h-6 w-6 items-center justify-center rounded-full bg-violet-500/20 text-violet-200">
                  3
                </span>
                <span>
                  <strong>Doc2Vec</strong> tạo vector tài liệu đầy đủ, hữu ích khi muốn gom cụm theo ngữ cảnh toàn bài.
                </span>
              </li>
            </ul>
          </div>

          <aside className="rounded-2xl border border-white/10 bg-white/5 p-8 shadow-xl backdrop-blur">
            <h3 className="text-xl font-semibold text-white">Tài nguyên mở rộng</h3>
            <p className="mt-2 text-sm text-slate-300">
              Nếu bạn muốn đọc sâu hơn về cách hoạt động của các biểu diễn, hãy tham khảo các tài liệu nền tảng dưới đây.
            </p>
            <ul className="mt-6 space-y-4">
              <li>
                <Link
                  href="https://medium.com/@hieptlq/t%C3%ACm-hi%E1%BB%83u-tf-idf-trong-x%E1%BB%AD-l%C3%BD-ng%C3%B4n-ng%E1%BB%AF-t%E1%BB%B1-nhi%C3%AAn-4b0ca5ed2025"
                  target="_blank"
                  className="group block rounded-lg border border-white/10 px-4 py-3 transition hover:border-emerald-300/60 hover:bg-emerald-400/10"
                >
                  <span className="font-medium text-emerald-200 group-hover:text-emerald-100">TF-IDF fundamentals</span>
                  <span className="block text-xs text-slate-400">Cách cân bằng tần suất từ và độ hiếm của chúng trong corpus.</span>
                </Link>
              </li>
              <li>
                <Link
                  href="https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html"
                  target="_blank"
                  className="group block rounded-lg border border-white/10 px-4 py-3 transition hover:border-sky-300/60 hover:bg-sky-400/10"
                >
                  <span className="font-medium text-sky-200 group-hover:text-sky-100">Huấn luyện Word2Vec với gensim</span>
                  <span className="block text-xs text-slate-400">Bước vào thế giới embedding từ với Skip-gram &amp; CBOW.</span>
                </Link>
              </li>
              <li>
                <Link
                  href="https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html"
                  target="_blank"
                  className="group block rounded-lg border border-white/10 px-4 py-3 transition hover:border-violet-300/60 hover:bg-violet-500/10"
                >
                  <span className="font-medium text-violet-200 group-hover:text-violet-100">Doc2Vec & case study</span>
                  <span className="block text-xs text-slate-400">Xây vector tài liệu và khai thác trong các bài toán phân loại.</span>
                </Link>
              </li>
            </ul>
          </aside>
        </section>

        <EvaluationShowcase />

        <Playground />
      </main>

      <footer className="border-t border-white/5 bg-black/20 py-6 text-center text-xs text-slate-500">
        <p>
          Demo được phát triển để trực quan hoá pipeline từ notebook IM-Final sang ứng dụng web. Backend inference sử dụng FastAPI + các
          mô hình đã huấn luyện offline.
        </p>
      </footer>
    </div>
  );
}
