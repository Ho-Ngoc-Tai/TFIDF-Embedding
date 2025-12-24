"use client";

import { useMemo, useState } from "react";

type ApiPipelineResult = {
    representation: string;
    classifier: string;
    prediction: string;
    confidence: number;
    extra?: {
        top_tokens?: {
            token: string;
            weight: number;
        }[];
    } | null;
};

type ApiResponse = {
    input_length: number;
    tokens_preview: string[];
    results: ApiPipelineResult[];
};

const PRESET_SAMPLES: { title: string; description: string; text: string }[] = [
    {
        title: "Bản tin giáo dục",
        description: "Thông tin về kỳ thi và tuyển sinh đại học",
        text: "Bộ Giáo dục vừa công bố lịch thi đánh giá năng lực năm 2026 với sự tham gia của hơn 14.000 thí sinh tại ba miền. Các trường đại học dự kiến sử dụng kết quả này để xét tuyển song song với điểm thi tốt nghiệp.",
    },
    {
        title: "Thị trường kinh doanh",
        description: "Diễn biến lợi nhuận và đầu tư",
        text: "Lợi nhuận quý IV của các ngân hàng thương mại tăng trưởng hai con số nhờ tín dụng phục hồi và biên lãi ròng mở rộng. Một số doanh nghiệp bất động sản công bố kế hoạch phát hành trái phiếu để tái cơ cấu dòng tiền.",
    },
    {
        title: "Sức khỏe & đời sống",
        description: "Tin về y tế cộng đồng",
        text: "Sở Y tế TP HCM cảnh báo số ca sốt xuất huyết tăng 15% so với cùng kỳ. Ngành y tế khuyến nghị người dân vệ sinh môi trường, loại bỏ lăng quăng và đến cơ sở y tế khi có dấu hiệu sốt kéo dài.",
    },
];

const FORMAT_PERCENT = new Intl.NumberFormat("vi-VN", {
    style: "percent",
    maximumFractionDigits: 1,
});

const prettyLabel = (slug: string) => slug.replace(/-/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());

function groupByRepresentation(results: ApiPipelineResult[]) {
    return results.reduce<Record<string, ApiPipelineResult[]>>((acc, item) => {
        acc[item.representation] ??= [];
        acc[item.representation].push(item);
        return acc;
    }, {});
}

export function Playground() {
    const [text, setText] = useState(PRESET_SAMPLES[0].text);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [response, setResponse] = useState<ApiResponse | null>(null);

    const groupedResults = useMemo(() => {
        if (!response?.results) return {};
        return groupByRepresentation(response.results);
    }, [response]);

    const handleSubmit = async () => {
        if (!text.trim()) {
            setError("Vui lòng nhập nội dung văn bản.");
            return;
        }

        setIsLoading(true);
        setError(null);
        setResponse(null);

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ text }),
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data?.error ?? "Không thể phân tích văn bản.");
            }

            setResponse(data as ApiResponse);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Có lỗi xảy ra.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleUsePreset = (sampleText: string) => {
        setText(sampleText);
        setResponse(null);
        setError(null);
    };

    return (
        <section className="rounded-3xl border border-white/10 bg-black/30 p-8 shadow-[0_0_60px_rgba(16,185,129,0.1)] backdrop-blur">
            <header className="flex flex-col gap-3 pb-6">
                <h2 className="text-2xl font-semibold text-white">Playground thử nghiệm</h2>
                <p className="max-w-3xl text-sm text-slate-300">
                    Nhập một đoạn văn bản tiếng Việt hoặc chọn từ danh sách mẫu bên dưới. Hệ thống sẽ chạy đồng thời ba pipeline và trả về
                    nhãn dự đoán, độ tự tin và (đối với TF-IDF) những từ khoá đóng góp lớn nhất.
                </p>
            </header>

            <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
                <div className="space-y-6">
                    <div>
                        <label className="mb-2 block text-sm font-semibold uppercase tracking-[0.2em] text-emerald-200">
                            Văn bản đầu vào
                        </label>
                        <textarea
                            value={text}
                            onChange={(event) => setText(event.target.value)}
                            rows={12}
                            className="w-full rounded-2xl border border-white/10 bg-slate-950/70 p-4 text-sm text-slate-100 outline-none transition focus:border-emerald-300 focus:ring-2 focus:ring-emerald-300/40"
                            placeholder="Nhập văn bản cần phân loại..."
                        />
                        <div className="mt-2 text-xs text-slate-500">Độ dài: {text.length} ký tự</div>
                    </div>

                    <div className="flex flex-wrap gap-3">
                        {PRESET_SAMPLES.map((sample) => (
                            <button
                                key={sample.title}
                                type="button"
                                onClick={() => handleUsePreset(sample.text)}
                                className="group flex-1 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-left transition hover:border-emerald-300/60 hover:bg-emerald-400/10"
                            >
                                <span className="block text-sm font-semibold text-white group-hover:text-emerald-100">
                                    {sample.title}
                                </span>
                                <span className="mt-1 block text-xs text-slate-400">{sample.description}</span>
                            </button>
                        ))}
                    </div>

                    <button
                        type="button"
                        onClick={handleSubmit}
                        disabled={isLoading}
                        className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-emerald-400 to-sky-500 px-6 py-3 text-sm font-semibold text-slate-900 transition hover:from-emerald-300 hover:to-sky-400 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                        {isLoading ? "Đang phân tích..." : "Chạy phân loại"}
                    </button>

                    {error && (
                        <div className="rounded-2xl border border-red-400/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                            {error}
                        </div>
                    )}
                </div>

                <div className="rounded-2xl border border-white/10 bg-slate-950/60 p-6 text-sm text-slate-100">
                    {!response && !isLoading && (
                        <div className="flex h-full flex-col items-center justify-center gap-3 text-center text-slate-400">
                            <span className="text-2xl">🪄</span>
                            <p>Chạy thử một văn bản để xem kết quả dự đoán ở đây.</p>
                        </div>
                    )}

                    {isLoading && (
                        <div className="flex h-full flex-col items-center justify-center gap-3 text-center text-slate-400">
                            <span className="animate-spin text-2xl">⏳</span>
                            <p>Đang gửi tới FastAPI inference...</p>
                        </div>
                    )}

                    {response && (
                        <div className="space-y-6">
                            <div>
                                <h3 className="text-base font-semibold text-white">Tokens preview</h3>
                                <p className="mt-1 text-xs text-slate-400">
                                    {response.tokens_preview.length > 0
                                        ? response.tokens_preview.join(" · ")
                                        : "Không có token nào sau khi xử lý."}
                                </p>
                            </div>

                            {Object.entries(groupedResults).map(([representation, rows]) => (
                                <div key={representation} className="rounded-xl border border-white/10 bg-white/5">
                                    <div className="border-b border-white/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-emerald-200">
                                        {representation}
                                    </div>
                                    <ul className="divide-y divide-white/10">
                                        {rows.map((row) => (
                                            <li key={`${representation}-${row.classifier}`} className="px-4 py-3">
                                                <div className="flex items-center justify-between">
                                                    <span className="text-sm font-semibold text-white">{row.classifier}</span>
                                                    <span className="text-xs uppercase tracking-[0.2em] text-slate-400">
                                                        {prettyLabel(row.prediction)}
                                                    </span>
                                                </div>
                                                <div className="mt-2 flex items-baseline justify-between text-xs text-slate-300">
                                                    <span>Confidence</span>
                                                    <span className="rounded-full bg-emerald-400/10 px-2 py-0.5 text-emerald-200">
                                                        {FORMAT_PERCENT.format(row.confidence)}
                                                    </span>
                                                </div>

                                                {row.extra?.top_tokens && row.extra.top_tokens.length > 0 && (
                                                    <div className="mt-2 text-xs text-slate-400">
                                                        <span className="font-semibold text-slate-300">Top tokens:</span>{" "}
                                                        {row.extra.top_tokens
                                                            .map((token) => `${token.token} (${token.weight.toFixed(2)})`)
                                                            .join(", ")}
                                                    </div>
                                                )}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </section>
    );
}
