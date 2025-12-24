import evaluation from "@/data/evaluation-results.json";

const METRIC_LABELS: Record<"accuracy" | "precision_macro" | "recall_macro" | "f1_macro", string> = {
    accuracy: "Accuracy",
    precision_macro: "Precision (macro)",
    recall_macro: "Recall (macro)",
    f1_macro: "F1 (macro)",
};

const FORMAT_PERCENT = new Intl.NumberFormat("vi-VN", {
    style: "percent",
    maximumFractionDigits: 1,
});

const prettyLabel = (slug: string) => slug.replace(/-/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());

type ResultEntry = (typeof evaluation)["results"][number];

const groupByRepresentation = (entries: ResultEntry[]) => {
    return entries.reduce<Record<string, ResultEntry[]>>((acc, entry) => {
        acc[entry.representation] ??= [];
        acc[entry.representation].push(entry);
        return acc;
    }, {});
};

export function EvaluationShowcase() {
    const grouped = groupByRepresentation(evaluation.results);

    return (
        <section className="rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/70 via-slate-900/40 to-black/30 p-8 shadow-2xl backdrop-blur">
            <header className="flex flex-col gap-2 pb-6">
                <p className="text-xs uppercase tracking-[0.35em] text-slate-400">Offline evaluation</p>
                <h2 className="text-2xl font-semibold text-white">Hiệu năng trên tập kiểm thử</h2>
                <p className="max-w-2xl text-sm leading-relaxed text-slate-300">
                    Huấn luyện trên {evaluation.data_summary.total_samples.toLocaleString("vi-VN")} bài báo thuộc
                    {" "}
                    {evaluation.data_summary.categories.length} chuyên mục. Bảng dưới đây thể hiện các chỉ số macro-average
                    cho từng tổ hợp biểu diễn và thuật toán.
                </p>
            </header>

            <div className="grid gap-6 md:grid-cols-[1fr_1.2fr]">
                <div className="rounded-2xl border border-white/5 bg-white/5 p-6">
                    <h3 className="text-lg font-semibold text-white">Phân bố nhãn</h3>
                    <ul className="mt-4 space-y-2 text-sm text-slate-200">
                        {Object.entries(evaluation.data_summary.labels).map(([label, count]) => (
                            <li key={label} className="flex items-center justify-between">
                                <span className="font-medium text-slate-300">{prettyLabel(label)}</span>
                                <span className="tabular-nums text-slate-100">{count.toLocaleString("vi-VN")}</span>
                            </li>
                        ))}
                    </ul>
                </div>

                <div className="overflow-x-auto rounded-2xl border border-white/5 bg-white/5">
                    {Object.entries(grouped).map(([representation, rows]) => (
                        <div key={representation} className="border-b border-white/5 last:border-b-0">
                            <div className="bg-white/10 px-6 py-3 text-sm font-semibold uppercase tracking-[0.2em] text-emerald-200">
                                {representation}
                            </div>
                            <table className="min-w-full divide-y divide-white/5 text-sm text-slate-100">
                                <thead className="bg-white/5 text-xs uppercase tracking-wide text-slate-400">
                                    <tr>
                                        <th scope="col" className="px-4 py-3 text-left">Classifier</th>
                                        {Object.values(METRIC_LABELS).map((label) => (
                                            <th key={label} scope="col" className="px-4 py-3 text-right">
                                                {label}
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-white/5">
                                    {rows.map((row) => (
                                        <tr key={`${row.representation}-${row.classifier}`} className="hover:bg-white/5">
                                            <td className="px-4 py-3 font-medium text-white">{row.classifier}</td>
                                            <td className="px-4 py-3 text-right text-slate-200">
                                                {FORMAT_PERCENT.format(row.accuracy)}
                                            </td>
                                            <td className="px-4 py-3 text-right text-slate-200">
                                                {FORMAT_PERCENT.format(row.precision_macro)}
                                            </td>
                                            <td className="px-4 py-3 text-right text-slate-200">
                                                {FORMAT_PERCENT.format(row.recall_macro)}
                                            </td>
                                            <td className="px-4 py-3 text-right text-emerald-200">
                                                {FORMAT_PERCENT.format(row.f1_macro)}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
