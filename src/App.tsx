import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type Dataset = { rows: number; columns: string[]; inputColumns: string[]; targetColumns: string[]; warnings: string[]; metadata?: { s11_points?: number; frequency_labels_verified?: boolean; frequency_available?: boolean }; frequencyHz?: number[]; preview: { rows: Record<string, unknown>[] } };
type Evaluation = { model: string; r2: number; rmse: number; mae: number; explained_variance?: number };
type Optimization = { algorithm: string; best_fitness: number; best_position: number[]; history: number[]; elapsed_seconds: number };
type AppState = { dataset: Dataset | null; models: string[]; evaluation: Evaluation[] | null; optimizations: Record<string, Optimization> };

const emptyState: AppState = { dataset: null, models: [], evaluation: null, optimizations: {} };
const navItems = ["Dashboard", "Dataset", "Machine Learning", "Prediction", "Optimization", "HFSS Validation", "Reports"];

async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `API request failed (${response.status} ${response.statusText}).`);
  }
  return response.headers.get("content-type")?.includes("application/json") ? response.json() : response.blob() as T;
}

function App() {
  const [page, setPage] = useState("Dashboard");
  const [state, setState] = useState<AppState>(emptyState);
  const [error, setError] = useState("");
  const refresh = () => api<AppState>("/api/state").then(setState).catch((e) => setError(e.message));
  useEffect(() => { refresh(); }, []);
  const context = { state, setState, setError, refresh };

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">∿</span><div><strong>Antenna Lab</strong><small>Research workspace</small></div></div>
      <nav>{navItems.map((item) => <button className={page === item ? "nav-item active" : "nav-item"} onClick={() => setPage(item)} key={item}><span>{iconFor(item)}</span>{item}</button>)}</nav>
      <div className="sidebar-note">Computed from uploaded data only.<br />No fabricated simulation results.</div>
    </aside>
    <main className="main">
      <header className="topbar"><div><span className="eyebrow">ENGINEERING WORKSPACE</span><h1>{page}</h1></div><div className="status"><span className="status-dot" /> API connected</div></header>
      {error && <div className="alert error">{error}<button onClick={() => setError("")}>Dismiss</button></div>}
      {page === "Dashboard" && <Dashboard {...context} />}
      {page === "Dataset" && <DatasetPage {...context} />}
      {page === "Machine Learning" && <MlPage {...context} />}
      {page === "Prediction" && <PredictionPage {...context} />}
      {page === "Optimization" && <OptimizationPage {...context} />}
      {page === "HFSS Validation" && <HfssPage {...context} />}
      {page === "Reports" && <ReportsPage {...context} />}
    </main>
  </div>;
}

function iconFor(page: string) {
  return ({ Dashboard: "⌂", Dataset: "▦", "Machine Learning": "◇", Prediction: "⌁", Optimization: "✦", "HFSS Validation": "◉", Reports: "▤" } as Record<string, string>)[page];
}

type PageProps = { state: AppState; setState: (s: AppState) => void; setError: (s: string) => void; refresh: () => void };
function Empty({ children }: { children: string }) { return <div className="empty"><div className="empty-icon">◌</div><h3>{children}</h3><p>Upload data and train a model to unlock this workspace.</p></div>; }
function Dashboard({ state }: PageProps) {
  if (!state.dataset) return <><Hero /><Empty>Ready for your antenna dataset</Empty></>;
  const best = state.evaluation?.[0];
  return <><Hero /><div className="metric-grid"><Metric label="Samples" value={state.dataset.rows.toLocaleString()} /><Metric label="Input parameters" value={state.dataset.inputColumns.length} /><Metric label="S11 points" value={state.dataset.metadata?.s11_points ?? state.dataset.targetColumns.length} /><Metric label="Trained models" value={state.models.length} /></div>
    <section className="card"><SectionTitle title="Validation overview" detail={best ? `Best model: ${best.model}` : "Train models to populate actual metrics"} />
      {state.dataset.metadata?.frequency_labels_verified === false && <div className="alert warning">Frequency labels are not present in the imported files; resonant frequency and bandwidth remain unavailable until frequency metadata is supplied.</div>}
      {state.evaluation ? <><div className="table-wrap"><EvaluationTable rows={state.evaluation} /></div><MetricChart rows={state.evaluation} /></> : <EmptyState text="No model evaluation available yet." />}</section></>;
}
function Hero() { return <section className="hero"><div><span className="eyebrow accent">MICROSTRIP PATCH ANTENNA</span><h2>Design with evidence.<br /><em>Optimize with confidence.</em></h2><p>Load measured data, train performance models, and explore bounded optimization experiments.</p></div><div className="hero-wave">∿<br />∿<br />∿</div></section>; }
function Metric({ label, value }: { label: string; value: string | number }) { return <div className="metric"><span>{label}</span><strong>{value}</strong><small>LIVE DATA</small></div>; }
function SectionTitle({ title, detail }: { title: string; detail: string }) { return <div className="section-title"><div><h3>{title}</h3><p>{detail}</p></div></div>; }
function EmptyState({ text }: { text: string }) { return <div className="empty-state">{text}</div>; }
function EvaluationTable({ rows }: { rows: Evaluation[] }) { return <table><thead><tr><th>Model</th><th>R² score</th><th>Explained variance</th><th>RMSE</th><th>MAE</th></tr></thead><tbody>{rows.map((row) => <tr key={row.model}><td><b>{row.model}</b></td><td className="positive">{row.r2.toFixed(4)}</td><td>{row.explained_variance?.toFixed(4) ?? "—"}</td><td>{row.rmse.toFixed(4)}</td><td>{row.mae.toFixed(4)}</td></tr>)}</tbody></table>; }
function MetricChart({ rows }: { rows: Evaluation[] }) { return <div className="chart"><ResponsiveContainer width="100%" height={230}><LineChart data={rows}><CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" /><XAxis dataKey="model" tick={{ fontSize: 11 }} /><YAxis tick={{ fontSize: 11 }} /><Tooltip /><Line type="monotone" dataKey="r2" stroke="#16a394" strokeWidth={3} dot={{ r: 4 }} /></LineChart></ResponsiveContainer></div>; }

function DatasetPage({ state, setState, setError }: PageProps) {
  const [files, setFiles] = useState<{ inputs?: File; real?: File; imag?: File; frequency?: File }>({});
  const [frequency, setFrequency] = useState({ start: "", stop: "", count: "" });
  const submit = async (event: FormEvent) => { event.preventDefault(); if (!files.inputs) return setError("Choose an input parameter file first."); const form = new FormData(); form.append("inputs", files.inputs); if (files.real) form.append("s11_real", files.real); if (files.imag) form.append("s11_imag", files.imag); if (files.frequency) form.append("frequency", files.frequency); if (frequency.start) form.append("frequency_start", frequency.start); if (frequency.stop) form.append("frequency_stop", frequency.stop); if (frequency.count) form.append("frequency_count", frequency.count); try { setState(await api<AppState>("/api/dataset", { method: "POST", body: form })); } catch (e) { setError((e as Error).message); } };
  const upload = (key: "inputs" | "real" | "imag" | "frequency") => (event: ChangeEvent<HTMLInputElement>) => setFiles((current) => ({ ...current, [key]: event.target.files?.[0] }));
  return <><Intro title="Dataset intake" text="Inspect your measured antenna data before making any modeling assumptions." /><form className="card upload-grid" onSubmit={submit}><FileBox title="Input parameters" required file={files.inputs} onChange={upload("inputs")} /><FileBox title="S11 real component" file={files.real} onChange={upload("real")} /><FileBox title="S11 imaginary component" file={files.imag} onChange={upload("imag")} /><FileBox title="Frequency metadata (optional)" file={files.frequency} onChange={upload("frequency")} /><div className="form-row"><label>Frequency start (GHz)<input type="number" value={frequency.start} onChange={(e) => setFrequency({ ...frequency, start: e.target.value })} /></label><label>Frequency stop (GHz)<input type="number" value={frequency.stop} onChange={(e) => setFrequency({ ...frequency, stop: e.target.value })} /></label><label>Points<input type="number" min="2" value={frequency.count} onChange={(e) => setFrequency({ ...frequency, count: e.target.value })} /></label></div><button className="primary full" type="submit">Load and inspect dataset <span>→</span></button></form>
    {state.dataset && <section className="card"><SectionTitle title="Processed preview" detail={`${state.dataset.rows.toLocaleString()} samples · ${state.dataset.columns.length} columns`} />{state.dataset.warnings.map((w) => <div className="alert warning" key={w}>{w}</div>)}<Preview data={state.dataset.preview.rows} /></section>}</>;
}
function FileBox({ title, required, file, onChange }: { title: string; required?: boolean; file?: File; onChange: (e: ChangeEvent<HTMLInputElement>) => void }) { return <label className="file-box"><span>{title}{required && <i> REQUIRED</i>}</span><div className="file-input">{file ? `✓ ${file.name}` : "Choose CSV or XLSX"}<input type="file" accept=".csv,.xlsx,.xls" required={required} onChange={onChange} /></div></label>; }
function Preview({ data }: { data: Record<string, unknown>[] }) { if (!data.length) return <EmptyState text="The dataset has no rows." />; const columns = Object.keys(data[0]); return <div className="table-wrap"><table><thead><tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr></thead><tbody>{data.slice(0, 8).map((row, i) => <tr key={i}>{columns.map((c) => <td key={c}>{String(row[c] ?? "—")}</td>)}</tr>)}</tbody></table></div>; }
function Intro({ title, text }: { title: string; text: string }) { return <div className="intro"><span className="eyebrow accent">WORKFLOW</span><h2>{title}</h2><p>{text}</p></div>; }

function MlPage({ state, setState, setError }: PageProps) {
  const [features, setFeatures] = useState<string[]>(state.dataset?.inputColumns ?? []);
  const [targets, setTargets] = useState<string[]>(state.dataset?.targetColumns ?? []);
  const [training, setTraining] = useState(false);
  if (!state.dataset) return <Empty>Load a dataset first</Empty>;
  const numeric = state.dataset.columns;
  const train = async () => { setTraining(true); try { setState(await api<AppState>("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features, targets }) })); } catch (e) { setError((e as Error).message); } finally { setTraining(false); } };
  return <><Intro title="Model training" text="Select numeric inputs and measured outputs. Validation uses an 80:20 holdout split. Wide S11 curves are compressed during fitting and reconstructed to all selected points." /><div className="two-col"><ColumnPicker title="Input features" options={numeric} selected={features} setSelected={setFeatures} /><ColumnPicker title="Prediction targets" options={numeric} selected={targets} setSelected={setTargets} /></div><button className="primary" disabled={training} onClick={train}>{training ? "Training models…" : "Train four models"} <span>→</span></button>{state.evaluation && <section className="card"><SectionTitle title="Validation results" detail="Metrics from held-out data, not training scores." /><EvaluationTable rows={state.evaluation} /><MetricChart rows={state.evaluation} /></section>}</>;
}
function ColumnPicker({ title, options, selected, setSelected }: { title: string; options: string[]; selected: string[]; setSelected: (s: string[]) => void }) { return <section className="card picker"><h3>{title}</h3>{options.map((option) => <label key={option}><input type="checkbox" checked={selected.includes(option)} onChange={(e) => setSelected(e.target.checked ? [...selected, option] : selected.filter((item) => item !== option))} /> {option}</label>)}</section>; }

function PredictionPage({ state, setError }: PageProps) {
  const [model, setModel] = useState(""); const [values, setValues] = useState<Record<string, number>>({}); const [result, setResult] = useState<{ prediction: Record<string, number>; metrics: Record<string, unknown> } | null>(null); const [warnings, setWarnings] = useState<string[]>([]);
  const selected = model || state.models[0] || ""; const features = state.dataset?.inputColumns ?? [];
  useEffect(() => { if (!model && state.models[0]) setModel(state.models[0]); }, [model, state.models]);
  if (!state.models.length || !state.dataset) return <Empty>Train a model first</Empty>;
  const submit = async () => { try { const response = await api<{ prediction: Record<string, number>; metrics: Record<string, unknown>; warnings: string[] }>("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: selected, values }) }); setResult({ prediction: response.prediction, metrics: response.metrics }); setWarnings(response.warnings); } catch (e) { setError((e as Error).message); } };
  return <><Intro title="Performance prediction" text="Evaluate a candidate design using the selected trained model." /><section className="card"><div className="form-row"><label>Model<select value={selected} onChange={(e) => setModel(e.target.value)}>{state.models.map((item) => <option key={item}>{item}</option>)}</select></label></div><div className="input-grid">{features.map((feature) => <label key={feature}>{feature}<input type="number" value={values[feature] ?? 0} onChange={(e) => setValues({ ...values, [feature]: Number(e.target.value) })} /></label>)}</div><button className="primary" onClick={submit}>Run prediction <span>→</span></button></section>{warnings.map((warning) => <div className="alert warning" key={warning}>{warning}</div>)}{result && <section className="card result-card"><SectionTitle title="Predicted S11 response" detail="Complete response curve from the selected model." /><div className="result-grid"><Metric label="Minimum S11 (dB)" value={Number(result.metrics.minimum_s11_db).toFixed(5)} /><Metric label="Resonant frequency" value={result.metrics.resonant_frequency_hz ? `${(Number(result.metrics.resonant_frequency_hz) / 1e9).toFixed(4)} GHz` : "Unavailable"} /><Metric label="−10 dB bandwidth" value={result.metrics.bandwidth_hz ? `${(Number(result.metrics.bandwidth_hz) / 1e6).toFixed(3)} MHz` : "Unavailable"} /></div><div className="chart"><ResponsiveContainer width="100%" height={280}><LineChart data={Object.entries(result.prediction).map(([name, value], index) => ({ name, index, value }))}><CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" /><XAxis dataKey="index" /><YAxis /><Tooltip /><Line type="monotone" dataKey="value" stroke="#16a394" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></div></section>}</>;
}

function OptimizationPage({ state, setState, setError }: PageProps) {
  const [model, setModel] = useState(""); const [algorithm, setAlgorithm] = useState("PSO"); const [population, setPopulation] = useState(25); const [iterations, setIterations] = useState(30); const [bounds, setBounds] = useState<Record<string, [number, number]>>({});
  const selected = model || state.models[0] || ""; const features = state.dataset?.inputColumns ?? [];
  if (!state.models.length || !state.dataset) return <Empty>Train a model first</Empty>;
  const submit = async () => { try { const result = await api<Optimization>("/api/optimize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: selected, algorithm, population, iterations, bounds: Object.fromEntries(features.map((f) => bounds[f] ?? [0, 1])) }) }); setState({ ...state, optimizations: { ...state.optimizations, [algorithm]: result } }); } catch (e) { setError((e as Error).message); } };
  return <>
    <Intro title="Bounded optimization" text="Search only within the observed training domain to reduce model extrapolation." />
    <section className="card">
      <div className="form-row">
        <label>Model<select value={selected} onChange={(e) => setModel(e.target.value)}>{state.models.map((item) => <option key={item}>{item}</option>)}</select></label>
        <label>Algorithm<select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}><option>PSO</option><option>DE</option><option>GWO</option></select></label>
        <label>Population<input type="number" min="5" max="100" value={population} onChange={(e) => setPopulation(Number(e.target.value))} /></label>
        <label>Iterations<input type="number" min="1" max="200" value={iterations} onChange={(e) => setIterations(Number(e.target.value))} /></label>
      </div>
      <div className="input-grid">
        {features.map((feature) => <label key={feature}>{feature} lower / upper
          <input placeholder="lower, upper" onChange={(e) => {
            const [lo, hi] = e.target.value.split(",").map(Number);
            if (!Number.isNaN(lo) && !Number.isNaN(hi)) setBounds({ ...bounds, [feature]: [lo, hi] });
          }} />
        </label>)}
      </div>
      <button className="primary" onClick={submit}>Run {algorithm} <span>→</span></button>
    </section>
    {state.optimizations[algorithm] && <OptimizationResult result={state.optimizations[algorithm]} />}
  </>;
}
function OptimizationResult({ result }: { result: Optimization }) { return <section className="card"><SectionTitle title={`${result.algorithm} result`} detail={`${result.elapsed_seconds.toFixed(2)} seconds`} /><div className="result-grid"><Metric label="Best fitness" value={result.best_fitness.toFixed(5)} /><Metric label="Iterations" value={result.history.length - 1} /></div><div className="chart"><ResponsiveContainer width="100%" height={240}><LineChart data={result.history.map((fitness, iteration) => ({ iteration, fitness }))}><CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" /><XAxis dataKey="iteration" /><YAxis /><Tooltip /><Line type="monotone" dataKey="fitness" stroke="#e8794f" strokeWidth={3} dot={false} /></LineChart></ResponsiveContainer></div></section>; }

function HfssPage({ setError }: PageProps) { const [result, setResult] = useState<Record<string, unknown> | null>(null); const submit = async (event: ChangeEvent<HTMLInputElement>) => { const file = event.target.files?.[0]; if (!file) return; const form = new FormData(); form.append("result", file); try { setResult(await api("/api/hfss", { method: "POST", body: form })); } catch (e) { setError((e as Error).message); } }; return <><Intro title="HFSS validation" text="Compare actual exported HFSS results. The application never fabricates unavailable simulations." /><section className="card"><FileBox title="HFSS CSV/XLSX result" required onChange={submit} /><div className="alert warning">Automated execution requires PyAEDT and a licensed Ansys Electronics Desktop installation.</div>{result && <Preview data={(result.rows as Record<string, unknown>[]) ?? []} />}</section></>; }
function ReportsPage({ state, setError }: PageProps) { const download = async () => { try { const blob = await api<Blob>("/api/report"); const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "antenna_report.pdf"; link.click(); URL.revokeObjectURL(link.href); } catch (e) { setError((e as Error).message); } }; return <><Intro title="Reports and exports" text="Download evidence from the current dataset and model evaluation." /><section className="card report-card"><div className="report-icon">▤</div><div><h3>Research summary report</h3><p>{state.dataset ? `${state.dataset.rows.toLocaleString()} samples · ${state.models.length} trained models` : "Load a dataset to generate a report."}</p></div><button className="secondary" disabled={!state.dataset} onClick={download}>Download PDF</button></section></>; }

export default App;
