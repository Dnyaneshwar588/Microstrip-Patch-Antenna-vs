import {
  ChangeEvent, FormEvent, useCallback, useEffect, useMemo, useRef, useState,
} from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import { api, downloadBlob, fmt, fmtGHz, fmtMHz } from "./api";
import type {
  AppState, CompareResult, CompareRow, Dataset, Evaluation,
  HfssCompareResponse, HfssSimpleResponse, OptimizationResult,
  PredictResponse, ResponseMetrics,
} from "./types";

// ─── Constants ─────────────────────────────────────────────────────────────────

const NAV_ITEMS = [
  { key: "Dashboard",            icon: "⌂" },
  { key: "Dataset Management",   icon: "▦" },
  { key: "Data Preprocessing",   icon: "⚙" },
  { key: "ML Models",            icon: "◇" },
  { key: "Performance Prediction", icon: "⌁" },
  { key: "Antenna Optimization", icon: "✦" },
  { key: "Algorithm Comparison", icon: "⇌" },
  { key: "HFSS Validation",      icon: "◉" },
  { key: "Reports & Results",    icon: "▤" },
];

const TEAL   = "#16a394";
const ORANGE = "#e8794f";
const NAVY   = "#10272d";
const CHART_COLORS = [TEAL, ORANGE, "#6c8ebf", "#82ca9d"];

// ─── Shared primitives ─────────────────────────────────────────────────────────

function Metric({ label, value, unit }: { label: string; value: string | number; unit?: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}{unit ? <em className="metric-unit"> {unit}</em> : null}</strong>
    </div>
  );
}

function SectionTitle({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="section-title">
      <div>
        <h3>{title}</h3>
        {detail && <p>{detail}</p>}
      </div>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return <div className="empty-state">{text}</div>;
}

function Empty({ children }: { children: string }) {
  return (
    <div className="empty">
      <div className="empty-icon">◌</div>
      <h3>{children}</h3>
      <p>Complete the previous step to unlock this workspace.</p>
    </div>
  );
}

function Intro({ title, text }: { title: string; text: string }) {
  return (
    <div className="intro">
      <span className="eyebrow accent">WORKFLOW</span>
      <h2>{title}</h2>
      <p>{text}</p>
    </div>
  );
}

function Alert({ type, children, onDismiss }: {
  type: "error" | "warning" | "info" | "success";
  children: React.ReactNode;
  onDismiss?: () => void;
}) {
  return (
    <div className={`alert ${type}`}>
      {children}
      {onDismiss && <button onClick={onDismiss} aria-label="Dismiss">✕</button>}
    </div>
  );
}

function Spinner({ label = "Loading…" }: { label?: string }) {
  return <div className="spinner-wrap"><div className="spinner" /><span>{label}</span></div>;
}

function Badge({ children, color = TEAL }: { children: React.ReactNode; color?: string }) {
  return <span className="badge" style={{ background: color }}>{children}</span>;
}

function FileBox({
  title, required, file, onChange, accept = ".csv,.xlsx,.xls",
}: {
  title: string; required?: boolean; file?: File; onChange: (e: ChangeEvent<HTMLInputElement>) => void; accept?: string;
}) {
  return (
    <label className="file-box">
      <span>{title}{required && <i> REQUIRED</i>}</span>
      <div className={`file-input${file ? " file-input--loaded" : ""}`}>
        {file ? `✓ ${file.name}` : "Choose CSV or XLSX"}
        <input type="file" accept={accept} required={required} onChange={onChange} />
      </div>
    </label>
  );
}

function EvaluationTable({ rows }: { rows: Evaluation[] }) {
  const best = rows.reduce((a, b) => (a.r2 > b.r2 ? a : b), rows[0]);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th><th>R² Score</th><th>Expl. Variance</th><th>RMSE</th><th>MAE</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.model} className={row.model === best.model ? "row-best" : ""}>
              <td><b>{row.model}</b>{row.model === best.model && <Badge> ★ Best</Badge>}</td>
              <td className="positive">{fmt(row.r2)}</td>
              <td>{row.explained_variance != null ? fmt(row.explained_variance) : "—"}</td>
              <td>{fmt(row.rmse)}</td>
              <td>{fmt(row.mae)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricComparChart({ rows }: { rows: Evaluation[] }) {
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height={230}>
        <LineChart data={rows}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" />
          <XAxis dataKey="model" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="r2" name="R²" stroke={TEAL} strokeWidth={3} dot={{ r: 4 }} />
          <Line type="monotone" dataKey="explained_variance" name="Expl. Var." stroke={ORANGE} strokeWidth={2} dot={{ r: 3 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── App root ──────────────────────────────────────────────────────────────────

type PageProps = {
  state: AppState;
  setState: (s: AppState) => void;
  setError: (s: string) => void;
  refresh: () => void;
};

function App() {
  const [page, setPage] = useState("Dashboard");
  const [state, setState] = useState<AppState>({ dataset: null, models: [], evaluation: null, optimizations: {} });
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState<"checking" | "ok" | "error">("checking");

  const refresh = useCallback(() => {
    api<AppState>("/api/state").then(setState).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    refresh();
    api("/api/health")
      .then(() => setApiStatus("ok"))
      .catch(() => setApiStatus("error"));
  }, [refresh]);

  const ctx = { state, setState, setError, refresh };

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">∿</span>
          <div>
            <strong>Antenna Lab</strong>
            <small>Research workspace</small>
          </div>
        </div>
        <nav>
          {NAV_ITEMS.map(({ key, icon }) => (
            <button
              key={key}
              className={page === key ? "nav-item active" : "nav-item"}
              onClick={() => setPage(key)}
            >
              <span>{icon}</span>{key}
            </button>
          ))}
        </nav>

      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <span className="eyebrow">ENGINEERING WORKSPACE</span>
            <h1>{page}</h1>
          </div>
          <div className="status">
            <span
              className="status-dot"
              style={{ background: apiStatus === "ok" ? "#25b89f" : apiStatus === "error" ? "#e05252" : "#e8c84f" }}
            />
            {apiStatus === "ok" ? "API connected" : apiStatus === "error" ? "API unreachable" : "Connecting…"}
          </div>
        </header>

        {error && (
          <Alert type="error" onDismiss={() => setError("")}>{error}</Alert>
        )}

        {page === "Dashboard"            && <DashboardPage {...ctx} />}
        {page === "Dataset Management"   && <DatasetManagementPage {...ctx} />}
        {page === "Data Preprocessing"   && <DataPreprocessingPage {...ctx} />}
        {page === "ML Models"            && <MlModelsPage {...ctx} />}
        {page === "Performance Prediction" && <PredictionPage {...ctx} />}
        {page === "Antenna Optimization" && <OptimizationPage {...ctx} />}
        {page === "Algorithm Comparison" && <AlgorithmComparisonPage {...ctx} />}
        {page === "HFSS Validation"      && <HfssPage {...ctx} />}
        {page === "Reports & Results"    && <ReportsPage {...ctx} />}
      </main>
    </div>
  );
}

// ─── 1. DASHBOARD ──────────────────────────────────────────────────────────────

function DashboardPage({ state }: PageProps) {
  const best = state.evaluation?.[0];
  const bestOpt = useMemo(() => {
    const entries = Object.entries(state.optimizations);
    if (!entries.length) return null;
    return entries.reduce((a, b) => (a[1].best_fitness < b[1].best_fitness ? a : b));
  }, [state.optimizations]);

  return (
    <>
      <section className="hero">
        <div>
          <span className="eyebrow accent">MICROSTRIP PATCH ANTENNA</span>
          <h2>Design with evidence.<br /><em>Optimize with confidence.</em></h2>
          <p>Load measured data, train performance models, and explore bounded optimization experiments.</p>
        </div>
        <div className="hero-wave">∿<br />∿<br />∿</div>
      </section>

      <div className="metric-grid">
        <Metric label="Dataset samples"  value={state.dataset ? state.dataset.rows.toLocaleString() : "—"} />
        <Metric label="Input parameters" value={state.dataset ? state.dataset.inputColumns.length : "—"} />
        <Metric label="S11 sweep points" value={state.dataset?.metadata?.s11_points ?? state.dataset?.targetColumns.length ?? "—"} />
        <Metric label="Trained models"   value={state.models.length || "—"} />
        <Metric label="Best ML model"    value={best?.model ?? "—"} />
        <Metric label="Best R² score"    value={best ? fmt(best.r2) : "—"} />
        <Metric label="Best optimizer"   value={bestOpt ? bestOpt[0] : "—"} />
        <Metric label="Best fitness"     value={bestOpt ? fmt(bestOpt[1].best_fitness, 5) : "—"} />
      </div>

      {!state.dataset && (
        <div className="empty">
          <div className="empty-icon">◌</div>
          <h3>Ready for your antenna dataset</h3>
          <p>Go to <b>Dataset Management</b> to upload input parameters and S11 measurement files.</p>
        </div>
      )}

      {state.dataset?.metadata?.frequency_labels_verified === false && (
        <Alert type="warning">
          Frequency labels are not present in the imported files. Resonant frequency and bandwidth
          remain unavailable until frequency metadata is supplied.
        </Alert>
      )}

      {state.evaluation && (
        <section className="card">
          <SectionTitle title="Model validation overview" detail={best ? `Best model: ${best.model} (R²=${fmt(best.r2)})` : ""} />
          <EvaluationTable rows={state.evaluation} />
          <MetricComparChart rows={state.evaluation} />
        </section>
      )}

      {Object.keys(state.optimizations).length > 0 && (
        <section className="card">
          <SectionTitle title="Latest optimization results" />
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Algorithm</th><th>Best fitness</th><th>Iterations</th><th>Time (s)</th></tr>
              </thead>
              <tbody>
                {Object.entries(state.optimizations).map(([alg, r]) => (
                  <tr key={alg}>
                    <td><b>{r.algorithm ?? alg}</b></td>
                    <td className="positive">{fmt(r.best_fitness, 5)}</td>
                    <td>{r.history.length - 1}</td>
                    <td>{fmt(r.elapsed_seconds, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  );
}

// ─── 2. DATASET MANAGEMENT ─────────────────────────────────────────────────────

function DatasetManagementPage({ state, setState, setError }: PageProps) {
  const [files, setFiles] = useState<{ inputs?: File; real?: File; imag?: File; frequency?: File }>({});
  const [frequency, setFrequency] = useState({ start: "", stop: "", count: "" });
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 10;

  const upload = (key: "inputs" | "real" | "imag" | "frequency") =>
    (e: ChangeEvent<HTMLInputElement>) =>
      setFiles((c) => ({ ...c, [key]: e.target.files?.[0] }));

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!files.inputs) { setError("Choose an input parameter file first."); return; }
    setLoading(true);
    const form = new FormData();
    form.append("inputs", files.inputs);
    if (files.real)      form.append("s11_real", files.real);
    if (files.imag)      form.append("s11_imag", files.imag);
    if (files.frequency) form.append("frequency", files.frequency);
    if (frequency.start) form.append("frequency_start", frequency.start);
    if (frequency.stop)  form.append("frequency_stop", frequency.stop);
    if (frequency.count) form.append("frequency_count", frequency.count);
    try {
      setState(await api<AppState>("/api/dataset", { method: "POST", body: form }));
      setPage(0);
    } catch (e) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const ds = state.dataset;
  const rows = ds?.preview.rows ?? [];
  const cols = rows.length ? Object.keys(rows[0]) : [];
  const totalPages = Math.ceil(rows.length / PAGE_SIZE);
  const pageRows = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <>
      <Intro title="Dataset intake" text="Upload antenna input parameters and measured S11 data. Inspect structure, missing values, and frequency metadata before modelling." />

      <form className="card upload-grid" onSubmit={submit}>
        <FileBox title="Input parameters"           required file={files.inputs}    onChange={upload("inputs")} />
        <FileBox title="S11 real component"                  file={files.real}      onChange={upload("real")} />
        <FileBox title="S11 imaginary component"             file={files.imag}      onChange={upload("imag")} />
        <FileBox title="Frequency metadata (optional)"       file={files.frequency} onChange={upload("frequency")} />

        <div className="form-row" style={{ gridColumn: "1/-1" }}>
          <label>Frequency start (GHz)
            <input type="number" step="any" value={frequency.start}
              onChange={(e) => setFrequency({ ...frequency, start: e.target.value })} />
          </label>
          <label>Frequency stop (GHz)
            <input type="number" step="any" value={frequency.stop}
              onChange={(e) => setFrequency({ ...frequency, stop: e.target.value })} />
          </label>
          <label>Sweep points
            <input type="number" min="2" value={frequency.count}
              onChange={(e) => setFrequency({ ...frequency, count: e.target.value })} />
          </label>
          <div />
        </div>

        <div style={{ gridColumn: "1/-1" }}>
          <button className="primary" type="submit" disabled={loading}>
            {loading ? <><Spinner label="" /> Loading…</> : <>Load and inspect dataset <span>→</span></>}
          </button>
        </div>
      </form>

      {ds && (
        <>
          {ds.warnings.map((w) => <Alert key={w} type="warning">{w}</Alert>)}

          <div className="metric-grid">
            <Metric label="Total samples"     value={ds.rows.toLocaleString()} />
            <Metric label="Total columns"     value={ds.columns.length} />
            <Metric label="Input features"    value={ds.inputColumns.length} />
            <Metric label="S11 targets"       value={ds.targetColumns.length} />
            <Metric label="Missing values"    value={ds.metadata?.missing_values ?? "—"} />
            <Metric label="Duplicate rows"    value={ds.metadata?.duplicate_rows ?? "—"} />
            <Metric label="S11 sweep points"  value={ds.metadata?.s11_points ?? ds.targetColumns.length} />
            <Metric label="Frequency verified" value={ds.metadata?.frequency_labels_verified === true ? "Yes" : ds.metadata?.frequency_labels_verified === false ? "No" : "—"} />
          </div>

          <section className="card">
            <SectionTitle title="Dataset preview" detail={`${ds.rows.toLocaleString()} samples · ${ds.columns.length} columns (showing first ${rows.length})`} />
            {rows.length > 0 ? (
              <>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        {cols.map((c) => (
                          <th key={c} title={c}>{c.length > 20 ? c.slice(0, 18) + "…" : c}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {pageRows.map((row, i) => (
                        <tr key={i}>
                          {cols.map((c) => (
                            <td key={c}>{row[c] != null ? String(row[c]) : "—"}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {totalPages > 1 && (
                  <div className="pagination">
                    <button className="secondary" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>‹ Prev</button>
                    <span>Page {page + 1} / {totalPages}</span>
                    <button className="secondary" disabled={page >= totalPages - 1} onClick={() => setPage((p) => p + 1)}>Next ›</button>
                  </div>
                )}
              </>
            ) : (
              <EmptyState text="No preview rows returned." />
            )}
          </section>

          {ds.inputColumns.length > 0 && (
            <section className="card">
              <SectionTitle title="Feature inspection" detail="Input parameter ranges from training data" />
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr><th>Feature</th><th>Type</th><th>Min</th><th>Max</th></tr>
                  </thead>
                  <tbody>
                    {ds.inputColumns.map((col) => {
                      const r = ds.metadata?.numeric_ranges?.[col];
                      return (
                        <tr key={col}>
                          <td><b>{col}</b></td>
                          <td>Input</td>
                          <td>{r ? fmt(r.min, 4) : "—"}</td>
                          <td>{r ? fmt(r.max, 4) : "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </>
  );
}

// ─── 3. DATA PREPROCESSING ────────────────────────────────────────────────────

function DataPreprocessingPage({ state }: PageProps) {
  if (!state.dataset) return <Empty>Load a dataset first</Empty>;
  const ds = state.dataset;

  const inputRanges = ds.inputColumns.map((col) => ({
    col,
    ...ds.metadata?.numeric_ranges?.[col],
  }));

  return (
    <>
      <Intro title="Data preprocessing" text="Review dataset structure, feature ranges, S11 target distribution, and frequency metadata before training." />

      <div className="two-col">
        <section className="card">
          <SectionTitle title="Column classification" />
          <table>
            <thead><tr><th>Column</th><th>Role</th></tr></thead>
            <tbody>
              {ds.inputColumns.map((c) => <tr key={c}><td>{c}</td><td><Badge>Input</Badge></td></tr>)}
              {ds.targetColumns.slice(0, 10).map((c) => <tr key={c}><td>{c.length > 25 ? c.slice(0, 23) + "…" : c}</td><td><Badge color={ORANGE}>S11 Target</Badge></td></tr>)}
              {ds.targetColumns.length > 10 && <tr><td colSpan={2}><em style={{ color: "#7e9095", fontSize: 11 }}>… and {ds.targetColumns.length - 10} more S11 targets</em></td></tr>}
            </tbody>
          </table>
        </section>

        <section className="card">
          <SectionTitle title="Dataset quality" />
          <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
              <tr><td>Total rows</td><td>{ds.rows.toLocaleString()}</td></tr>
              <tr><td>Total columns</td><td>{ds.columns.length}</td></tr>
              <tr><td>Input features</td><td>{ds.inputColumns.length}</td></tr>
              <tr><td>S11 output points</td><td>{ds.targetColumns.length}</td></tr>
              <tr><td>Missing values</td><td className={ds.metadata?.missing_values ? "warn-cell" : "positive"}>{ds.metadata?.missing_values ?? "—"}</td></tr>
              <tr><td>Duplicate rows</td><td className={ds.metadata?.duplicate_rows ? "warn-cell" : "positive"}>{ds.metadata?.duplicate_rows ?? "—"}</td></tr>
              <tr><td>Frequency metadata</td><td>{ds.metadata?.frequency_available ? "✓ Available" : "✗ Not provided"}</td></tr>
              <tr><td>Freq. labels verified</td><td>{ds.metadata?.frequency_labels_verified === true ? "✓ Verified" : ds.metadata?.frequency_labels_verified === false ? "✗ Not verified" : "—"}</td></tr>
            </tbody>
          </table>
        </section>
      </div>

      {inputRanges.length > 0 && (
        <section className="card">
          <SectionTitle title="Input parameter ranges" detail="Observed min/max from training data — used as default optimization bounds" />
          <div className="table-wrap">
            <table>
              <thead><tr><th>Parameter</th><th>Min</th><th>Max</th><th>Range</th></tr></thead>
              <tbody>
                {inputRanges.map(({ col, min, max }) => (
                  <tr key={col}>
                    <td><b>{col}</b></td>
                    <td>{min != null ? fmt(min, 4) : "—"}</td>
                    <td>{max != null ? fmt(max, 4) : "—"}</td>
                    <td>{min != null && max != null ? fmt(max - min, 4) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {ds.frequencyHz && ds.frequencyHz.length > 0 && (
        <section className="card">
          <SectionTitle title="Frequency sweep metadata" detail={`${ds.frequencyHz.length} points · ${fmtGHz(ds.frequencyHz[0])} – ${fmtGHz(ds.frequencyHz[ds.frequencyHz.length - 1])}`} />
          <div className="chart">
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={ds.frequencyHz.map((f, i) => ({ index: i, freq: f / 1e9 }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" />
                <XAxis dataKey="index" tick={{ fontSize: 10 }} label={{ value: "Point index", position: "insideBottom", fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} unit=" GHz" />
                <Tooltip formatter={(v: number) => [`${v.toFixed(4)} GHz`, "Frequency"]} />
                <Line type="monotone" dataKey="freq" stroke={TEAL} strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      {ds.warnings.map((w) => <Alert key={w} type="warning">{w}</Alert>)}
    </>
  );
}

// ─── 4. ML MODELS ─────────────────────────────────────────────────────────────

function MlModelsPage({ state, setState, setError }: PageProps) {
  const [features, setFeatures] = useState<string[]>(state.dataset?.inputColumns ?? []);
  const [targets, setTargets] = useState<string[]>(state.dataset?.targetColumns ?? []);
  const [training, setTraining] = useState(false);

  if (!state.dataset) return <Empty>Load a dataset first</Empty>;

  const numeric = state.dataset.columns;

  const train = async () => {
    if (!features.length) { setError("Select at least one input feature."); return; }
    if (!targets.length)  { setError("Select at least one target column."); return; }
    setTraining(true);
    try {
      setState(await api<AppState>("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features, targets }),
      }));
    } catch (e) { setError((e as Error).message); }
    finally { setTraining(false); }
  };

  return (
    <>
      <Intro title="Model training" text="Select numeric inputs and measured S11 outputs. Training uses an 80:20 holdout split across four models: Random Forest, Gradient Boosting, SVR, and ANN." />

      <div className="two-col">
        <ColumnPicker title="Input features" options={numeric} selected={features} setSelected={setFeatures}
          hint={`${features.length} selected`} />
        <ColumnPicker title="Prediction targets (S11)" options={numeric} selected={targets} setSelected={setTargets}
          hint={`${targets.length} selected`} />
      </div>

      <button className="primary" disabled={training} onClick={train}>
        {training ? <><Spinner label="" /> Training four models…</> : <>Train four models <span>→</span></>}
      </button>

      {training && (
        <section className="card">
          <Spinner label="Training Random Forest, Gradient Boosting, SVR, and ANN. This may take a moment…" />
        </section>
      )}

      {state.evaluation && (
        <section className="card">
          <SectionTitle title="Validation results" detail="Metrics computed on held-out 20% test data, not training scores." />
          <EvaluationTable rows={state.evaluation} />
          <MetricComparChart rows={state.evaluation} />
        </section>
      )}

      {state.evaluation && (
        <section className="card">
          <SectionTitle title="Detailed metric comparison" />
          <div className="chart">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={state.evaluation}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" />
                <XAxis dataKey="model" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line yAxisId="right" type="monotone" dataKey="rmse" name="RMSE" stroke={ORANGE} strokeWidth={2} dot={{ r: 4 }} />
                <Line yAxisId="right" type="monotone" dataKey="mae"  name="MAE"  stroke="#6c8ebf" strokeWidth={2} dot={{ r: 4 }} />
                <Line yAxisId="left"  type="monotone" dataKey="r2"   name="R²"   stroke={TEAL}   strokeWidth={3} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}
    </>
  );
}

function ColumnPicker({
  title, options, selected, setSelected, hint,
}: {
  title: string; options: string[]; selected: string[]; setSelected: (s: string[]) => void; hint?: string;
}) {
  const all = options.every((o) => selected.includes(o));
  return (
    <section className="card picker">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <h3>{title}</h3>
        {hint && <span style={{ fontSize: 11, color: TEAL, fontWeight: 700 }}>{hint}</span>}
      </div>
      <label style={{ marginBottom: 8 }}>
        <input type="checkbox" checked={all}
          onChange={(e) => setSelected(e.target.checked ? [...options] : [])} />
        {" "}Select all
      </label>
      {options.map((option) => (
        <label key={option}>
          <input
            type="checkbox"
            checked={selected.includes(option)}
            onChange={(e) =>
              setSelected(e.target.checked ? [...selected, option] : selected.filter((x) => x !== option))
            }
          /> {option}
        </label>
      ))}
    </section>
  );
}

// ─── 5. PERFORMANCE PREDICTION ────────────────────────────────────────────────

function PredictionPage({ state, setError }: PageProps) {
  const [model, setModel] = useState("");
  const [values, setValues] = useState<Record<string, number>>({});
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const selected = model || state.models[0] || "";
  const features = state.dataset?.inputColumns ?? [];

  // Initialise with training-data median values from numeric_ranges midpoints
  useEffect(() => {
    if (!state.models[0] && !model) return;
    if (!state.dataset) return;
    const ranges = state.dataset.metadata?.numeric_ranges ?? {};
    const init: Record<string, number> = {};
    for (const f of features) {
      const r = ranges[f];
      init[f] = r ? (r.min + r.max) / 2 : 0;
    }
    setValues(init);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.models, features.join(",")]);

  useEffect(() => {
    if (!model && state.models[0]) setModel(state.models[0]);
  }, [model, state.models]);

  if (!state.models.length || !state.dataset) return <Empty>Train a model first</Empty>;

  const predict = async () => {
    setLoading(true);
    try {
      const r = await api<PredictResponse>("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: selected, values }),
      });
      setResult(r);
    } catch (e) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const downloadCsv = async () => {
    try {
      const blob = await api<Blob>("/api/predict.csv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: selected, values }),
      });
      downloadBlob(blob, "predicted_s11.csv");
    } catch (e) { setError((e as Error).message); }
  };

  const chartData = useMemo(() => {
    if (!result) return [];
    const freqs = result.metrics.frequency_hz;
    return Object.entries(result.prediction).map(([, value], i) => ({
      freq: freqs ? freqs[i] / 1e9 : i,
      s11: value,
    }));
  }, [result]);

  const m = result?.metrics;

  // Specification thresholds
  const specOk = m ? m.minimum_s11_db <= -10 : null;

  return (
    <>
      <Intro title="Performance prediction" text="Evaluate a candidate antenna design using a trained ML model. Input values are pre-filled with training-data midpoint values." />

      <section className="card">
        <div className="form-row">
          <label>Model
            <select value={selected} onChange={(e) => setModel(e.target.value)}>
              {state.models.map((m) => <option key={m}>{m}</option>)}
            </select>
          </label>
        </div>

        <div className="input-grid">
          {features.map((f) => {
            const r = state.dataset?.metadata?.numeric_ranges?.[f];
            return (
              <label key={f}>
                {f}{r && <span style={{ color: "#aaa", fontWeight: 400 }}> [{fmt(r.min, 2)}–{fmt(r.max, 2)}]</span>}
                <input
                  type="number"
                  step="any"
                  value={values[f] ?? 0}
                  onChange={(e) => setValues({ ...values, [f]: Number(e.target.value) })}
                />
              </label>
            );
          })}
        </div>

        <div style={{ display: "flex", gap: 10 }}>
          <button className="primary" onClick={predict} disabled={loading}>
            {loading ? <><Spinner label="" /> Running…</> : <>Run prediction <span>→</span></>}
          </button>
          {result && (
            <button className="secondary" onClick={downloadCsv}>↓ Download CSV</button>
          )}
        </div>
      </section>

      {result?.warnings.map((w) => <Alert key={w} type="warning">{w}</Alert>)}

      {result && m && (
        <section className="card">
          <SectionTitle title="Predicted S11 response" detail={`Model: ${selected}`} />

          <div className="metric-grid">
            <Metric label="Min S11"           value={fmt(m.minimum_s11_db, 3)} unit="dB" />
            <Metric label="Resonant frequency" value={m.resonant_frequency_hz ? fmtGHz(m.resonant_frequency_hz) : "—"} />
            <Metric label="−10 dB bandwidth"  value={m.bandwidth_hz && m.bandwidth_hz > 0 ? fmtMHz(m.bandwidth_hz) : "—"} />
            <Metric label="VSWR at resonance"  value={m.vswr ? fmt(m.vswr[m.minimum_s11_index ?? 0], 3) : "—"} />
            <Metric label="Spec. ≤ −10 dB"    value={specOk === null ? "—" : specOk ? "✓ Pass" : "✗ Fail"} />
            <Metric label="Frac. bandwidth"    value={m.fractional_bandwidth ? `${(m.fractional_bandwidth * 100).toFixed(2)}%` : "—"} />
          </div>

          {!m.frequency_available && (
            <Alert type="info">Frequency metadata unavailable. X-axis shows sweep index. Upload frequency data for GHz labelling.</Alert>
          )}

          <div className="chart">
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" />
                <XAxis dataKey="freq"
                  label={{ value: m.frequency_available ? "Frequency (GHz)" : "Sweep point", position: "insideBottom", offset: -4, fontSize: 11 }}
                  tick={{ fontSize: 10 }}
                  tickFormatter={(v) => typeof v === "number" ? v.toFixed(2) : v}
                />
                <YAxis tick={{ fontSize: 11 }} label={{ value: "S11 (dB)", angle: -90, position: "insideLeft", fontSize: 11 }} />
                <Tooltip
                  formatter={(v: number) => [`${v.toFixed(3)} dB`, "S11"]}
                  labelFormatter={(l: number) => m.frequency_available ? `${l.toFixed(4)} GHz` : `Point ${l}`}
                />
                <ReferenceLine y={-10} stroke={ORANGE} strokeDasharray="5 5" label={{ value: "−10 dB", position: "insideTopLeft", fontSize: 10, fill: ORANGE }} />
                <Line type="monotone" dataKey="s11" stroke={TEAL} strokeWidth={2} dot={false} name="S11 (dB)" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <section style={{ marginTop: 20 }}>
            <SectionTitle title="Complete prediction table" detail={`${Object.keys(result.prediction).length} frequency points`} />
            <PredictionTable prediction={result.prediction} freqs={m.frequency_hz} vswr={m.vswr} />
          </section>
        </section>
      )}
    </>
  );
}

function PredictionTable({
  prediction, freqs, vswr,
}: {
  prediction: Record<string, number>; freqs?: number[]; vswr?: number[];
}) {
  const [pg, setPg] = useState(0);
  const PAGE = 20;
  const entries = Object.entries(prediction);
  const totalPg = Math.ceil(entries.length / PAGE);
  const slice = entries.slice(pg * PAGE, (pg + 1) * PAGE);

  return (
    <>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>{freqs ? "Frequency (GHz)" : "Sweep index"}</th>
              <th>S11 (dB)</th>
              <th>VSWR</th>
            </tr>
          </thead>
          <tbody>
            {slice.map(([, s11], i) => {
              const idx = pg * PAGE + i;
              return (
                <tr key={idx} className={s11 <= -10 ? "row-pass" : ""}>
                  <td>{idx + 1}</td>
                  <td>{freqs ? `${(freqs[idx] / 1e9).toFixed(4)} GHz` : idx}</td>
                  <td>{fmt(s11, 3)}</td>
                  <td>{vswr ? fmt(vswr[idx], 3) : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {totalPg > 1 && (
        <div className="pagination">
          <button className="secondary" disabled={pg === 0} onClick={() => setPg((p) => p - 1)}>‹ Prev</button>
          <span>Page {pg + 1} / {totalPg}</span>
          <button className="secondary" disabled={pg >= totalPg - 1} onClick={() => setPg((p) => p + 1)}>Next ›</button>
        </div>
      )}
    </>
  );
}

// ─── 6. ANTENNA OPTIMIZATION ──────────────────────────────────────────────────

function OptimizationPage({ state, setState, setError }: PageProps) {
  const features = state.dataset?.inputColumns ?? [];
  const [model, setModel]       = useState("");
  const [algorithm, setAlgo]    = useState("PSO");
  const [population, setPop]    = useState(25);
  const [iterations, setIter]   = useState(50);
  const [targetFreq, setTFreq]  = useState("");
  const [minS11, setMinS11]     = useState("-10");
  const [minBW, setMinBW]       = useState("0");
  const [maxVSWR, setMaxVSWR]   = useState("2.0");
  const [loading, setLoading]   = useState(false);

  // Bounds: auto-initialised from training data ranges
  const defaultBounds = useMemo<Record<string, [number, number]>>(() => {
    const ranges = state.dataset?.metadata?.numeric_ranges ?? {};
    const b: Record<string, [number, number]> = {};
    for (const f of features) {
      const r = ranges[f];
      b[f] = r ? [r.min, r.max] : [0, 1];
    }
    return b;
  }, [features, state.dataset]);

  const [bounds, setBounds] = useState<Record<string, [number, number]>>(defaultBounds);

  useEffect(() => setBounds(defaultBounds), [defaultBounds]);
  useEffect(() => { if (!model && state.models[0]) setModel(state.models[0]); }, [model, state.models]);

  if (!state.models.length || !state.dataset) return <Empty>Train a model first</Empty>;

  const run = async () => {
    setLoading(true);
    const boundsPayload: Record<string, [number, number]> = {};
    for (const f of features) boundsPayload[f] = bounds[f] ?? defaultBounds[f] ?? [0, 1];
    try {
      const result = await api<OptimizationResult>("/api/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model || state.models[0],
          algorithm,
          population,
          iterations,
          bounds: boundsPayload,
          target_frequency_hz: targetFreq ? Number(targetFreq) * 1e9 : null,
          desired_min_s11_db: Number(minS11),
          minimum_bandwidth_hz: Number(minBW) * 1e6,
          maximum_vswr: Number(maxVSWR),
        }),
      });
      setState({ ...state, optimizations: { ...state.optimizations, [algorithm]: result } });
    } catch (e) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const opt = state.optimizations[algorithm];
  const freqs = state.dataset.frequencyHz;

  // Compute optimized antenna predicted S11 from best_position
  const optParams = useMemo(() => {
    if (!opt) return null;
    return Object.fromEntries(features.map((f, i) => [f, opt.best_position[i] ?? 0]));
  }, [opt, features]);

  return (
    <>
      <Intro title="Antenna optimization" text="Search the design space using PSO, Differential Evolution, or Grey Wolf Optimization. Bounds are auto-initialized from training-data ranges." />

      <section className="card">
        <SectionTitle title="Optimization configuration" />
        <div className="form-row">
          <label>Model
            <select value={model || state.models[0]} onChange={(e) => setModel(e.target.value)}>
              {state.models.map((m) => <option key={m}>{m}</option>)}
            </select>
          </label>
          <label>Algorithm
            <select value={algorithm} onChange={(e) => setAlgo(e.target.value)}>
              <option>PSO</option>
              <option>DE</option>
              <option>GWO</option>
            </select>
          </label>
          <label>Population size
            <input type="number" min="5" max="100" value={population} onChange={(e) => setPop(Number(e.target.value))} />
          </label>
          <label>Iterations
            <input type="number" min="1" max="200" value={iterations} onChange={(e) => setIter(Number(e.target.value))} />
          </label>
        </div>

        <div className="form-row">
          <label>Target frequency (GHz)
            <input type="number" step="any" value={targetFreq} placeholder="e.g. 2.45"
              onChange={(e) => setTFreq(e.target.value)}
              disabled={!state.dataset?.metadata?.frequency_available}
              title={!state.dataset?.metadata?.frequency_available ? "Requires verified frequency metadata" : ""} />
          </label>
          <label>Desired min S11 (dB)
            <input type="number" step="any" value={minS11} max="0" onChange={(e) => setMinS11(e.target.value)} />
          </label>
          <label>Min −10 dB BW (MHz)
            <input type="number" step="any" min="0" value={minBW} onChange={(e) => setMinBW(e.target.value)} />
          </label>
          <label>Max VSWR
            <input type="number" step="any" min="1.001" value={maxVSWR} onChange={(e) => setMaxVSWR(e.target.value)} />
          </label>
        </div>

        <SectionTitle title="Parameter bounds" detail="Auto-initialized from training data. Adjust as needed." />
        <div className="bounds-grid">
          {features.map((f) => (
            <div key={f} className="bound-row">
              <span className="bound-label">{f}</span>
              <label>Lower
                <input type="number" step="any" value={bounds[f]?.[0] ?? 0}
                  onChange={(e) => setBounds({ ...bounds, [f]: [Number(e.target.value), bounds[f]?.[1] ?? 1] })} />
              </label>
              <label>Upper
                <input type="number" step="any" value={bounds[f]?.[1] ?? 1}
                  onChange={(e) => setBounds({ ...bounds, [f]: [bounds[f]?.[0] ?? 0, Number(e.target.value)] })} />
              </label>
            </div>
          ))}
        </div>

        <button className="primary" onClick={run} disabled={loading}>
          {loading ? <><Spinner label="" /> Running {algorithm}…</> : <>Run {algorithm} <span>→</span></>}
        </button>
      </section>

      {opt && (
        <>
          <section className="card">
            <SectionTitle title={`${opt.algorithm ?? algorithm} results`} detail={`Completed in ${fmt(opt.elapsed_seconds, 2)} s`} />
            <div className="metric-grid">
              <Metric label="Best fitness"  value={fmt(opt.best_fitness, 5)} />
              <Metric label="Iterations"    value={opt.history.length - 1} />
              <Metric label="Elapsed time"  value={fmt(opt.elapsed_seconds, 2)} unit="s" />
            </div>
          </section>

          <div className="two-col">
            <section className="card">
              <SectionTitle title="Convergence curve" />
              <div className="chart">
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={opt.history.map((f, i) => ({ iteration: i, fitness: f }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" />
                    <XAxis dataKey="iteration" label={{ value: "Iteration", position: "insideBottom", offset: -4, fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 11 }} label={{ value: "Fitness", angle: -90, position: "insideLeft", fontSize: 10 }} />
                    <Tooltip />
                    <Line type="monotone" dataKey="fitness" stroke={ORANGE} strokeWidth={3} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="card">
              <SectionTitle title="Optimized antenna dimensions" />
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Parameter</th><th>Initial (mid)</th><th>Optimized</th></tr></thead>
                  <tbody>
                    {features.map((f, i) => {
                      const r = state.dataset?.metadata?.numeric_ranges?.[f];
                      const mid = r ? (r.min + r.max) / 2 : 0;
                      return (
                        <tr key={f}>
                          <td><b>{f}</b></td>
                          <td>{fmt(mid, 4)}</td>
                          <td className="positive">{fmt(opt.best_position[i] ?? 0, 4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          </div>

          {optParams && freqs && (
            <OptimizationPredictionChart model={model || state.models[0]} params={optParams} freqs={freqs} setError={(e) => console.error(e)} />
          )}
        </>
      )}
    </>
  );
}

function OptimizationPredictionChart({
  model, params, freqs, setError,
}: {
  model: string; params: Record<string, number>; freqs: number[]; setError: (s: string) => void;
}) {
  const [data, setData] = useState<{ freq: number; s11: number }[]>([]);
  useEffect(() => {
    api<PredictResponse>("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, values: params }),
    }).then((r) => {
      const vals = Object.values(r.prediction);
      setData(freqs.map((f, i) => ({ freq: f / 1e9, s11: vals[i] ?? 0 })));
    }).catch((e) => setError(e.message));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, JSON.stringify(params)]);

  if (!data.length) return null;
  return (
    <section className="card">
      <SectionTitle title="Optimized S11 response" detail="ML prediction using the optimized antenna dimensions" />
      <div className="chart">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" />
            <XAxis dataKey="freq" tick={{ fontSize: 10 }} tickFormatter={(v) => v.toFixed(2)} label={{ value: "Frequency (GHz)", position: "insideBottom", offset: -4, fontSize: 10 }} />
            <YAxis tick={{ fontSize: 11 }} label={{ value: "S11 (dB)", angle: -90, position: "insideLeft", fontSize: 10 }} />
            <Tooltip formatter={(v: number) => [`${v.toFixed(3)} dB`, "S11"]} labelFormatter={(l: number) => `${l.toFixed(4)} GHz`} />
            <ReferenceLine y={-10} stroke={ORANGE} strokeDasharray="5 5" label={{ value: "−10 dB", position: "insideTopLeft", fontSize: 10, fill: ORANGE }} />
            <Line type="monotone" dataKey="s11" stroke={TEAL} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

// ─── 7. ALGORITHM COMPARISON ──────────────────────────────────────────────────

function AlgorithmComparisonPage({ state, setError }: PageProps) {
  const features = state.dataset?.inputColumns ?? [];
  const [model, setModel]      = useState("");
  const [population, setPop]   = useState(25);
  const [iterations, setIter]  = useState(30);
  const [trials, setTrials]    = useState(3);
  const [targetFreq, setTFreq] = useState("");
  const [loading, setLoading]  = useState(false);
  const [result, setResult]    = useState<CompareResult | null>(null);

  const defaultBounds = useMemo<Record<string, [number, number]>>(() => {
    const ranges = state.dataset?.metadata?.numeric_ranges ?? {};
    const b: Record<string, [number, number]> = {};
    for (const f of features) {
      const r = ranges[f];
      b[f] = r ? [r.min, r.max] : [0, 1];
    }
    return b;
  }, [features, state.dataset]);

  const [bounds, setBounds] = useState<Record<string, [number, number]>>(defaultBounds);
  useEffect(() => setBounds(defaultBounds), [defaultBounds]);
  useEffect(() => { if (!model && state.models[0]) setModel(state.models[0]); }, [model, state.models]);

  if (!state.models.length || !state.dataset) return <Empty>Train a model first</Empty>;

  const run = async () => {
    setLoading(true);
    try {
      const r = await api<CompareResult>("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model || state.models[0],
          algorithm: "PSO",
          population,
          iterations,
          trials,
          bounds,
          target_frequency_hz: targetFreq ? Number(targetFreq) * 1e9 : null,
          seed: 42,
        }),
      });
      setResult(r);
    } catch (e) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const chartData = useMemo(() => {
    if (!result) return [];
    const maxLen = Math.max(...result.results.map((r) => r.history.length));
    return Array.from({ length: maxLen }, (_, i) => {
      const pt: Record<string, number | string> = { iteration: i };
      for (const r of result.results) {
        pt[r.algorithm] = r.history[i] ?? r.history[r.history.length - 1];
      }
      return pt;
    });
  }, [result]);

  return (
    <>
      <Intro title="Algorithm comparison" text="Compare PSO, Differential Evolution, and Grey Wolf Optimization across multiple trials. Results are computed from actual backend runs — no fabrication." />

      <section className="card">
        <SectionTitle title="Comparison configuration" />
        <div className="form-row">
          <label>Model
            <select value={model || state.models[0]} onChange={(e) => setModel(e.target.value)}>
              {state.models.map((m) => <option key={m}>{m}</option>)}
            </select>
          </label>
          <label>Population
            <input type="number" min="5" max="100" value={population} onChange={(e) => setPop(Number(e.target.value))} />
          </label>
          <label>Iterations
            <input type="number" min="1" max="200" value={iterations} onChange={(e) => setIter(Number(e.target.value))} />
          </label>
          <label>Trials per algorithm
            <input type="number" min="1" max="10" value={trials} onChange={(e) => setTrials(Number(e.target.value))} />
          </label>
        </div>
        <div className="form-row">
          <label>Target frequency (GHz)
            <input type="number" step="any" value={targetFreq} placeholder="Optional"
              onChange={(e) => setTFreq(e.target.value)}
              disabled={!state.dataset?.metadata?.frequency_available} />
          </label>
        </div>

        <details>
          <summary style={{ cursor: "pointer", fontSize: 11, color: TEAL, fontWeight: 700, marginBottom: 10 }}>
            Parameter bounds (auto-initialized from training data)
          </summary>
          <div className="bounds-grid">
            {features.map((f) => (
              <div key={f} className="bound-row">
                <span className="bound-label">{f}</span>
                <label>Lower
                  <input type="number" step="any" value={bounds[f]?.[0] ?? 0}
                    onChange={(e) => setBounds({ ...bounds, [f]: [Number(e.target.value), bounds[f]?.[1] ?? 1] })} />
                </label>
                <label>Upper
                  <input type="number" step="any" value={bounds[f]?.[1] ?? 1}
                    onChange={(e) => setBounds({ ...bounds, [f]: [bounds[f]?.[0] ?? 0, Number(e.target.value)] })} />
                </label>
              </div>
            ))}
          </div>
        </details>

        <button className="primary" onClick={run} disabled={loading}>
          {loading ? <><Spinner label="" /> Running all three algorithms…</> : <>Compare PSO · DE · GWO <span>→</span></>}
        </button>
      </section>

      {loading && (
        <section className="card">
          <Spinner label="Running PSO, Differential Evolution, and Grey Wolf Optimization. This may take a minute…" />
        </section>
      )}

      {result && (
        <>
          {result.best_algorithm && (
            <Alert type="success">
              Best algorithm: <b>{result.best_algorithm}</b> (lowest average fitness across {trials} trial{trials > 1 ? "s" : ""})
            </Alert>
          )}

          <section className="card">
            <SectionTitle title="Summary statistics" detail={`${trials} trial${trials > 1 ? "s" : ""} per algorithm · ${iterations} iterations · population ${population}`} />
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Algorithm</th>
                    <th>Best fitness</th>
                    <th>Avg fitness</th>
                    <th>Worst</th>
                    <th>Std dev</th>
                    <th>Time (s)</th>
                    <th>Fn evals</th>
                  </tr>
                </thead>
                <tbody>
                  {result.results.map((r) => (
                    <tr key={r.algorithm} className={r.algorithm === result.best_algorithm ? "row-best" : ""}>
                      <td><b>{r.algorithm}</b>{r.algorithm === result.best_algorithm && <Badge> ★ Best</Badge>}</td>
                      <td className="positive">{fmt(r.best_fitness, 5)}</td>
                      <td>{fmt(r.average_fitness, 5)}</td>
                      <td>{fmt(r.worst_fitness, 5)}</td>
                      <td>{fmt(r.std_fitness, 5)}</td>
                      <td>{fmt(r.elapsed_seconds, 2)}</td>
                      <td>{r.objective_evaluations.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="card">
            <SectionTitle title="Convergence curves" detail="Best trial convergence for each algorithm" />
            <div className="chart">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" />
                  <XAxis dataKey="iteration" label={{ value: "Iteration", position: "insideBottom", offset: -4, fontSize: 10 }} tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 11 }} label={{ value: "Best fitness", angle: -90, position: "insideLeft", fontSize: 10 }} />
                  <Tooltip />
                  <Legend />
                  {result.results.map((r, i) => (
                    <Line key={r.algorithm} type="monotone" dataKey={r.algorithm} stroke={CHART_COLORS[i % CHART_COLORS.length]} strokeWidth={2} dot={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="card">
            <SectionTitle title="Optimized parameters per algorithm" />
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Parameter</th>
                    {result.results.map((r) => <th key={r.algorithm}>{r.algorithm}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {features.map((f, i) => (
                    <tr key={f}>
                      <td><b>{f}</b></td>
                      {result.results.map((r) => (
                        <td key={r.algorithm}>{fmt(r.best_position[i] ?? 0, 4)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </>
  );
}

// ─── 8. HFSS VALIDATION ───────────────────────────────────────────────────────

function HfssPage({ state, setError }: PageProps) {
  const [mode, setMode]         = useState<"simple" | "compare">("compare");
  const [file, setFile]         = useState<File | undefined>();
  const [modelSel, setModelSel] = useState("");
  const [values, setValues]     = useState<Record<string, number>>({});
  const [loading, setLoading]   = useState(false);
  const [simpleRes, setSimpleRes] = useState<HfssSimpleResponse | null>(null);
  const [compareRes, setCompareRes] = useState<HfssCompareResponse | null>(null);

  const features = state.dataset?.inputColumns ?? [];
  const selectedModel = modelSel || state.models[0] || "";

  useEffect(() => {
    if (!state.models[0] && !modelSel) return;
    const ranges = state.dataset?.metadata?.numeric_ranges ?? {};
    const init: Record<string, number> = {};
    for (const f of features) {
      const r = ranges[f];
      init[f] = r ? (r.min + r.max) / 2 : 0;
    }
    setValues(init);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.models, features.join(",")]);

  const submitSimple = async () => {
    if (!file) { setError("Choose an HFSS result file."); return; }
    setLoading(true);
    const form = new FormData();
    form.append("result", file);
    try {
      setSimpleRes(await api<HfssSimpleResponse>("/api/hfss", { method: "POST", body: form }));
      setCompareRes(null);
    } catch (e) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const submitCompare = async () => {
    if (!file) { setError("Choose an HFSS result file."); return; }
    if (!selectedModel) { setError("Select a trained model."); return; }
    setLoading(true);
    const form = new FormData();
    form.append("result", file);
    form.append("model_name", selectedModel);
    form.append("values_json", JSON.stringify(values));
    try {
      setCompareRes(await api<HfssCompareResponse>("/api/hfss/compare", { method: "POST", body: form }));
      setSimpleRes(null);
    } catch (e) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const chartData = compareRes
    ? compareRes.frequency_hz.map((f, i) => ({
        freq: f / 1e9,
        predicted: compareRes.predicted_s11_db[i],
        hfss: compareRes.hfss_s11_db[i],
      }))
    : [];

  return (
    <>
      <Intro title="HFSS validation" text="Upload an actual HFSS S11 export and compare it against ML predictions. The application never fabricates simulation data." />

      <Alert type="warning">
        Automated HFSS execution requires PyAEDT and a licensed Ansys Electronics Desktop installation.
        Only manually exported CSV/XLSX files are accepted here.
      </Alert>

      <section className="card">
        <div style={{ display: "flex", gap: 10, marginBottom: 18 }}>
          <button className={`secondary${mode === "compare" ? " tab-active" : ""}`} onClick={() => setMode("compare")}>
            ML vs HFSS comparison
          </button>
          <button className={`secondary${mode === "simple" ? " tab-active" : ""}`} onClick={() => setMode("simple")}>
            Import HFSS only
          </button>
        </div>

        <FileBox title="HFSS result file (CSV or XLSX)" required file={file}
          onChange={(e) => setFile(e.target.files?.[0])} />

        {mode === "compare" && state.models.length > 0 && (
          <>
            <div className="form-row" style={{ marginTop: 16 }}>
              <label>ML model for comparison
                <select value={selectedModel} onChange={(e) => setModelSel(e.target.value)}>
                  {state.models.map((m) => <option key={m}>{m}</option>)}
                </select>
              </label>
            </div>
            <div className="input-grid">
              {features.map((f) => (
                <label key={f}>{f}
                  <input type="number" step="any" value={values[f] ?? 0}
                    onChange={(e) => setValues({ ...values, [f]: Number(e.target.value) })} />
                </label>
              ))}
            </div>
          </>
        )}

        <button className="primary" onClick={mode === "compare" ? submitCompare : submitSimple} disabled={loading}>
          {loading ? <><Spinner label="" /> Processing…</> : <>Upload and validate <span>→</span></>}
        </button>
      </section>

      {simpleRes && (
        <section className="card">
          <Alert type="info">{simpleRes.validation_status}</Alert>
          <SectionTitle title="HFSS data preview" detail={`${simpleRes.total} rows`} />
          <div className="table-wrap">
            <table>
              <thead><tr>{simpleRes.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>
                {simpleRes.rows.slice(0, 10).map((row, i) => (
                  <tr key={i}>{simpleRes.columns.map((c) => <td key={c}>{String(row[c] ?? "—")}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {compareRes && (
        <>
          <Alert type={compareRes.mae_db < 2 ? "success" : "warning"}>
            {compareRes.validation_status} — MAE: <b>{fmt(compareRes.mae_db, 3)} dB</b> · RMSE: <b>{fmt(compareRes.rmse_db, 3)} dB</b>
          </Alert>

          <div className="metric-grid">
            <Metric label="ML min S11"      value={compareRes.predicted_metrics?.minimum_s11_db != null ? `${fmt(compareRes.predicted_metrics.minimum_s11_db, 3)} dB` : "—"} />
            <Metric label="HFSS min S11"    value={compareRes.hfss_metrics?.minimum_s11_db != null ? `${fmt(compareRes.hfss_metrics.minimum_s11_db, 3)} dB` : "—"} />
            <Metric label="ML resonant freq" value={compareRes.predicted_metrics?.resonant_frequency_hz ? fmtGHz(compareRes.predicted_metrics.resonant_frequency_hz) : "—"} />
            <Metric label="HFSS resonant freq" value={compareRes.hfss_metrics?.resonant_frequency_hz ? fmtGHz(compareRes.hfss_metrics.resonant_frequency_hz) : "—"} />
            <Metric label="ML bandwidth"    value={compareRes.predicted_metrics?.bandwidth_hz ? fmtMHz(compareRes.predicted_metrics.bandwidth_hz) : "—"} />
            <Metric label="HFSS bandwidth"  value={compareRes.hfss_metrics?.bandwidth_hz ? fmtMHz(compareRes.hfss_metrics.bandwidth_hz) : "—"} />
            <Metric label="MAE"             value={fmt(compareRes.mae_db, 3)} unit="dB" />
            <Metric label="RMSE"            value={fmt(compareRes.rmse_db, 3)} unit="dB" />
          </div>

          <section className="card">
            <SectionTitle title="S11 comparison: ML prediction vs HFSS" />
            <div className="chart">
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" />
                  <XAxis dataKey="freq" tick={{ fontSize: 10 }} tickFormatter={(v) => v.toFixed(2)}
                    label={{ value: "Frequency (GHz)", position: "insideBottom", offset: -4, fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 11 }} label={{ value: "S11 (dB)", angle: -90, position: "insideLeft", fontSize: 10 }} />
                  <Tooltip formatter={(v: number, n: string) => [`${v.toFixed(3)} dB`, n]} labelFormatter={(l: number) => `${l.toFixed(4)} GHz`} />
                  <Legend />
                  <ReferenceLine y={-10} stroke="#aaa" strokeDasharray="5 5" />
                  <Line type="monotone" dataKey="predicted" name="ML predicted" stroke={TEAL}   strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="hfss"      name="HFSS actual"  stroke={ORANGE} strokeWidth={2} dot={false} strokeDasharray="6 3" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="card">
            <SectionTitle title="Frequency-wise prediction error" detail={`Error = ML − HFSS · ${compareRes.frequency_hz.length} points`} />
            <div className="chart">
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData.map((d) => ({ freq: d.freq, error: d.predicted - d.hfss }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" />
                  <XAxis dataKey="freq" tick={{ fontSize: 10 }} tickFormatter={(v) => v.toFixed(2)} />
                  <YAxis tick={{ fontSize: 11 }} label={{ value: "Error (dB)", angle: -90, position: "insideLeft", fontSize: 10 }} />
                  <Tooltip formatter={(v: number) => [`${v.toFixed(3)} dB`, "Error"]} />
                  <ReferenceLine y={0} stroke="#aaa" />
                  <Line type="monotone" dataKey="error" stroke="#6c8ebf" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
        </>
      )}

      {!state.models.length && mode === "compare" && (
        <Alert type="info">Train an ML model first to enable HFSS vs ML comparison.</Alert>
      )}
    </>
  );
}

// ─── 9. REPORTS & RESULTS ─────────────────────────────────────────────────────

function ReportsPage({ state, setError }: PageProps) {
  const [loading, setLoading] = useState<Record<string, boolean>>({});

  const download = async (url: string, filename: string) => {
    setLoading((prev) => ({ ...prev, [filename]: true }));
    try {
      const blob = await api<Blob>(url);
      downloadBlob(blob, filename);
    } catch (e) { setError((e as Error).message); }
    finally { setLoading((prev) => ({ ...prev, [filename]: false })); }
  };

  const reports = [
    {
      icon: "▤",
      title: "Complete research report",
      description: state.dataset
        ? `${state.dataset.rows.toLocaleString()} samples · ${state.models.length} trained models`
        : "Load a dataset to generate.",
      action: () => download("/api/report", "antenna_report.pdf"),
      filename: "antenna_report.pdf",
      disabled: !state.dataset,
      format: "PDF",
    },
  ];

  return (
    <>
      <Intro title="Reports & results" text="Download evidence-based reports generated from your uploaded data and trained models. Only information derived from real data is included." />

      {!state.dataset && (
        <Alert type="info">Load a dataset first to enable report generation.</Alert>
      )}

      <div style={{ display: "grid", gap: 14 }}>
        {reports.map((r) => (
          <section key={r.title} className="card report-card">
            <div className="report-icon">{r.icon}</div>
            <div>
              <h3>{r.title}</h3>
              <p>{r.description}</p>
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginLeft: "auto" }}>
              <span style={{ fontSize: 10, color: "#7e9095", fontFamily: "DM Mono, monospace", fontWeight: 700 }}>{r.format}</span>
              <button className="secondary" disabled={r.disabled || loading[r.filename]} onClick={r.action}>
                {loading[r.filename] ? "Downloading…" : "↓ Download"}
              </button>
            </div>
          </section>
        ))}
      </div>

      {state.dataset && (
        <section className="card" style={{ marginTop: 20 }}>
          <SectionTitle title="Current workspace summary" />
          <div className="metric-grid">
            <Metric label="Dataset samples"  value={state.dataset.rows.toLocaleString()} />
            <Metric label="Input features"   value={state.dataset.inputColumns.length} />
            <Metric label="S11 targets"      value={state.dataset.targetColumns.length} />
            <Metric label="Trained models"   value={state.models.length} />
            {state.evaluation && <>
              <Metric label="Best R²"        value={fmt(state.evaluation[0]?.r2 ?? 0)} />
              <Metric label="Best RMSE"      value={fmt(state.evaluation[0]?.rmse ?? 0)} />
            </>}
            <Metric label="Optimizations run" value={Object.keys(state.optimizations).length} />
          </div>
        </section>
      )}
    </>
  );
}

// ─── Export ────────────────────────────────────────────────────────────────────
export default App;
