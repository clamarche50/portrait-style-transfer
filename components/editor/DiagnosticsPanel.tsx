"use client";

import { Activity, ChevronDown, CircleAlert } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { getJobDiagnostics } from "@/lib/api/client";
import type { JobRecord, QualityWarning } from "@/lib/api/types";

function humanize(value: string) {
  return value.toLowerCase().replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function warningParts(warning: QualityWarning | string) {
  return typeof warning === "string"
    ? { code: warning, message: humanize(warning) }
    : warning;
}

export function DiagnosticsPanel({ job }: { job: JobRecord }) {
  const query = useQuery({
    queryKey: ["job-diagnostics", job.id],
    queryFn: () => getJobDiagnostics(job.id),
    enabled: ["SUCCEEDED", "FAILED", "CANCELLED"].includes(job.status),
    retry: false,
  });
  const artifacts = query.data
    ? query.data.artifacts.map((artifact) => ({
        id: artifact.asset_id,
        kind: artifact.kind,
        label: undefined,
        url: artifact.download_url ?? undefined,
      }))
    : (job.artifacts ?? []);
  const diagnostics = query.data?.diagnostics ?? job.diagnostics ?? job.diagnostics_summary;
  const warnings = job.warnings ?? [];
  return (
    <details className="diagnostics-card">
      <summary>
        <span><Activity size={18} aria-hidden="true" /><span><strong>Processing diagnostics</strong><small>Alignment, masks, energy, and fallbacks</small></span></span>
        <span className="diagnostic-count">{artifacts.length} artifacts</span>
        <ChevronDown size={18} aria-hidden="true" />
      </summary>
      <div className="diagnostics-body">
        {warnings.length > 0 && (
          <div className="warning-list">
            {warnings.map((warning) => {
              const item = warningParts(warning);
              return <p key={`${item.code}-${item.message}`}><CircleAlert size={16} aria-hidden="true" /><span><strong>{humanize(item.code)}</strong>{item.message}</span></p>;
            })}
          </div>
        )}
        {query.isLoading ? <p className="empty-copy">Loading private diagnostics…</p> : artifacts.length > 0 ? (
          <div className="artifact-grid">
            {artifacts.map((artifact, index) => (
              <figure key={artifact.id ?? `${artifact.kind}-${index}`}>
                {artifact.url ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={artifact.url} alt={`${artifact.label ?? humanize(artifact.kind)} diagnostic`} />
                ) : <span className="artifact-placeholder" />}
                <figcaption>{artifact.label ?? humanize(artifact.kind)}</figcaption>
              </figure>
            ))}
          </div>
        ) : <p className="empty-copy">Enable “Save diagnostics” before a run to retain private intermediate previews. They expire with the job.</p>}
        {query.isError && <p className="inline-error" role="alert">Private diagnostics could not be loaded.</p>}
        {diagnostics && <pre className="diagnostic-json">{JSON.stringify(diagnostics, null, 2)}</pre>}
      </div>
    </details>
  );
}
