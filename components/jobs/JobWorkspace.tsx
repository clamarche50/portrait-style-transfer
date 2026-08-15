"use client";

import Link from "next/link";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Download, RefreshCw, Trash2 } from "lucide-react";
import { deleteJob, getJob, getJobDownloadUrl, rerunJob } from "@/lib/api/client";
import type { JobRecord } from "@/lib/api/types";
import { DiagnosticsPanel } from "@/components/editor/DiagnosticsPanel";
import { PortraitComparison } from "@/components/editor/PortraitComparison";
import { ProgressTimeline } from "@/components/editor/ProgressTimeline";

export function JobWorkspace({ jobId }: { jobId: string }) {
  const queryClient = useQueryClient();
  const query = useQuery({ queryKey: ["job", jobId], queryFn: () => getJob(jobId), refetchInterval: (result) => ["QUEUED", "RUNNING", "CANCEL_REQUESTED"].includes(result.state.data?.status ?? "") ? 2_000 : false });
  const job = query.data;

  async function download() {
    const signed = await getJobDownloadUrl(jobId);
    window.location.assign(signed.url);
  }

  function replaceJob(nextJob: JobRecord) { queryClient.setQueryData(["job", jobId], nextJob); }
  async function retry() { replaceJob(await rerunJob(jobId)); }
  async function remove() { if (window.confirm("Permanently delete this job and every related image?")) { await deleteJob(jobId); window.location.assign("/"); } }

  if (query.isLoading) return <main id="main-content" className="page-container"><div className="loading-state"><span /> Loading private job…</div></main>;
  if (query.error || !job) return <main id="main-content" className="page-container"><div className="error-state"><h1>Job unavailable</h1><p>It may have expired, been deleted, or belong to another session.</p><Link className="button button--primary" href="/">Return to editor</Link></div></main>;

  return (
    <main id="main-content" className="page-container job-page">
      <div className="page-topline"><Link href="/"><ArrowLeft size={16} aria-hidden="true" /> Editor</Link><span>Job {job.id.slice(0, 8)}</span></div>
      <div className="page-heading"><div><span className="eyebrow">Private result</span><h1>Portrait workspace</h1></div><div><button className="button button--ghost" type="button" onClick={retry}><RefreshCw size={16} aria-hidden="true" /> Rerun</button><button className="button button--ghost danger" type="button" onClick={remove}><Trash2 size={16} aria-hidden="true" /> Delete</button><button className="button button--primary" type="button" onClick={download} disabled={job.status !== "SUCCEEDED"}><Download size={16} aria-hidden="true" /> Download</button></div></div>
      <ProgressTimeline job={job} />
      {job.status === "SUCCEEDED" && job.input_preview_url && job.output_url && <PortraitComparison job={job} inputUrl={job.input_preview_url} outputUrl={job.output_url} onDownload={download} onRetry={retry} onDelete={remove} />}
      <DiagnosticsPanel job={job} />
    </main>
  );
}
