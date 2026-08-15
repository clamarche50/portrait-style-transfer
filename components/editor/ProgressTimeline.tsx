"use client";

import { Check, LoaderCircle } from "lucide-react";
import type { JobRecord } from "@/lib/api/types";

const stages = [
  ["VALIDATING", "Validate"],
  ["FACE_LANDMARKS", "Map features"],
  ["SEGMENTATION", "Build matte"],
  ["AI_GENERATION", "Generate portrait"],
  ["BACKGROUND", "Compose background"],
  ["POSTPROCESSING", "Finish"],
] as const;

const stageRank: Record<string, number> = {
  VALIDATING: 0, DECODING: 0, FACE_LANDMARKS: 1, QUALITY_ANALYSIS: 1,
  SEGMENTATION: 2, REFERENCE_SELECTION: 2, AI_GENERATION: 3,
  BACKGROUND: 4, POSTPROCESSING: 5,
  UPLOADING_OUTPUT: 5, COMPLETED: 6,
};

export function ProgressTimeline({ job }: { job: JobRecord }) {
  const active = job.status === "SUCCEEDED" ? stages.length : (stageRank[job.stage] ?? 0);
  return (
    <section className="progress-card" aria-live="polite" aria-label="Portrait processing progress">
      <div className="progress-card__heading">
        <div>
          <span className="eyebrow">Making your portrait</span>
          <h2>{job.status === "SUCCEEDED" ? "Your finish is ready" : job.status === "FAILED" ? "Processing stopped" : "Generating your styled portrait"}</h2>
        </div>
        <strong>{Math.round(job.progress)}%</strong>
      </div>
      <div className="progress-track"><span style={{ width: `${job.progress}%` }} /></div>
      <ol className="stage-list">
        {stages.map(([key, label], index) => {
          const done = active > index;
          const current = active === index && job.status !== "FAILED";
          return (
            <li key={key} className={done ? "is-done" : current ? "is-current" : ""}>
              <span>{done ? <Check size={14} aria-hidden="true" /> : current ? <LoaderCircle size={14} aria-hidden="true" /> : index + 1}</span>
              {label}
            </li>
          );
        })}
      </ol>
      {job.status === "FAILED" && <p className="job-error" role="alert">{job.error_message_safe ?? "This pair could not be processed safely."}</p>}
    </section>
  );
}
