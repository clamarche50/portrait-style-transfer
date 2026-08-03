"use client";

import { CheckCircle2, CircleAlert, ScanFace } from "lucide-react";
import type { AssetRecord } from "@/lib/api/types";

export function CompatibilityCard({ input, reference }: { input: AssetRecord; reference: AssetRecord }) {
  const warnings = [...(input.analysis?.warnings ?? []), ...(reference.analysis?.warnings ?? [])];
  const analyzed = typeof input.analysis?.quality_score === "number" && typeof reference.analysis?.quality_score === "number";
  const score = analyzed ? Math.round((input.analysis!.quality_score! + reference.analysis!.quality_score!) * 50) : null;
  return (
    <section className="compatibility-card" aria-labelledby="compatibility-title">
      <div>
        <span className="compatibility-icon"><ScanFace size={20} aria-hidden="true" /></span>
        <span><strong id="compatibility-title">Portrait compatibility</strong><small>Pose, detail, crop, and matte confidence</small></span>
      </div>
      <div className="compatibility-score">{score === null ? <strong>Queued</strong> : <><strong>{score}</strong><span>/ 100</span></>}</div>
      <div className="compatibility-message">
        {!analyzed ? <><ScanFace size={17} aria-hidden="true" /> Pair analysis runs before transfer</> : warnings.length === 0 ? <><CheckCircle2 size={17} aria-hidden="true" /> No compatibility warning was found</> : <><CircleAlert size={17} aria-hidden="true" /> {warnings[0].message}</>}
      </div>
    </section>
  );
}
