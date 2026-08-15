"use client";

import { Download, RotateCcw, Trash2, ZoomIn } from "lucide-react";
import { useState } from "react";
import type { JobRecord } from "@/lib/api/types";

interface PortraitComparisonProps {
  job: JobRecord;
  inputUrl: string;
  outputUrl: string;
  onDownload: () => void;
  onRetry: () => void;
  onDelete: () => void;
}

export function PortraitComparison({ job, inputUrl, outputUrl, onDownload, onRetry, onDelete }: PortraitComparisonProps) {
  const [split, setSplit] = useState(52);
  const [zoom, setZoom] = useState(1);

  return (
    <section className="result-card" aria-labelledby="result-title">
      <div className="result-card__heading">
        <div><span className="eyebrow">AI-generated edit</span><h2 id="result-title">Style transferred. Review the likeness.</h2></div>
        <span className="result-status">InstantStyle AI · {job.settings.output_format}</span>
      </div>
      <p className="result-disclosure">Diffusion can alter facial details, age cues, accessories, or expression. Compare the result with the source before publishing or using it.</p>
      <div className="comparison-stage" style={{ "--portrait-zoom": zoom } as React.CSSProperties}>
        {/* Presigned private image URLs. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={inputUrl} alt="Original portrait" />
        <div className="comparison-stage__after" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={outputUrl} alt="Style-transferred portrait" />
        </div>
        <span className="comparison-label comparison-label--before">Before</span>
        <span className="comparison-label comparison-label--after">After</span>
        <span className="comparison-divider" style={{ left: `${split}%` }} aria-hidden="true"><span>↔</span></span>
        <input className="comparison-slider" type="range" min="0" max="100" value={split} onChange={(event) => setSplit(Number(event.target.value))} aria-label="Reveal before and after" />
      </div>
      <div className="result-toolbar">
        <label className="zoom-control"><ZoomIn size={16} aria-hidden="true" /> Zoom <input type="range" min="1" max="2" step="0.1" value={zoom} onChange={(event) => setZoom(Number(event.target.value))} /></label>
        <div>
          <button className="button button--ghost" type="button" onClick={onRetry}><RotateCcw size={16} aria-hidden="true" /> Retry settings</button>
          <button className="button button--ghost danger" type="button" onClick={onDelete}><Trash2 size={16} aria-hidden="true" /> Delete</button>
          <button className="button button--primary" type="button" onClick={onDownload}><Download size={17} aria-hidden="true" /> Download {job.settings.output_format}</button>
        </div>
      </div>
    </section>
  );
}
