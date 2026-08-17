"use client";

import { Crosshair, Eye, Paintbrush, Save, SlidersHorizontal } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { rerunJob, saveCorrections } from "@/lib/api/client";
import type { CorrectionPayload, JobRecord } from "@/lib/api/types";

type Tool = "mask" | "alignment" | "gain" | "eye";
type Point = [number, number];

const tools: Array<{ id: Tool; label: string; icon: typeof Paintbrush }> = [
  { id: "mask", label: "Matte", icon: Paintbrush },
  { id: "alignment", label: "Alignment", icon: Crosshair },
  { id: "gain", label: "Local gain", icon: SlidersHorizontal },
  { id: "eye", label: "Eyes", icon: Eye },
];

export function CorrectionStudio({ job, previewUrl, onRerun }: { job: JobRecord; previewUrl?: string | null; onRerun: (job: JobRecord) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tool, setTool] = useState<Tool>("mask");
  const [operation, setOperation] = useState<"ADD" | "REMOVE">("ADD");
  const [radius, setRadius] = useState(18);
  const [strokes, setStrokes] = useState<Array<{ operation: "ADD" | "REMOVE"; radius: number; points: Point[] }>>([]);
  const [activeStroke, setActiveStroke] = useState<Point[] | null>(null);
  const [inputPoint, setInputPoint] = useState<Point>([0.5, 0.5]);
  const [referencePoint, setReferencePoint] = useState<Point>([0.5, 0.5]);
  const [eyeSide, setEyeSide] = useState<"left" | "right">("left");
  const [eyeCenter, setEyeCenter] = useState<Point>([0.35, 0.43]);
  const [eyeScale, setEyeScale] = useState(1);
  const [gainSource, setGainSource] = useState<Point>([0.35, 0.55]);
  const [gainTarget, setGainTarget] = useState<Point>([0.65, 0.55]);
  const [gainSize, setGainSize] = useState(0.12);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    for (const stroke of [...strokes, ...(activeStroke ? [{ operation, radius, points: activeStroke }] : [])]) {
      if (stroke.points.length === 0) continue;
      context.beginPath();
      context.lineCap = "round";
      context.lineJoin = "round";
      context.lineWidth = stroke.radius * 2;
      context.strokeStyle = stroke.operation === "ADD" ? "rgba(190, 244, 111, .7)" : "rgba(255, 109, 92, .72)";
      stroke.points.forEach(([x, y], index) => {
        const px = x * canvas.width;
        const py = y * canvas.height;
        if (index === 0) context.moveTo(px, py); else context.lineTo(px, py);
      });
      context.stroke();
    }
  }, [strokes, activeStroke, operation, radius]);

  function pointFromEvent(event: React.PointerEvent<HTMLCanvasElement>): Point {
    const rect = event.currentTarget.getBoundingClientRect();
    return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))];
  }

  function corrections(): CorrectionPayload[] {
    if (tool === "mask") return strokes.map((stroke) => ({ ...stroke, type: "mask", radius: stroke.radius / 640 }));
    if (tool === "alignment") return [{ type: "alignment", input_points: [inputPoint], reference_points: [referencePoint] }];
    if (tool === "gain") {
      const polygon = ([x, y]: Point): Point[] => [
        [Math.max(0, x - gainSize), Math.max(0, y - gainSize)],
        [Math.min(1, x + gainSize), Math.max(0, y - gainSize)],
        [Math.min(1, x + gainSize), Math.min(1, y + gainSize)],
        [Math.max(0, x - gainSize), Math.min(1, y + gainSize)],
      ];
      return [{ type: "gain_copy", source_polygon: polygon(gainSource), target_polygon: polygon(gainTarget), levels: [0, 1, 2, 3, 4, 5] }];
    }
    return [{ type: "eye", eye: eyeSide.toUpperCase(), pupil_center: eyeCenter, highlight_scale: eyeScale }];
  }

  async function saveAndRerun() {
    setSaving(true); setMessage(null);
    try {
      await saveCorrections(job.id, corrections());
      const rerun = await rerunJob(job.id);
      onRerun(rerun);
      setMessage("Correction saved. The job restarted from the earliest affected stage.");
    } catch (reason) { setMessage(reason instanceof Error ? reason.message : "The correction could not be saved."); }
    finally { setSaving(false); }
  }

  return (
    <section className="correction-card" aria-labelledby="correction-title">
      <div className="section-heading">
        <div><span className="eyebrow">Advanced correction</span><h2 id="correction-title">Refine only what missed</h2></div>
        <p>Edits are non-destructive and invalidate only the required processing stages.</p>
      </div>
      <div className="correction-layout">
        <div className="correction-tools" role="tablist" aria-label="Correction type">
          {tools.map(({ id, label, icon: Icon }) => <button key={id} type="button" role="tab" aria-selected={tool === id} className={tool === id ? "is-active" : ""} onClick={() => setTool(id)}><Icon size={17} aria-hidden="true" />{label}</button>)}
        </div>
        <div className="correction-canvas-wrap">
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={previewUrl} alt="Portrait correction preview" />
          ) : <span className="correction-placeholder">Portrait preview</span>}
          {tool === "mask" && <canvas
            ref={canvasRef}
            width={640}
            height={640}
            aria-label="Paint head matte corrections"
            onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); setActiveStroke([pointFromEvent(event)]); }}
            onPointerMove={(event) => { if (activeStroke) setActiveStroke([...activeStroke, pointFromEvent(event)]); }}
            onPointerUp={() => { if (activeStroke?.length) setStrokes([...strokes, { operation, radius, points: activeStroke }]); setActiveStroke(null); }}
          />}
          {tool === "alignment" && <div className="point-preview" aria-hidden="true"><span style={{ left: `${inputPoint[0] * 100}%`, top: `${inputPoint[1] * 100}%` }} /><span style={{ left: `${referencePoint[0] * 100}%`, top: `${referencePoint[1] * 100}%` }} /></div>}
        </div>
        <div className="correction-options">
          {tool === "mask" && <>
            <h3>Head matte</h3><p>Paint areas the automatic matte should include or exclude.</p>
            <div className="segmented-control"><button type="button" className={operation === "ADD" ? "is-active" : ""} onClick={() => setOperation("ADD")}>Add</button><button type="button" className={operation === "REMOVE" ? "is-active" : ""} onClick={() => setOperation("REMOVE")}>Remove</button></div>
            <label className="range-control"><span className="range-control__label"><span>Brush size</span><output>{radius}px</output></span><input type="range" min="4" max="64" value={radius} onChange={(event) => setRadius(Number(event.target.value))} /></label>
            <button className="text-button" type="button" onClick={() => setStrokes(strokes.slice(0, -1))} disabled={strokes.length === 0}>Undo last stroke</button>
          </>}
          {tool === "alignment" && <>
            <h3>Feature pair</h3><p>Add a high-weight matching point when an edge lands in the wrong place.</p>
            <label>X on input <input type="number" min="0" max="1" step="0.01" value={inputPoint[0]} onChange={(e) => setInputPoint([Number(e.target.value), inputPoint[1]])} /></label>
            <label>Y on input <input type="number" min="0" max="1" step="0.01" value={inputPoint[1]} onChange={(e) => setInputPoint([inputPoint[0], Number(e.target.value)])} /></label>
            <label>X on reference <input type="number" min="0" max="1" step="0.01" value={referencePoint[0]} onChange={(e) => setReferencePoint([Number(e.target.value), referencePoint[1]])} /></label>
            <label>Y on reference <input type="number" min="0" max="1" step="0.01" value={referencePoint[1]} onChange={(e) => setReferencePoint([referencePoint[0], Number(e.target.value)])} /></label>
          </>}
          {tool === "gain" && <>
            <h3>Copy local gain</h3><p>Copy the multiscale gain from a reliable source patch into a mismatched target patch.</p>
            <label>Source X <input type="number" min="0" max="1" step="0.01" value={gainSource[0]} onChange={(e) => setGainSource([Number(e.target.value), gainSource[1]])} /></label>
            <label>Source Y <input type="number" min="0" max="1" step="0.01" value={gainSource[1]} onChange={(e) => setGainSource([gainSource[0], Number(e.target.value)])} /></label>
            <label>Target X <input type="number" min="0" max="1" step="0.01" value={gainTarget[0]} onChange={(e) => setGainTarget([Number(e.target.value), gainTarget[1]])} /></label>
            <label>Target Y <input type="number" min="0" max="1" step="0.01" value={gainTarget[1]} onChange={(e) => setGainTarget([gainTarget[0], Number(e.target.value)])} /></label>
            <label>Region radius <input type="number" min="0.02" max="0.3" step="0.01" value={gainSize} onChange={(e) => setGainSize(Number(e.target.value))} /></label>
            <div className="gain-key"><span />Applied to all six paper scales</div>
          </>}
          {tool === "eye" && <>
            <h3>Eye highlight</h3><p>Correct catchlight placement without repeating alignment or multiscale transfer.</p>
            <div className="segmented-control"><button type="button" className={eyeSide === "left" ? "is-active" : ""} onClick={() => { setEyeSide("left"); setEyeCenter([0.35, eyeCenter[1]]); }}>Left</button><button type="button" className={eyeSide === "right" ? "is-active" : ""} onClick={() => { setEyeSide("right"); setEyeCenter([0.65, eyeCenter[1]]); }}>Right</button></div>
            <label>Pupil X <input type="number" min="0" max="1" step="0.01" value={eyeCenter[0]} onChange={(e) => setEyeCenter([Number(e.target.value), eyeCenter[1]])} /></label>
            <label>Pupil Y <input type="number" min="0" max="1" step="0.01" value={eyeCenter[1]} onChange={(e) => setEyeCenter([eyeCenter[0], Number(e.target.value)])} /></label>
            <label>Highlight scale <input type="number" min="0.5" max="1.5" step="0.05" value={eyeScale} onChange={(e) => setEyeScale(Number(e.target.value))} /></label>
          </>}
          <button className="button button--primary" type="button" onClick={saveAndRerun} disabled={saving || (tool === "mask" && strokes.length === 0)}><Save size={16} aria-hidden="true" />{saving ? "Saving…" : "Save & rerun"}</button>
          {message && <p className="correction-message" role="status">{message}</p>}
        </div>
      </div>
    </section>
  );
}
