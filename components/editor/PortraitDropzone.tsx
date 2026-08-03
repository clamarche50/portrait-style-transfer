"use client";

import { AlertTriangle, CheckCircle2, ImagePlus, ScanFace, X } from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { validatePortraitFile } from "@/lib/validation/portrait";

interface PortraitDropzoneProps {
  eyebrow: string;
  title: string;
  description: string;
  file: File | null;
  onFile: (file: File | null) => void;
  tone: "warm" | "cool";
}

export function PortraitDropzone({ eyebrow, title, description, file, onFile, tone }: PortraitDropzoneProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const preview = useMemo(() => file ? URL.createObjectURL(file) : null, [file]);

  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview); };
  }, [preview]);

  function accept(next: File | undefined) {
    if (!next) return;
    const validationError = validatePortraitFile(next);
    setError(validationError);
    if (!validationError) onFile(next);
  }

  return (
    <section className={`upload-card upload-card--${tone}`} aria-labelledby={`${inputId}-title`}>
      <div className="upload-card__topline">
        <span>{eyebrow}</span>
        <span className="file-rule">JPEG · PNG · WEBP</span>
      </div>
      <button
        className={`dropzone ${dragging ? "is-dragging" : ""} ${file ? "has-file" : ""}`}
        type="button"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
        }}
        onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => { event.preventDefault(); setDragging(false); }}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          accept(event.dataTransfer.files[0]);
        }}
        aria-describedby={`${inputId}-description ${error ? `${inputId}-error` : ""}`}
      >
        {preview ? (
          <>
            {/* User-selected local preview; uploaded only after explicit submission. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img className="dropzone__preview" src={preview} alt="Selected portrait preview" />
            <span className="face-guide" aria-hidden="true"><span /></span>
            <span className="dropzone__file-pill">
              <CheckCircle2 size={15} aria-hidden="true" /> Ready to analyze
            </span>
          </>
        ) : (
          <div className="dropzone__empty">
            <span className="dropzone__icon"><ImagePlus size={24} strokeWidth={1.7} aria-hidden="true" /></span>
            <h2 id={`${inputId}-title`}>{title}</h2>
            <p id={`${inputId}-description`}>{description}</p>
            <span className="text-action">Choose portrait <span aria-hidden="true">↗</span></span>
          </div>
        )}
      </button>
      <input
        ref={inputRef}
        id={inputId}
        className="visually-hidden"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={(event) => accept(event.target.files?.[0])}
      />
      {file && (
        <div className="file-meta">
          <span><ScanFace size={15} aria-hidden="true" /> {file.name}</span>
          <span>{(file.size / 1024 / 1024).toFixed(1)} MB</span>
          <button type="button" onClick={() => { onFile(null); setError(null); }} aria-label={`Remove ${title.toLowerCase()}`}>
            <X size={15} aria-hidden="true" />
          </button>
        </div>
      )}
      {error && <p className="inline-error" id={`${inputId}-error`} role="alert"><AlertTriangle size={15} aria-hidden="true" /> {error}</p>}
    </section>
  );
}
