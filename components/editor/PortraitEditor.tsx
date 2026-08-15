"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowRight, CircleAlert, LockKeyhole, ShieldCheck, Sparkles } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, createJob, deleteJob, getJob, getJobDownloadUrl, listStyles, subscribeToJob, uploadAsset } from "@/lib/api/client";
import type { AssetRecord, JobRecord, StyleRecord, TransferSettings } from "@/lib/api/types";
import { defaultSettings, settingsSchema } from "@/lib/validation/portrait";
import { CompatibilityCard } from "./CompatibilityCard";
import { DiagnosticsPanel } from "./DiagnosticsPanel";
import { PortraitComparison } from "./PortraitComparison";
import { PortraitDropzone } from "./PortraitDropzone";
import { ProgressTimeline } from "./ProgressTimeline";
import { SettingsPanel } from "./SettingsPanel";

type ReferenceMode = "upload" | "library";

export function PortraitEditor() {
  const router = useRouter();
  const [inputFile, setInputFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [referenceMode, setReferenceMode] = useState<ReferenceMode>("upload");
  const [selectedStyle, setSelectedStyle] = useState<string | null>(null);
  const [settings, setSettings] = useState<TransferSettings>(defaultSettings);
  const [assets, setAssets] = useState<{ input: AssetRecord; reference?: AssetRecord } | null>(null);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const styles = useQuery({ queryKey: ["styles"], queryFn: listStyles, retry: false });
  const inputPreview = useMemo(() => inputFile ? URL.createObjectURL(inputFile) : null, [inputFile]);
  const liveJobId = job?.id;
  const liveJobStatus = job?.status;

  const canSubmit = Boolean(inputFile && (referenceMode === "upload" ? referenceFile : selectedStyle) && !submitting);

  useEffect(() => {
    return () => { if (inputPreview) URL.revokeObjectURL(inputPreview); };
  }, [inputPreview]);

  useEffect(() => {
    if (!liveJobId || !liveJobStatus || !["QUEUED", "RUNNING", "CANCEL_REQUESTED"].includes(liveJobStatus)) return;
    let fallbackTimer: ReturnType<typeof setInterval> | null = null;
    const cleanup = subscribeToJob(
      liveJobId,
      (update) => {
        setJob(update);
        if (update.status === "SUCCEEDED" && update.output_url) setOutputUrl(update.output_url);
      },
      () => {
        if (!fallbackTimer) {
          fallbackTimer = setInterval(async () => {
            try {
              const update = await getJob(liveJobId);
              setJob(update);
              if (update.status === "SUCCEEDED" && update.output_url) setOutputUrl(update.output_url);
              if (!["QUEUED", "RUNNING", "CANCEL_REQUESTED"].includes(update.status) && fallbackTimer) clearInterval(fallbackTimer);
            } catch { /* A later poll or explicit retry can recover. */ }
          }, 2_000);
        }
      },
    );
    return () => { cleanup(); if (fallbackTimer) clearInterval(fallbackTimer); };
  }, [liveJobId, liveJobStatus]);

  const privacyLine = useMemo(() => (
    <span className="privacy-line"><LockKeyhole size={14} aria-hidden="true" /> Nothing uploads until you click Create portrait. Files are private and expire after 24 hours.</span>
  ), []);

  async function submit() {
    if (!inputFile || (referenceMode === "upload" && !referenceFile) || (referenceMode === "library" && !selectedStyle)) return;
    setSubmitting(true); setError(null); setOutputUrl(null);
    try {
      const parsedSettings = settingsSchema.parse(settings);
      const inputAsset = await uploadAsset(inputFile, "INPUT");
      let referenceAsset: AssetRecord | undefined;
      if (referenceMode === "upload" && referenceFile) referenceAsset = await uploadAsset(referenceFile, "REFERENCE");
      setAssets({ input: inputAsset, reference: referenceAsset });
      const created = await createJob({
        input_asset_id: inputAsset.id,
        ...(referenceAsset ? { reference_asset_id: referenceAsset.id } : { style_id: selectedStyle! }),
        settings: parsedSettings,
      });
      setJob(created);
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : "We could not start this portrait. Check the images and try again.");
    } finally { setSubmitting(false); }
  }

  async function download() {
    if (!job) return;
    try {
      const signed = await getJobDownloadUrl(job.id);
      window.location.assign(signed.url);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "The download link could not be created."); }
  }

  async function remove() {
    if (!job || !window.confirm("Delete this job and all of its uploaded and generated images?")) return;
    try { await deleteJob(job.id); setJob(null); setAssets(null); setOutputUrl(null); } catch (reason) { setError(reason instanceof Error ? reason.message : "Deletion failed."); }
  }

  return (
    <>
      <section className="editor-hero">
        <div className="editor-hero__copy">
          <p className="kicker"><span /> Research-grade portrait finishing</p>
          <h1>Move the light.<br /><em>Shape the finish.</em></h1>
          <p>Guide color, texture, and illumination with a reference headshot while using the source pose and facial geometry as structure. This is a generative edit: review likeness and details before use.</p>
        </div>
        <div className="method-note">
          <span>01</span>
          <p><strong>AI portrait transfer.</strong> InstantStyle keeps your face and pose from the photo, then paints the lighting, palette, and finish of the reference portrait on top with diffusion.</p>
        </div>
      </section>

      <section className="editor-workspace" aria-label="Portrait style transfer editor">
        <div className="upload-column">
          <PortraitDropzone eyebrow="01 · Source" title="Choose your portrait" description="One front-facing headshot with clear eyes and fine facial detail." file={inputFile} onFile={setInputFile} tone="warm" />
        </div>
        <div className="upload-column">
          <div className="reference-tabs" role="tablist" aria-label="Reference source">
            <button type="button" role="tab" aria-selected={referenceMode === "upload"} className={referenceMode === "upload" ? "is-active" : ""} onClick={() => setReferenceMode("upload")}>Upload reference</button>
            <button type="button" role="tab" aria-selected={referenceMode === "library"} className={referenceMode === "library" ? "is-active" : ""} onClick={() => setReferenceMode("library")}>Style library</button>
          </div>
          {referenceMode === "upload" ? (
            <PortraitDropzone eyebrow="02 · Style" title="Choose a reference" description="Use a headshot with a similar pose, crop, and expression." file={referenceFile} onFile={setReferenceFile} tone="cool" />
          ) : (
            <section className="upload-card upload-card--cool style-picker" aria-labelledby="style-picker-title">
              <div className="upload-card__topline"><span>02 · Style</span><span className="file-rule">PRIVATE LIBRARY</span></div>
              <h2 id="style-picker-title">Choose a collection</h2>
              <p>The engine automatically ranks each collection’s examples for your portrait.</p>
              <div className="style-options">
                {(styles.data ?? []).map((style: StyleRecord) => (
                  <button key={style.id} type="button" onClick={() => setSelectedStyle(style.id)} className={selectedStyle === style.id ? "is-selected" : ""}>
                    <span className="style-swatch">{style.preview_url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={style.preview_url} alt="" />
                    ) : <Sparkles size={18} aria-hidden="true" />}</span>
                    <span><strong>{style.name}</strong><small>{style.example_count ?? 0} examples</small></span>
                  </button>
                ))}
                {!styles.isLoading && (styles.data?.length ?? 0) === 0 && <p className="empty-copy">Your library is empty. Add a rights-cleared collection from Style library.</p>}
              </div>
            </section>
          )}
        </div>
        <SettingsPanel settings={settings} onChange={setSettings} />
      </section>

      <section className="action-dock">
        <div>
          {privacyLine}
          <span className="requirements-line"><ShieldCheck size={14} aria-hidden="true" /> One near-frontal face · minimum 150 px between eyes · max 15 MB / 8 megapixels</span>
        </div>
        <button className="button button--create" type="button" disabled={!canSubmit} onClick={submit}>
          {submitting ? "Securing uploads…" : "Create portrait"}<ArrowRight size={19} aria-hidden="true" />
        </button>
      </section>

      {error && <div className="global-error" role="alert"><CircleAlert size={18} aria-hidden="true" /><span>{error}</span><button type="button" onClick={() => setError(null)}>Dismiss</button></div>}
      {assets?.reference && <CompatibilityCard input={assets.input} reference={assets.reference} />}
      {job && <ProgressTimeline job={job} />}
      {job?.status === "SUCCEEDED" && inputPreview && (outputUrl ?? job.output_url) && (
        <div className="result-stack">
          <PortraitComparison job={job} inputUrl={inputPreview} outputUrl={(outputUrl ?? job.output_url)!} onDownload={download} onRetry={() => router.push(`/jobs/${job.id}?edit=1`)} onDelete={remove} />
          <DiagnosticsPanel job={job} />
        </div>
      )}

      <section className="method-strip" aria-label="How the portrait transfer works">
        <div><span>01</span><strong>Understand</strong><p>Encode source structure and reference appearance separately.</p></div>
        <div><span>02</span><strong>Correspond</strong><p>Match semantic portrait regions before transferring style.</p></div>
        <div><span>03</span><strong>Generate</strong><p>Use structure-guided diffusion to protect pose and facial geometry.</p></div>
        <div><span>04</span><strong>Finish</strong><p>Restore the requested background, format, and original composition.</p></div>
      </section>
    </>
  );
}
