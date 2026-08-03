"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleAlert, ImagePlus, LibraryBig, Plus, ShieldCheck, Trash2, Upload } from "lucide-react";
import { FormEvent, useState } from "react";
import { addStyleExample, createStyle, deleteStyle, listStyles } from "@/lib/api/client";
import type { StyleRecord } from "@/lib/api/types";
import { validatePortraitFile } from "@/lib/validation/portrait";

function StyleCard({ style, onDelete, onAdd }: { style: StyleRecord; onDelete: () => void; onAdd: (file: File) => Promise<void> }) {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function add() {
    if (!file) return;
    const validationError = validatePortraitFile(file);
    if (validationError) { setError(validationError); return; }
    setBusy(true); setError(null);
    try { await onAdd(file); setFile(null); } catch (reason) { setError(reason instanceof Error ? reason.message : "Example upload failed."); }
    finally { setBusy(false); }
  }
  return (
    <article className="style-card">
      <div className="style-card__visual">
        {style.preview_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={style.preview_url} alt={`${style.name} collection preview`} />
        ) : <span><LibraryBig size={28} strokeWidth={1.5} aria-hidden="true" /></span>}
        <small>{style.example_count ?? 0} examples</small>
      </div>
      <div className="style-card__body"><h2>{style.name}</h2><p>{style.description || "Private portrait finish collection"}</p><span className="rights-badge"><ShieldCheck size={14} aria-hidden="true" /> Rights confirmed</span></div>
      <div className="style-card__actions">
        <label className="button button--ghost"><Upload size={15} aria-hidden="true" /> Choose example<input className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => setFile(event.target.files?.[0] ?? null)} /></label>
        {file && <button type="button" className="button button--primary" disabled={busy} onClick={add}>{busy ? "Indexing…" : `Add ${file.name}`}</button>}
        <button className="icon-button danger" type="button" aria-label={`Delete ${style.name}`} onClick={onDelete}><Trash2 size={16} aria-hidden="true" /></button>
      </div>
      {error && <p className="inline-error"><CircleAlert size={15} aria-hidden="true" />{error}</p>}
    </article>
  );
}

export function StyleLibrary() {
  const queryClient = useQueryClient();
  const styles = useQuery({ queryKey: ["styles"], queryFn: listStyles });
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [rights, setRights] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!name.trim() || !rights) return;
    setCreating(true); setError(null);
    try {
      await createStyle({ name: name.trim(), description: description.trim(), rights_confirmed: rights });
      setName(""); setDescription(""); setRights(false);
      await queryClient.invalidateQueries({ queryKey: ["styles"] });
    } catch (reason) { setError(reason instanceof Error ? reason.message : "The collection could not be created."); }
    finally { setCreating(false); }
  }

  async function remove(style: StyleRecord) {
    if (!window.confirm(`Delete “${style.name}” and all indexed examples?`)) return;
    await deleteStyle(style.id);
    await queryClient.invalidateQueries({ queryKey: ["styles"] });
  }

  async function add(style: StyleRecord, file: File) {
    await addStyleExample(style.id, file);
    await queryClient.invalidateQueries({ queryKey: ["styles"] });
  }

  return (
    <main id="main-content" className="page-container styles-page">
      <section className="library-hero"><span className="kicker"><span /> Private by default</span><h1>Build a language<br />of <em>light.</em></h1><p>Group rights-cleared headshots with a consistent finish. Portrait Studio indexes their multiscale energy, then chooses the closest example for each new face.</p></section>
      <section className="library-layout">
        <form className="create-style-card" onSubmit={submit}>
          <span className="eyebrow">New collection</span><h2>Add a visual direction</h2>
          <label>Name<input value={name} onChange={(event) => setName(event.target.value)} maxLength={80} placeholder="Soft north light" required /></label>
          <label>Description<textarea value={description} onChange={(event) => setDescription(event.target.value)} maxLength={500} placeholder="Muted chroma, open shadows, restrained texture." /></label>
          <label className="rights-check"><input type="checkbox" checked={rights} onChange={(event) => setRights(event.target.checked)} /><span><strong>I have permission to use these portraits</strong><small>Do not upload photographer collections without explicit rights.</small></span></label>
          <button className="button button--create" type="submit" disabled={!name.trim() || !rights || creating}><Plus size={18} aria-hidden="true" />{creating ? "Creating…" : "Create collection"}</button>
          {error && <p className="inline-error"><CircleAlert size={15} aria-hidden="true" />{error}</p>}
        </form>
        <div className="collection-list">
          <div className="collection-heading"><div><span className="eyebrow">Your library</span><h2>{styles.data?.length ?? 0} collections</h2></div><p>Each example is processed privately and gets its own quality and compatibility profile.</p></div>
          {styles.isLoading && <div className="loading-state"><span /> Loading collections…</div>}
          {styles.data?.map((style) => <StyleCard key={style.id} style={style} onDelete={() => remove(style)} onAdd={(file) => add(style, file)} />)}
          {!styles.isLoading && styles.data?.length === 0 && <div className="empty-library"><ImagePlus size={28} aria-hidden="true" /><h3>No collections yet</h3><p>Create a collection, then add at least one reference portrait.</p></div>}
        </div>
      </section>
    </main>
  );
}
