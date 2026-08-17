"use client";

import { ChevronDown, Info, Sparkles } from "lucide-react";
import type { BackgroundMode, OutputFormat, TransferSettings } from "@/lib/api/types";

interface SettingsPanelProps {
  settings: TransferSettings;
  onChange: (settings: TransferSettings) => void;
}

function RangeControl({ label, value, min, max, step, hint, onChange }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  hint: string;
  onChange: (value: number) => void;
}) {
  const percent = ((value - min) / (max - min)) * 100;
  return (
    <label className="range-control">
      <span className="range-control__label">
        <span>{label} <span className="hint-dot" title={hint}><Info size={13} aria-hidden="true" /></span></span>
        <output>{value.toFixed(2)}</output>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        style={{ "--range-progress": `${percent}%` } as React.CSSProperties}
      />
    </label>
  );
}

export function SettingsPanel({ settings, onChange }: SettingsPanelProps) {
  function patch(next: Partial<TransferSettings>) { onChange({ ...settings, ...next }); }

  return (
    <aside className="settings-card" aria-labelledby="settings-title">
      <div className="section-heading section-heading--compact">
        <div>
          <span className="eyebrow">03 · Direction</span>
          <h2 id="settings-title">Shape the finish</h2>
        </div>
        <span className="method-badge"><Sparkles size={14} aria-hidden="true" /> Source 2014</span>
      </div>

      <div className="settings-stack">
        <RangeControl label="Texture & contrast" value={settings.transfer_strength} min={0} max={1} step={0.05} hint="At 0 the original detail is kept; at 1 the paper-exact gain is applied." onChange={(value) => patch({ transfer_strength: value })} />
        <RangeControl label="Broad lighting" value={settings.residual_strength} min={0} max={1} step={0.05} hint="Controls how strongly the reference's low-frequency light and color are transferred." onChange={(value) => patch({ residual_strength: value })} />
        <RangeControl label="Global range" value={settings.global_range_mix} min={0} max={1} step={0.05} hint="Adds mask-aware histogram matching when local transfer alone is too restrained." onChange={(value) => patch({ global_range_mix: value })} />
      </div>

      <fieldset className="segmented-fieldset">
        <legend>Background</legend>
        <div className="segmented-control segmented-control--four">
          {(["KEEP", "BLUR", "SOLID", "REFERENCE"] as BackgroundMode[]).map((mode) => (
            <button key={mode} type="button" className={settings.background_mode === mode ? "is-active" : ""} onClick={() => patch({ background_mode: mode, background_color: mode === "SOLID" ? (settings.background_color ?? "#d7d2c8") : null })}>
              {{ KEEP: "Keep", BLUR: "Blur", SOLID: "Color", REFERENCE: "Ref." }[mode]}
            </button>
          ))}
        </div>
        {settings.background_mode === "SOLID" && (
          <label className="color-field">Background color <input type="color" value={settings.background_color ?? "#d7d2c8"} onChange={(event) => patch({ background_color: event.target.value })} /></label>
        )}
      </fieldset>

      <div className="toggle-row">
        <div><strong>Eye highlights</strong><span>Transfer catchlights when both eyes are reliable.</span></div>
        <label className="switch"><input type="checkbox" checked={settings.eye_highlights} onChange={(event) => patch({ eye_highlights: event.target.checked })} /><span /></label>
      </div>

      <fieldset className="segmented-fieldset">
        <legend>Output</legend>
        <div className="segmented-control">
          {(["PNG", "JPEG"] as OutputFormat[]).map((format) => (
            <button key={format} type="button" className={settings.output_format === format ? "is-active" : ""} onClick={() => patch({ output_format: format })}>{format}</button>
          ))}
        </div>
      </fieldset>

      <details className="advanced-settings">
        <summary>Advanced <ChevronDown size={16} aria-hidden="true" /></summary>
        <div>
          <div className="toggle-row">
            <div><strong>Dense alignment</strong><span>Slower, but improves contours and fine placement.</span></div>
            <label className="switch"><input type="checkbox" checked={settings.dense_alignment} onChange={(event) => patch({ dense_alignment: event.target.checked })} /><span /></label>
          </div>
          <div className="toggle-row">
            <div><strong>Save diagnostics</strong><span>Private artifacts expire with the job.</span></div>
            <label className="switch"><input type="checkbox" checked={settings.debug_artifacts} onChange={(event) => patch({ debug_artifacts: event.target.checked })} /><span /></label>
          </div>
        </div>
      </details>
    </aside>
  );
}
