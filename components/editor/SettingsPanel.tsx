"use client";

import { ChevronDown, Info, Sparkles } from "lucide-react";
import type { BackgroundMode, OutputFormat, TransferSettings } from "@/lib/api/types";

interface SettingsPanelProps {
  settings: TransferSettings;
  onChange: (settings: TransferSettings) => void;
}

function RangeControl({ label, value, min, max, step, hint, decimals = 2, onChange }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  hint: string;
  decimals?: number;
  onChange: (value: number) => void;
}) {
  const percent = ((value - min) / (max - min)) * 100;
  return (
    <label className="range-control">
      <span className="range-control__label">
        <span>{label} <span className="hint-dot" title={hint}><Info size={13} aria-hidden="true" /></span></span>
        <output>{value.toFixed(decimals)}</output>
      </span>
      <input
        type="range"
        aria-label={label}
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
        <span className="method-badge"><Sparkles size={14} aria-hidden="true" /> DGPST · AI</span>
      </div>

      <div className="settings-stack">
        <RangeControl
          label="Style strength"
          value={settings.style_strength}
          min={0}
          max={1}
          step={0.05}
          hint="Controls how strongly the reference portrait's color, lighting, and texture guide the result."
          onChange={(value) => patch({ style_strength: value })}
        />
        <RangeControl
          label="Identity & structure"
          value={settings.structure_strength}
          min={0}
          max={1}
          step={0.05}
          hint="Higher values more strongly preserve the source portrait's pose, expression, and facial geometry."
          onChange={(value) => patch({ structure_strength: value })}
        />
        <RangeControl
          label="Inference steps"
          value={settings.inference_steps}
          min={10}
          max={50}
          step={1}
          decimals={0}
          hint="More diffusion steps can refine detail but take longer and use the GPU for more time."
          onChange={(value) => patch({ inference_steps: value })}
        />
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
          {settings.output_format === "JPEG" && (
            <RangeControl
              label="JPEG quality"
              value={settings.jpeg_quality}
              min={70}
              max={100}
              step={1}
              decimals={0}
              hint="Higher quality preserves more detail but creates a larger file."
              onChange={(value) => patch({ jpeg_quality: value })}
            />
          )}
          <label className="number-field">
            <span><strong>Random seed</strong><small>Reuse a seed to reproduce the same AI variation.</small></span>
            <input
              type="number"
              min={0}
              max={2_147_483_647}
              step={1}
              value={settings.random_seed}
              onChange={(event) => patch({ random_seed: Number(event.target.value) })}
            />
          </label>
        </div>
      </details>
    </aside>
  );
}
