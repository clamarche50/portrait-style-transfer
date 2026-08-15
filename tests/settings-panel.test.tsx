import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { SettingsPanel } from "@/components/editor/SettingsPanel";
import { defaultSettings } from "@/lib/validation/portrait";

describe("AI transfer settings panel", () => {
  it("shows the InstantStyle controls and removes classical-engine controls", () => {
    const onChange = vi.fn();
    render(<SettingsPanel settings={defaultSettings} onChange={onChange} />);

    expect(screen.getByText("InstantStyle · AI")).toBeTruthy();
    expect(screen.getByRole("slider", { name: /Style strength/ })).toBeTruthy();
    expect(screen.getByRole("slider", { name: /Identity & structure/ })).toBeTruthy();
    expect(screen.getByRole("slider", { name: /Inference steps/ })).toBeTruthy();
    expect(screen.queryByText("Dense alignment")).toBeNull();
    expect(screen.queryByText("Eye highlights")).toBeNull();

    fireEvent.change(screen.getByRole("slider", { name: /Style strength/ }), {
      target: { value: "0.5" },
    });
    expect(onChange).toHaveBeenLastCalledWith({ ...defaultSettings, style_strength: 0.5 });
  });

  it("exposes deterministic seeds and JPEG quality", () => {
    const onChange = vi.fn();
    render(
      <SettingsPanel
        settings={{ ...defaultSettings, output_format: "JPEG" }}
        onChange={onChange}
      />,
    );

    expect(screen.getByRole("spinbutton", { name: /Random seed/ })).toBeTruthy();
    expect(screen.getByRole("slider", { name: /JPEG quality/ })).toBeTruthy();
  });
});
