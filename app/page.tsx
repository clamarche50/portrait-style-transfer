import { PortraitEditor } from "@/components/editor/PortraitEditor";
import { SiteHeader } from "@/components/SiteHeader";

export default function Home() {
  return (
    <div className="site-shell">
      <SiteHeader />
      <main id="main-content">
        <PortraitEditor />
      </main>
    </div>
  );
}
