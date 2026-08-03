import type { Metadata } from "next";
import { JobWorkspace } from "@/components/jobs/JobWorkspace";
import { SiteHeader } from "@/components/SiteHeader";

export const metadata: Metadata = { title: "Portrait job", robots: { index: false, follow: false } };

export default async function JobPage({ params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = await params;
  return <div className="site-shell"><SiteHeader /><JobWorkspace jobId={jobId} /></div>;
}
