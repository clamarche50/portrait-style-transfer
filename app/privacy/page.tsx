import type { Metadata } from "next";
import Link from "next/link";
import { Clock3, Database, EyeOff, ShieldCheck, Trash2 } from "lucide-react";
import { SiteHeader } from "@/components/SiteHeader";

export const metadata: Metadata = { title: "Privacy", description: "How Portrait Studio protects uploaded headshots and generated results." };

const principles = [
  { icon: EyeOff, title: "No training", copy: "Your portraits are used only to complete the transfer job you request. They are never added to a model-training corpus." },
  { icon: Clock3, title: "24-hour expiry", copy: "Inputs, references, outputs, and diagnostics expire automatically after 24 hours unless your deployment changes the retention policy." },
  { icon: Database, title: "Private objects", copy: "Image bytes stay in private object storage. Database records contain metadata, not portraits or persistent face embeddings." },
  { icon: Trash2, title: "Delete anytime", copy: "Deleting a job removes its input, reference, output, diagnostics, and related metadata. Failed deletions are retried and audited." },
];

export default function PrivacyPage() {
  return (
    <div className="site-shell"><SiteHeader /><main id="main-content" className="page-container privacy-page">
      <section className="privacy-hero"><span className="privacy-shield"><ShieldCheck size={30} aria-hidden="true" /></span><span className="eyebrow">Privacy by design</span><h1>Your face is not<br /><em>the product.</em></h1><p>Portrait Studio processes sensitive images without recognition, identity search, demographic labels, or permanent biometric profiles.</p></section>
      <section className="privacy-grid">{principles.map(({ icon: Icon, title, copy }) => <article key={title}><Icon size={21} aria-hidden="true" /><h2>{title}</h2><p>{copy}</p></article>)}</section>
      <section className="privacy-detail"><div><span className="eyebrow">What we inspect</span><h2>Image quality, never identity</h2></div><div><p>To protect result quality, the pipeline estimates face count, crop, pose, blur, exposure, mask confidence, occlusion, and compatibility with the reference. It does not infer name, age, gender, race, or identity.</p><p>Uploads accept JPEG, PNG, and WebP only. Metadata and filenames are removed, contents are decoder-validated, and download links expire quickly.</p><Link className="button button--primary" href="/">Return to editor</Link></div></section>
    </main></div>
  );
}
