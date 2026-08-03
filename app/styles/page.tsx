import type { Metadata } from "next";
import { SiteHeader } from "@/components/SiteHeader";
import { StyleLibrary } from "@/components/styles/StyleLibrary";

export const metadata: Metadata = { title: "Style library", description: "Manage private, rights-cleared portrait style collections." };

export default function StylesPage() {
  return <div className="site-shell"><SiteHeader /><StyleLibrary /></div>;
}
