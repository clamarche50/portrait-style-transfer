import Link from "next/link";

export function SiteHeader() {
  return (
    <header className="site-header">
      <Link className="brand" href="/" aria-label="Portrait Studio home">
        <span className="brand-mark" aria-hidden="true"><span /></span>
        <span>Portrait Studio</span>
      </Link>
      <nav aria-label="Primary navigation">
        <Link href="/styles">Style library</Link>
        <Link href="/privacy">Privacy</Link>
        <a className="research-link" href="https://people.csail.mit.edu/yichangshih/portrait_web/" target="_blank" rel="noreferrer">
          Research method <span aria-hidden="true">↗</span>
        </a>
      </nav>
    </header>
  );
}
