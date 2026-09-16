"""Render the council run report as a single static HTML file."""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "report" / "results.json").read_text())

FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' fill='%230A1522'/%3E"
    "%3Cpath d='M6 22 L16 8 L26 22' fill='none' stroke='%23E9A13B' stroke-width='2.5'/%3E"
    "%3Ccircle cx='16' cy='26' r='2' fill='%2356B8CE'/%3E%3C/svg%3E"
)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def chain_html(nodes: list[str]) -> str:
    if not nodes:
        return '<span class="chain-empty">path unconfirmed by the graph</span>'
    parts = "".join(
        f'<span class="node">{esc(n)}</span>'
        + ('<span class="arrow" aria-hidden="true">&rarr;</span>' if i < len(nodes) - 1 else "")
        for i, n in enumerate(nodes)
    )
    return f'<div class="chain">{parts}</div>'


def tape_html(runs: list[dict]) -> str:
    if not runs:
        return '<div class="tape"><span class="tape-none">no sandbox run</span></div>'
    cells = []
    for r in runs:
        ok = r["success"]
        cls = "cell pass" if ok else "cell fail"
        mark = "exit 0" if ok else f"exit {r['exit_code']}"
        note = "ran clean" if ok else ("crash" if r["exit_code"] not in (0, -1) else "no proof")
        cells.append(
            f'<div class="{cls}"><span class="try">try {r["attempt"]}</span>'
            f'<span class="rc">{esc(mark)}</span><span class="note">{esc(note)}</span></div>'
        )
    return f'<div class="tape">{"".join(cells)}</div>'


def finding_html(f: dict, index: int) -> str:
    proved = f["status"] == "exploited"
    badge = (
        '<span class="badge proved">proved in sandbox</span>'
        if proved
        else f'<span class="badge">{esc(f["status"])}</span>'
    )
    runs = f.get("poc_runs") or []
    last = runs[-1] if runs else None
    script = html.escape(last["script"]) if last and last.get("script") else ""
    output = ""
    if last:
        combined = (last.get("stdout") or "") + (last.get("stderr") or "")
        if combined.strip():
            output = (
                '<details class="console"><summary>sandbox output</summary>'
                f"<pre>{html.escape(combined[:2600])}</pre></details>"
            )
    script_block = (
        f'<details class="script"><summary>proof script ({len(script.splitlines())} lines)</summary>'
        f"<pre>{script}</pre></details>"
        if script
        else ""
    )
    return f"""
      <article class="finding {'proved' if proved else ''}">
        <div class="finding-head">
          <span class="file">{esc(f['file'])}<span class="line">:{esc(f['line'])}</span></span>
          {badge}
        </div>
        <div class="sink">sink <b>{esc(f['sink'])}</b></div>
        {chain_html(f['call_path'])}
        <p class="why">{esc(f.get('detail') or '')}</p>
        {tape_html(runs)}
        {script_block}
        {output}
      </article>"""


def run_html(run: dict) -> str:
    lang = "Python" if run["name"] == "pyservice" else "C"
    counts = run.get("status_counts", {})
    proved = counts.get("exploited", 0)
    total = len(run["findings"])
    findings = "".join(finding_html(f, i) for i, f in enumerate(run["findings"]))
    return f"""
    <section class="codebase reveal">
      <div class="cb-head">
        <span class="lang">{esc(lang)}</span>
        <h2>{esc(run['name'])}</h2>
      </div>
      <p class="cb-sub">{esc(run['label'])} &middot; {proved} of {total} proved &middot;
        {esc(run['llm_calls'])} model calls &middot; {esc(run['tool_calls'])} tool calls &middot;
        {esc(f'{run["global_tokens"]:,}')} tokens</p>
      {findings}
    </section>"""


def main() -> int:
    totals = DATA["totals"]
    proved = totals["exploited"]
    candidates = totals["candidates"]
    attempts = totals["poc_attempts"]
    sections = "".join(run_html(r) for r in DATA["runs"])

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Council audit report</title>
<link rel="icon" href="{FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {{
  --ink: #0A1522;
  --ink-2: #101F2E;
  --ink-3: #16283A;
  --line: #24384A;
  --paper: #DCE6EE;
  --muted: #7E93A5;
  --amber: #E9A13B;
  --cyan: #56B8CE;
  --rose: #D6655F;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  --sans: "IBM Plex Sans", system-ui, -apple-system, Segoe UI, sans-serif;
  --display: "Space Grotesk", var(--sans);
}}
* {{ box-sizing: border-box; }}
html {{ -webkit-text-size-adjust: 100%; }}
body {{
  margin: 0;
  background: var(--ink);
  color: var(--paper);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.55;
  background-image:
    radial-gradient(1100px 520px at 82% -8%, rgba(86,184,206,.10), transparent 62%),
    linear-gradient(180deg, #0A1522 0%, #0A1522 62%, #0C1A27 100%);
  background-attachment: fixed;
}}
.wrap {{ max-width: 680px; margin: 0 auto; padding: 0 18px 72px; }}
.eyebrow {{
  font-family: var(--mono); font-size: 12px; letter-spacing: .16em;
  text-transform: uppercase; color: var(--cyan);
  padding-top: 34px; display: flex; align-items: center; gap: 10px;
}}
.eyebrow::after {{ content: ""; flex: 1; height: 1px; background: var(--line); }}
h1 {{
  font-family: var(--display); font-weight: 700; letter-spacing: -.02em;
  font-size: clamp(34px, 9.4vw, 52px); line-height: 1.02; margin: 20px 0 0;
}}
h1 .big {{ color: var(--amber); display: block; font-size: clamp(84px, 26vw, 132px); line-height: .82; }}
.lede {{ color: var(--muted); font-size: 17px; margin: 18px 0 0; max-width: 46ch; }}
.funnel {{
  margin-top: 30px; display: grid; grid-template-columns: 1fr 1fr;
  border: 1px solid var(--line); background: var(--ink-2);
}}
.funnel div {{ padding: 14px 14px 15px; border-top: 1px solid var(--line); border-left: 1px solid var(--line); }}
.funnel div:nth-child(-n+2) {{ border-top: 0; }}
.funnel div:nth-child(odd) {{ border-left: 0; }}
.funnel b {{ display: block; font-family: var(--display); font-size: 30px; color: var(--paper); }}
.funnel span {{ font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); }}
.seal {{ margin: 14px 0 0; padding: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--line); border: 1px solid var(--line); }}
.seal > div {{ background: var(--ink-2); padding: 11px 13px; }}
.seal dt {{ font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); margin: 0 0 3px; }}
.seal dd {{ margin: 0; font-family: var(--mono); font-size: 13px; color: var(--cyan); }}
.codebase {{ margin-top: 54px; }}
.cb-head {{ display: flex; align-items: baseline; gap: 12px; border-bottom: 1px solid var(--line); padding-bottom: 10px; }}
.cb-head h2 {{ font-family: var(--display); font-weight: 500; font-size: 27px; margin: 0; letter-spacing: -.01em; }}
.lang {{ font-family: var(--mono); font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--ink); background: var(--cyan); padding: 3px 7px; }}
.cb-sub {{ font-family: var(--mono); font-size: 12px; color: var(--muted); margin: 10px 0 0; }}
.finding {{ border: 1px solid var(--line); background: var(--ink-2); margin-top: 16px; padding: 16px; border-left: 3px solid var(--amber); }}
.finding.proved {{ border-left-color: var(--amber); }}
.finding-head {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; }}
.file {{ font-family: var(--mono); font-size: 14px; color: var(--paper); }}
.file .line {{ color: var(--amber); }}
.badge {{ font-family: var(--mono); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; border: 1px solid var(--line); color: var(--muted); padding: 3px 8px; }}
.badge.proved {{ color: var(--ink); background: var(--amber); border-color: var(--amber); }}
.sink {{ font-family: var(--mono); font-size: 12px; color: var(--muted); margin-top: 10px; }}
.sink b {{ color: var(--paper); }}
.chain {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 12px; }}
.chain .node {{ font-family: var(--mono); font-size: 12.5px; color: var(--paper); background: var(--ink-3); border: 1px solid var(--line); padding: 4px 8px; }}
.chain .arrow {{ color: var(--muted); font-size: 13px; }}
.chain-empty {{ font-family: var(--mono); font-size: 12px; color: var(--muted); }}
.why {{ color: #B9C8D6; font-size: 15px; margin: 14px 0 0; }}
.tape {{ display: flex; gap: 1px; margin-top: 16px; background: var(--line); border: 1px solid var(--line); }}
.tape .cell {{ flex: 1; background: var(--ink-3); padding: 9px 11px; display: grid; gap: 2px; }}
.tape .cell.pass {{ background: rgba(233,161,59,.14); box-shadow: inset 0 -2px 0 var(--amber); }}
.tape .cell.fail {{ background: rgba(214,101,95,.10); box-shadow: inset 0 -2px 0 var(--rose); }}
.tape .try {{ font-family: var(--mono); font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }}
.tape .rc {{ font-family: var(--mono); font-size: 15px; color: var(--paper); }}
.tape .cell.pass .rc {{ color: var(--amber); }}
.tape .cell.fail .rc {{ color: var(--rose); }}
.tape .note {{ font-family: var(--mono); font-size: 11px; color: var(--muted); }}
.tape-none {{ font-family: var(--mono); font-size: 12px; color: var(--muted); padding: 10px; }}
details {{ margin-top: 12px; border: 1px solid var(--line); background: var(--ink); }}
details summary {{ cursor: pointer; padding: 11px 13px; font-family: var(--mono); font-size: 12.5px; color: var(--cyan); min-height: 44px; display: flex; align-items: center; }}
details summary:focus-visible {{ outline: 2px solid var(--cyan); outline-offset: -2px; }}
details pre {{ margin: 0; padding: 13px; overflow-x: auto; font-family: var(--mono); font-size: 12px; line-height: 1.5; color: #C6D4E0; border-top: 1px solid var(--line); }}
.method {{ margin-top: 60px; border-top: 1px solid var(--line); padding-top: 26px; }}
.method h2 {{ font-family: var(--display); font-weight: 500; font-size: 22px; margin: 0 0 6px; }}
.method ol {{ margin: 14px 0 0; padding-left: 20px; }}
.method li {{ margin-bottom: 9px; color: #B9C8D6; font-size: 15px; }}
.method li b {{ color: var(--paper); font-weight: 600; }}
.foot {{ margin-top: 34px; font-family: var(--mono); font-size: 11.5px; color: var(--muted); line-height: 1.7; }}
.reveal {{ opacity: 1; }}
@media (prefers-reduced-motion: no-preference) {{
  .reveal {{ opacity: 0; transform: translateY(14px); transition: opacity .6s ease, transform .6s ease; }}
  .reveal.in {{ opacity: 1; transform: none; }}
}}
@media (min-width: 560px) {{
  .seal {{ grid-template-columns: repeat(4, 1fr); }}
  .funnel {{ grid-template-columns: repeat(4, 1fr); }}
  .funnel div {{ border-top: 0; }}
  .funnel div:nth-child(odd) {{ border-left: 1px solid var(--line); }}
  .funnel div:first-child {{ border-left: 0; }}
}}
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">agent-forum / council</div>
  <h1><span class="big">{proved}</span> findings proved by running code</h1>
  <p class="lede">Four small codebases written to be audited. AI agents read each one, argued over
  every finding, then wrote a proof-of-concept and rewrote it until it actually ran inside an
  isolated Linux container.</p>

  <div class="funnel">
    <div><b>{candidates}</b><span>candidates</span></div>
    <div><b>{proved}</b><span>proved</span></div>
    <div><b>{attempts}</b><span>poc attempts</span></div>
    <div><b>{totals['llm_calls']}</b><span>model calls</span></div>
  </div>

  <dl class="seal">
    <div><dt>sandbox</dt><dd>docker</dd></div>
    <div><dt>network</dt><dd>none</dd></div>
    <div><dt>root fs</dt><dd>read-only</dd></div>
    <div><dt>caps</dt><dd>dropped</dd></div>
  </dl>

  {sections}

  <section class="method reveal">
    <h2>How a finding gets proved</h2>
    <ol>
      <li><b>Finder</b> walks a tree-sitter call graph, follows source to sink, and posts a candidate
      with the file, line, sink and call path.</li>
      <li><b>Killer</b> tries to break it with a bound check, a clamped size, or a dead path.</li>
      <li><b>Judge</b> keeps it only if the path survives, and only overrides the killer with new evidence.</li>
      <li><b>Exploit writer</b> writes a proof-of-concept, the sandbox runs it, and the failure output is
      fed back for a rewrite until it exits clean and prints the success marker.</li>
    </ol>
    <p class="cb-sub" style="margin-top:16px">Every proof-of-concept ran inside the agent's container:
    no network, read-only root filesystem, all capabilities dropped, no-new-privileges, 1 GB memory,
    256 processes, one writable mount. Nothing ran on the host.</p>
  </section>

  <p class="foot">
    {totals['tokens_global']:,} tokens across {totals['llm_calls']} model calls and
    {totals['tool_calls']} tool calls.<br>
    Model: deepseek-v4.1-flash. Static analysis: tree-sitter. Sandbox: docker on colima.<br>
    Reproduce: <code>python -m harness.cli council run --checkout corpus/&lt;name&gt;</code>
  </p>
</div>
<script>
(function () {{
  var items = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window) ||
      window.matchMedia("(prefers-reduced-motion: reduce)").matches) {{
    items.forEach(function (el) {{ el.classList.add("in"); }});
    return;
  }}
  var io = new IntersectionObserver(function (entries) {{
    entries.forEach(function (e) {{
      if (e.isIntersecting) {{ e.target.classList.add("in"); io.unobserve(e.target); }}
    }});
  }}, {{ rootMargin: "0px 0px -8% 0px", threshold: 0.05 }});
  items.forEach(function (el) {{ io.observe(el); }});
}})();
</script>
</body>
</html>
"""
    out = ROOT / "report" / "index.html"
    out.write_text(doc)
    print(f"wrote {out} ({len(doc)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
