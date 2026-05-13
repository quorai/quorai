const GROUP_COLORS = {
  deep_value: "var(--c-deep-value)",
  quality_compounders: "var(--c-quality-compounders)",
  growth_and_catalyst: "var(--c-growth-and-catalyst)",
  macro_and_cycle: "var(--c-macro-and-cycle)",
  quant_systematic: "var(--c-quant-systematic)",
  sentiment_and_analytical: "var(--c-sentiment-analytical)",
};

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function buildCard(agent, groupColor, groupDisplayName) {
  const quote = agent.pull_quote
    ? `<p class="card-quote">${escapeHtml(agent.pull_quote)}</p>`
    : "";
  return `
    <article class="agent-card" style="--group-color:${groupColor}">
      <div class="card-top">
        <div class="card-avatar">${escapeHtml(agent.initials)}</div>
        <div class="card-identity">
          <div class="card-name">${escapeHtml(agent.display_name)}</div>
          <div class="card-tagline">${escapeHtml(agent.description)}</div>
        </div>
      </div>
      <span class="card-badge">${escapeHtml(groupDisplayName)}</span>
      <p class="card-style">${escapeHtml(agent.investing_style)}</p>
      ${quote}
    </article>`;
}

function buildGroupSection(group) {
  const color = GROUP_COLORS[group.key] || "#818cf8";
  const cards = group.agents.map((a) => buildCard(a, color, group.display_name)).join("");
  return `
    <section class="group-section" id="group-${escapeHtml(group.key)}">
      <div class="group-header" style="--group-color:${color}">
        <h2 class="group-title">${escapeHtml(group.display_name)}</h2>
        <span class="group-count">${group.agents.length} agent${group.agents.length !== 1 ? "s" : ""}</span>
      </div>
      <p class="group-desc">${escapeHtml(group.description)}</p>
      <div class="group-divider" style="--group-color:${color}"></div>
      <div class="card-grid">${cards}</div>
    </section>`;
}

function buildMermaidSpec(flowData) {
  const { stages, groups } = flowData;
  const lines = ["flowchart LR"];

  lines.push(`    input(["${stages.input.label}"])`);
  lines.push(`    preflight["${stages.preflight.label}\\n${stages.preflight.note}"]`);

  lines.push("    subgraph analysts[\" \"]");
  for (const g of groups) {
    lines.push(`        ${g.key}["${g.display_name}\\n(${g.count} agents)"]:::${g.key}`);
  }
  lines.push("    end");

  lines.push(`    debate["${stages.debate.label}\\n${stages.debate.note}"]`);
  lines.push(`    risk["${stages.risk.label}\\n${stages.risk.note}"]`);
  lines.push(`    pm["${stages.pm.label}\\n${stages.pm.note}"]`);
  lines.push(`    decision(["${stages.decision.label}"])`);

  const groupKeys = groups.map((g) => g.key).join(" & ");
  lines.push(`    input --> preflight`);
  lines.push(`    preflight --> ${groupKeys}`);
  lines.push(`    ${groupKeys} --> debate`);
  lines.push("    debate --> risk --> pm --> decision");

  for (const g of groups) {
    lines.push(`    classDef ${g.key} fill:${g.color},color:#fff,stroke:${g.color}`);
  }

  return lines.join("\n");
}

async function renderFlow(flowData) {
  const container = document.getElementById("flow-diagram");
  container.innerHTML = `<p class="loading">Loading diagram…</p>`;
  try {
    const mermaid = (
      await import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs")
    ).default;
    mermaid.initialize({ startOnLoad: false, theme: "neutral" });
    const spec = buildMermaidSpec(flowData);
    const pre = document.createElement("pre");
    pre.className = "mermaid";
    pre.textContent = spec;
    container.innerHTML = "";
    container.appendChild(pre);
    await mermaid.run({ nodes: [pre] });
  } catch (err) {
    container.innerHTML = `<p class="loading">Failed to load diagram: ${escapeHtml(String(err))}</p>`;
  }
}

function setupTabs(flowData) {
  const tabs = document.querySelectorAll(".tab");
  const views = document.querySelectorAll(".view");
  let flowRendered = false;

  tabs.forEach((tab) => {
    tab.addEventListener("click", async () => {
      const target = tab.dataset.tab;

      tabs.forEach((t) => {
        t.classList.toggle("tab--active", t === tab);
        t.setAttribute("aria-selected", t === tab ? "true" : "false");
      });
      views.forEach((v) => {
        v.classList.toggle("view--active", v.id === target);
      });

      if (target === "view-flow" && !flowRendered) {
        flowRendered = true;
        if (flowData) {
          await renderFlow(flowData);
        } else {
          document.getElementById("flow-diagram").innerHTML =
            `<p class="loading">Flow data unavailable.</p>`;
        }
      }
    });
  });
}

async function init() {
  const gallery = document.getElementById("gallery");
  try {
    const [agentsResp, flowResp] = await Promise.all([
      fetch("agents.json"),
      fetch("flow.json"),
    ]);
    if (!agentsResp.ok) throw new Error(`HTTP ${agentsResp.status}`);
    const agentsData = await agentsResp.json();
    gallery.innerHTML = agentsData.groups.map(buildGroupSection).join("");

    const flowData = flowResp.ok ? await flowResp.json() : null;
    setupTabs(flowData);
  } catch (err) {
    gallery.innerHTML = `<p class="loading">Failed to load agents: ${escapeHtml(String(err))}</p>`;
  }
}

init();
