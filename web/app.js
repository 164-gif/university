(() => {
  const SUBJECT_NAMES = {
    physics: "Физика",
    analytical_geometry: "Аналитическая геометрия",
    obzh: "ОБЖ / БЖД",
    introduction_to_information_security: "Введение в информационную безопасность",
    information_security_fundamentals: "Основы информационной безопасности",
  };
  const SECTION_NAMES = {
    notes: "Конспекты",
    summaries: "Кратко",
    concepts: "Термины",
    exam: "Подготовка и тесты",
    checks: "Проверка информации",
    raw: "Исходники",
    transcripts: "Расшифровки",
    laboratory_physics: "Лабораторные",
    root: "Общее",
  };
  const SECTION_ORDER = ["notes", "summaries", "concepts", "exam", "laboratory_physics", "checks", "transcripts", "raw", "root"];
  const MEDIA_IMAGE = new Set(["svg", "png", "jpg", "jpeg", "webp", "gif", "avif"]);
  const MEDIA_VIDEO = new Set(["mp4", "webm"]);
  const MEDIA_AUDIO = new Set(["mp3", "wav", "ogg", "m4a"]);

  const el = (id) => document.getElementById(id);
  const contentEl = el("content");
  const navEl = el("navigation");
  const tocEl = el("toc");
  const crumbsEl = el("breadcrumbs");
  const searchDialog = el("searchDialog");
  const searchInput = el("searchInput");
  const searchResults = el("searchResults");
  let manifest = null;
  let docs = [];
  let docsByPath = new Map();
  let assetSet = new Set();
  let basenameIndex = new Map();
  let titleIndex = new Map();
  let searchSelection = 0;

  marked.setOptions({ gfm: true, breaks: true });

  function subjectName(key) {
    if (SUBJECT_NAMES[key]) return SUBJECT_NAMES[key];
    return key.replaceAll("_", " ").replace(/\b\w/g, (m) => m.toUpperCase());
  }

  function sectionName(key) { return SECTION_NAMES[key] || key.replaceAll("_", " "); }
  function stem(path) { return path.split("/").pop().replace(/\.(md|txt)$/i, ""); }
  function dirname(path) { return path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : ""; }
  function extname(path) { const m = path.toLowerCase().match(/\.([a-z0-9]+)$/); return m ? m[1] : ""; }
  function normalizePath(path) {
    const out = [];
    for (const part of path.replaceAll("\\", "/").split("/")) {
      if (!part || part === ".") continue;
      if (part === "..") out.pop(); else out.push(part);
    }
    return out.join("/");
  }
  function relativePath(currentPath, target) {
    if (target.startsWith("subjects/") || target === "README.md") return normalizePath(target);
    return normalizePath(`${dirname(currentPath)}/${target}`);
  }
  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  }
  function escapeAttr(value) { return escapeHtml(value); }
  function slugify(value) {
    return value.toLowerCase().trim().replace(/<[^>]+>/g, "").replace(/[^\p{L}\p{N}\s-]/gu, "").replace(/\s+/g, "-").replace(/-+/g, "-") || "section";
  }
  function docHref(path, heading = "") {
    return `#/doc/${encodeURIComponent(path)}${heading ? `?h=${encodeURIComponent(heading)}` : ""}`;
  }

  function buildIndexes() {
    docsByPath = new Map(docs.map((d) => [normalizePath(d.path), d]));
    assetSet = new Set(manifest.assets.map(normalizePath));
    basenameIndex = new Map();
    titleIndex = new Map();
    for (const doc of docs) {
      const keys = [stem(doc.path).toLowerCase(), doc.title.toLowerCase()];
      for (const key of keys) {
        if (!basenameIndex.has(key)) basenameIndex.set(key, []);
        basenameIndex.get(key).push(doc.path);
      }
      titleIndex.set(doc.title.toLowerCase(), doc.path);
    }
  }

  function resolveDoc(target, currentPath) {
    let raw = target.trim();
    let heading = "";
    const hash = raw.indexOf("#");
    if (hash >= 0) { heading = raw.slice(hash + 1); raw = raw.slice(0, hash); }
    raw = raw.replace(/\.(md|txt)$/i, "");
    const candidates = [];
    const withExt = (p) => [p, `${p}.md`, `${p}.txt`];
    if (raw.startsWith("./") || raw.startsWith("../") || raw.includes("/")) {
      for (const p of withExt(relativePath(currentPath, raw))) candidates.push(p);
      for (const p of withExt(normalizePath(raw))) candidates.push(p);
      const subjectRoot = currentPath.split("/").slice(0, 2).join("/");
      for (const p of withExt(normalizePath(`${subjectRoot}/${raw}`))) candidates.push(p);
    }
    for (const p of candidates) if (docsByPath.has(p)) return { path: p, heading };
    const byBase = basenameIndex.get(raw.toLowerCase());
    if (byBase?.length) return { path: byBase[0], heading };
    const byTitle = titleIndex.get(raw.toLowerCase());
    if (byTitle) return { path: byTitle, heading };
    return null;
  }

  function resolveAsset(target, currentPath) {
    const clean = target.split("#")[0].trim();
    const candidates = [relativePath(currentPath, clean), normalizePath(clean)];
    const subjectRoot = currentPath.split("/").slice(0, 2).join("/");
    candidates.push(normalizePath(`${subjectRoot}/${clean}`));
    for (const p of candidates) if (assetSet.has(p)) return p;
    const wanted = clean.split("/").pop().toLowerCase();
    const found = manifest.assets.find((p) => p.split("/").pop().toLowerCase() === wanted);
    return found || candidates[0];
  }

  function embedHtml(spec, currentPath) {
    const pieces = spec.split("|").map((s) => s.trim());
    const target = pieces[0];
    const width = pieces.find((p) => /^\d{2,4}$/.test(p));
    const extension = extname(target);
    if (!extension || extension === "md" || extension === "txt") {
      const doc = resolveDoc(target, currentPath);
      if (!doc) return `<span class="broken-link">[[${escapeHtml(spec)}]]</span>`;
      return `<a class="asset-link" href="${docHref(doc.path, doc.heading)}">↗ ${escapeHtml(stem(doc.path))}</a>`;
    }
    const path = resolveAsset(target, currentPath);
    const url = `content/${encodeURI(path).replaceAll("#", "%23")}`;
    const style = width ? ` style="max-width:${Number(width)}px"` : "";
    if (MEDIA_IMAGE.has(extension)) return `<img src="${escapeAttr(url)}" alt="${escapeAttr(stem(target))}"${style}>`;
    if (MEDIA_VIDEO.has(extension)) return `<video controls src="${escapeAttr(url)}"></video>`;
    if (MEDIA_AUDIO.has(extension)) return `<audio controls src="${escapeAttr(url)}"></audio>`;
    if (extension === "pdf") return `<iframe src="${escapeAttr(url)}" title="${escapeAttr(stem(target))}"></iframe>`;
    return `<a class="asset-link" href="${escapeAttr(url)}" target="_blank" rel="noopener">Открыть файл: ${escapeHtml(target)}</a>`;
  }

  function preprocessMarkdown(raw, currentPath) {
    let text = raw.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, "");
    const math = [];
    text = text.replace(/\$\$[\s\S]*?\$\$|\$(?:\\.|[^$\n])+\$/g, (m) => {
      const token = `@@UNIVMATH${math.length}@@`;
      math.push(m);
      return token;
    });
    text = text.replace(/!\[\[([^\]]+)\]\]/g, (_, spec) => embedHtml(spec, currentPath));
    text = text.replace(/\[\[([^\]]+)\]\]/g, (_, spec) => {
      const split = spec.split("|");
      const target = split[0].trim();
      const label = (split.slice(1).join("|") || target.split("#")[0]).trim();
      const doc = resolveDoc(target, currentPath);
      return doc ? `[${label}](${docHref(doc.path, doc.heading)})` : `<span class="broken-link">${escapeHtml(label)}</span>`;
    });
    let html = marked.parse(text);
    html = html.replace(/@@UNIVMATH(\d+)@@/g, (_, n) => escapeHtml(math[Number(n)] || ""));
    return html;
  }

  function upgradeCallouts(root) {
    for (const quote of [...root.querySelectorAll("blockquote")]) {
      const first = quote.querySelector("p");
      if (!first) continue;
      const match = first.innerHTML.match(/^\[!([a-zA-Z-]+)\](?:\s*([^<]*))?/);
      if (!match) continue;
      const type = match[1].toLowerCase();
      const title = (match[2] || type).trim();
      first.innerHTML = first.innerHTML.replace(match[0], "").trim();
      const box = document.createElement("div");
      box.className = `callout callout-${type}`;
      const head = document.createElement("div");
      head.className = "callout-title";
      head.textContent = title || type;
      box.appendChild(head);
      while (quote.firstChild) box.appendChild(quote.firstChild);
      quote.replaceWith(box);
    }
  }

  function upgradeLinksAndMedia(root, currentPath) {
    for (const img of root.querySelectorAll("img")) {
      const src = img.getAttribute("src") || "";
      if (/^(https?:|data:|content\/)/i.test(src)) continue;
      img.src = `content/${encodeURI(resolveAsset(src, currentPath))}`;
    }
    for (const a of root.querySelectorAll("a")) {
      const href = a.getAttribute("href") || "";
      if (!href || href.startsWith("#/") || href.startsWith("#")) continue;
      if (/^(https?:|mailto:|tel:)/i.test(href)) { a.target = "_blank"; a.rel = "noopener noreferrer"; continue; }
      const clean = href.split("#")[0];
      const heading = href.includes("#") ? href.split("#").slice(1).join("#") : "";
      const resolved = resolveDoc(clean, currentPath);
      if (resolved) a.href = docHref(resolved.path, heading || resolved.heading);
      else {
        const asset = resolveAsset(clean, currentPath);
        if (assetSet.has(asset)) a.href = `content/${encodeURI(asset)}`;
      }
    }
  }

  function addHeadingIds(root) {
    const used = new Map();
    for (const h of root.querySelectorAll("h1,h2,h3,h4")) {
      let base = slugify(h.textContent);
      const count = used.get(base) || 0;
      used.set(base, count + 1);
      h.id = count ? `${base}-${count + 1}` : base;
    }
  }

  function buildToc() {
    const heads = [...contentEl.querySelectorAll("h2,h3")];
    if (!heads.length) { tocEl.innerHTML = ""; return; }
    tocEl.innerHTML = `<div class="toc-title">На странице</div>` + heads.map((h) => `<a data-level="${h.tagName.slice(1)}" href="#${encodeURIComponent(h.id)}">${escapeHtml(h.textContent)}</a>`).join("");
    tocEl.querySelectorAll("a").forEach((a) => a.addEventListener("click", (e) => {
      e.preventDefault(); document.getElementById(decodeURIComponent(a.hash.slice(1)))?.scrollIntoView({behavior:"smooth"});
    }));
  }

  async function renderEnhancements(currentPath, requestedHeading) {
    upgradeCallouts(contentEl);
    upgradeLinksAndMedia(contentEl, currentPath);
    addHeadingIds(contentEl);
    for (const code of [...contentEl.querySelectorAll("pre > code.language-mermaid")]) {
      const pre = code.parentElement;
      const div = document.createElement("div");
      div.className = "mermaid";
      div.textContent = code.textContent;
      pre.replaceWith(div);
    }
    if (window.renderMathInElement) {
      renderMathInElement(contentEl, { delimiters: [
        {left:"$$", right:"$$", display:true}, {left:"$", right:"$", display:false}
      ], throwOnError:false, strict:false });
    }
    if (window.__mermaid && contentEl.querySelector(".mermaid")) {
      const theme = document.documentElement.dataset.theme === "dark" ? "dark" : "default";
      window.__mermaid.initialize({ startOnLoad:false, securityLevel:"loose", theme });
      try { await window.__mermaid.run({ nodes: contentEl.querySelectorAll(".mermaid") }); } catch (e) { console.warn(e); }
    }
    buildToc();
    if (requestedHeading) {
      requestAnimationFrame(() => {
        const wanted = slugify(requestedHeading);
        const target = document.getElementById(wanted) || [...contentEl.querySelectorAll("h1,h2,h3,h4")].find((h) => h.textContent.trim().toLowerCase() === requestedHeading.trim().toLowerCase());
        target?.scrollIntoView();
      });
    } else window.scrollTo({top:0});
  }

  function currentRoute() {
    const hash = location.hash || "#/";
    if (!hash.startsWith("#/doc/")) return { type:"home" };
    const rest = hash.slice(6);
    const [encoded, query = ""] = rest.split("?", 2);
    const params = new URLSearchParams(query);
    try { return { type:"doc", path:decodeURIComponent(encoded), heading:params.get("h") || "" }; }
    catch { return { type:"home" }; }
  }

  function renderBreadcrumbs(doc) {
    if (!doc) { crumbsEl.innerHTML = ""; return; }
    crumbsEl.innerHTML = `<a href="#/">Главная</a> <span>›</span> <span>${escapeHtml(subjectName(doc.subject))}</span> <span>›</span> <span>${escapeHtml(sectionName(doc.section))}</span>`;
  }

  function subjectEntry(subjectDocs) {
    return subjectDocs.find((d) => d.section === "notes" && /^00\s*-/.test(stem(d.path))) || subjectDocs.find((d) => /README\.md$/i.test(d.path)) || subjectDocs.find((d) => d.section === "notes") || subjectDocs[0];
  }

  function renderHome() {
    renderBreadcrumbs(null); tocEl.innerHTML = "";
    const groups = Object.groupBy ? Object.groupBy(docs.filter((d) => d.subject !== "repository"), (d) => d.subject) : docs.filter((d) => d.subject !== "repository").reduce((a,d)=>((a[d.subject]??=[]).push(d),a),{});
    const subjects = Object.entries(groups).sort((a,b) => subjectName(a[0]).localeCompare(subjectName(b[0]), "ru"));
    contentEl.innerHTML = `<div class="home-hero"><h1>University Notes</h1><div class="home-subtitle">Все конспекты из репозитория в браузере: формулы LaTeX, Obsidian-ссылки, схемы, таблицы и исходные материалы.</div><div class="stats"><div class="stat">${manifest.stats.documents} документов</div><div class="stat">${subjects.length} предметов</div><div class="stat">светлая / тёмная тема</div></div></div><div class="subject-grid">${subjects.map(([key, list]) => { const entry = subjectEntry(list); return `<a class="subject-card" href="${docHref(entry.path)}"><div class="subject-card-title">${escapeHtml(subjectName(key))}</div><div class="subject-card-meta">${list.length} материалов · ${new Set(list.map(d=>d.section)).size} разделов</div></a>`; }).join("")}</div>`;
    updateActiveNav();
  }

  async function renderDoc(path, heading = "") {
    const doc = docsByPath.get(normalizePath(path));
    if (!doc) {
      renderBreadcrumbs(null); tocEl.innerHTML = "";
      contentEl.innerHTML = `<div class="error-box"><strong>Материал не найден.</strong><br><a href="#/">Вернуться на главную</a></div>`;
      return;
    }
    renderBreadcrumbs(doc);
    contentEl.innerHTML = preprocessMarkdown(doc.content, doc.path);
    if (!contentEl.querySelector("h1")) contentEl.insertAdjacentHTML("afterbegin", `<h1>${escapeHtml(doc.title)} <span class="file-badge">${escapeHtml(doc.ext)}</span></h1>`);
    await renderEnhancements(doc.path, heading);
    updateActiveNav();
    closeSidebar();
  }

  function renderNavigation() {
    const subjectDocs = docs.filter((d) => d.subject !== "repository");
    const groups = subjectDocs.reduce((acc, d) => ((acc[d.subject] ??= []).push(d), acc), {});
    navEl.innerHTML = Object.keys(groups).sort((a,b)=>subjectName(a).localeCompare(subjectName(b),"ru")).map((subject) => {
      const sections = groups[subject].reduce((acc,d)=>((acc[d.section]??=[]).push(d),acc),{});
      const ordered = Object.keys(sections).sort((a,b) => {
        const ai=SECTION_ORDER.indexOf(a), bi=SECTION_ORDER.indexOf(b);
        return (ai<0?99:ai)-(bi<0?99:bi) || a.localeCompare(b);
      });
      return `<details data-subject="${escapeAttr(subject)}"><summary>${escapeHtml(subjectName(subject))}</summary>${ordered.map((section)=>`<div class="nav-section"><div class="nav-section-title">${escapeHtml(sectionName(section))}</div>${sections[section].sort((a,b)=>a.title.localeCompare(b.title,"ru",{numeric:true})).map((d)=>`<a class="nav-link" data-path="${escapeAttr(d.path)}" href="${docHref(d.path)}">${escapeHtml(d.title)}</a>`).join("")}</div>`).join("")}</details>`;
    }).join("");
  }

  function updateActiveNav() {
    const route = currentRoute();
    navEl.querySelectorAll(".nav-link").forEach((a) => a.classList.toggle("active", route.type === "doc" && a.dataset.path === route.path));
    if (route.type === "doc") {
      const doc = docsByPath.get(route.path);
      if (doc) {
        const details = navEl.querySelector(`details[data-subject="${CSS.escape(doc.subject)}"]`);
        if (details) details.open = true;
      }
    }
  }

  function openSearch() {
    if (!searchDialog.open) searchDialog.showModal();
    searchInput.value = ""; searchSelection = 0; renderSearch("");
    setTimeout(() => searchInput.focus(), 0);
  }
  function searchDocs(query) {
    const q = query.trim().toLowerCase();
    if (!q) return docs.filter((d)=>d.subject!=="repository").slice(0,18);
    const terms = q.split(/\s+/).filter(Boolean);
    return docs.map((d) => {
      const hayTitle = `${d.title} ${subjectName(d.subject)} ${sectionName(d.section)}`.toLowerCase();
      const hay = `${hayTitle}\n${d.content}`.toLowerCase();
      if (!terms.every((t)=>hay.includes(t))) return null;
      let score = terms.reduce((s,t)=>s+(hayTitle.includes(t)?20:1),0);
      if (d.title.toLowerCase().startsWith(q)) score += 30;
      return {d,score};
    }).filter(Boolean).sort((a,b)=>b.score-a.score).slice(0,40).map((x)=>x.d);
  }
  function renderSearch(query) {
    const found = searchDocs(query);
    searchSelection = Math.min(searchSelection, Math.max(0, found.length - 1));
    searchResults.innerHTML = found.length ? found.map((d,i)=>`<a class="search-item ${i===searchSelection?"selected":""}" data-index="${i}" href="${docHref(d.path)}"><div class="search-item-title">${escapeHtml(d.title)}</div><div class="search-item-path">${escapeHtml(subjectName(d.subject))} · ${escapeHtml(sectionName(d.section))}</div></a>`).join("") : `<div class="empty">Ничего не найдено</div>`;
    searchResults.querySelectorAll("a").forEach((a)=>a.addEventListener("click",()=>searchDialog.close()));
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("university-theme", theme);
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "dark" ? "#0f1117" : "#f7f8fb");
  }
  function initTheme() {
    const saved = localStorage.getItem("university-theme");
    applyTheme(saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  }
  function toggleTheme() { applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"); const r=currentRoute(); if(r.type==="doc" && contentEl.querySelector(".mermaid")) renderDoc(r.path,r.heading); }
  function openSidebar() { el("sidebar").classList.add("open"); el("overlay").classList.add("open"); }
  function closeSidebar() { el("sidebar").classList.remove("open"); el("overlay").classList.remove("open"); }

  async function route() {
    const r = currentRoute();
    if (r.type === "doc") await renderDoc(r.path, r.heading); else renderHome();
  }

  async function init() {
    initTheme();
    contentEl.innerHTML = `<div class="empty">Загрузка конспектов…</div>`;
    try {
      manifest = await fetch("content.json", {cache:"no-cache"}).then((r) => { if(!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); });
      docs = manifest.docs;
      buildIndexes(); renderNavigation(); await route();
    } catch (error) {
      contentEl.innerHTML = `<div class="error-box"><strong>Не удалось загрузить материалы.</strong><br>${escapeHtml(error.message)}</div>`;
    }
  }

  addEventListener("hashchange", route);
  el("themeBtn").addEventListener("click", toggleTheme);
  el("searchBtn").addEventListener("click", openSearch);
  el("menuBtn").addEventListener("click", openSidebar);
  el("closeSidebar").addEventListener("click", closeSidebar);
  el("overlay").addEventListener("click", closeSidebar);
  searchInput.addEventListener("input", () => { searchSelection=0; renderSearch(searchInput.value); });
  searchInput.addEventListener("keydown", (e) => {
    const items=[...searchResults.querySelectorAll(".search-item")];
    if(e.key==="ArrowDown"||e.key==="ArrowUp") { e.preventDefault(); searchSelection=(searchSelection+(e.key==="ArrowDown"?1:-1)+Math.max(items.length,1))%Math.max(items.length,1); renderSearch(searchInput.value); searchResults.querySelector(".selected")?.scrollIntoView({block:"nearest"}); }
    if(e.key==="Enter" && items.length) { e.preventDefault(); const selected=searchResults.querySelector(".selected"); searchDialog.close(); if(selected) location.hash=selected.getAttribute("href"); }
  });
  addEventListener("keydown", (e) => {
    if((e.key==="/" && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) || ((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="k")) { e.preventDefault(); openSearch(); }
  });
  init();
})();
