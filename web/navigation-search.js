(() => {
  const SECTION_META = {
    notes: { label: "Конспекты", icon: "▤" },
    summaries: { label: "Кратко", icon: "≡" },
    concepts: { label: "Термины", icon: "Aa" },
    exam: { label: "Подготовка и тесты", icon: "✓" },
    laboratory_physics: { label: "Лабораторные", icon: "⚗" },
    checks: { label: "Проверка информации", icon: "?" },
    transcripts: { label: "Расшифровки", icon: "◫" },
    raw: { label: "Исходники", icon: "{}" },
    root: { label: "Общее", icon: "•" },
  };

  const SUBJECT_NAMES = {
    physics: "Физика",
    analytical_geometry: "Аналитическая геометрия",
    obzh: "ОБЖ / БЖД",
    introduction_to_information_security: "Введение в информационную безопасность",
    information_security_fundamentals: "Основы информационной безопасности",
    russian_statehood_fundamentals: "Основы российской государственности",
  };

  const nav = document.getElementById("navigation");
  const searchDialog = document.getElementById("searchDialog");
  const searchInput = document.getElementById("searchInput");
  const searchResults = document.getElementById("searchResults");
  if (!nav || !searchDialog || !searchInput || !searchResults) return;

  let manifest = null;
  let searchIndex = [];
  let activeSection = "all";
  let selection = 0;
  let lastQuery = "";

  const normalize = (value) => String(value || "")
    .toLowerCase()
    .replaceAll("ё", "е")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

  const plainText = (value) => String(value || "")
    .replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, "")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/!\[\[[^\]]+\]\]/g, " ")
    .replace(/\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]/g, (_, target, label) => label || target)
    .replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1")
    .replace(/[#>*_`~|]/g, " ")
    .replace(/\$+[^$]*\$+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const subjectName = (key) => SUBJECT_NAMES[key] || String(key || "").replaceAll("_", " ").replace(/\b\w/g, (m) => m.toUpperCase());
  const sectionName = (key) => SECTION_META[key]?.label || String(key || "").replaceAll("_", " ");
  const cohortName = (id) => manifest?.cohorts?.find((c) => c.id === id)?.title || id;
  const docHref = (path) => `#/doc/${encodeURIComponent(path)}`;

  function tokenize(value) {
    return new Set(normalize(value).match(/[\p{L}\p{N}]{2,}/gu) || []);
  }

  function editDistanceAtMostOne(a, b) {
    if (a === b) return true;
    if (Math.abs(a.length - b.length) > 1) return false;
    if (a.length === b.length) {
      let diff = 0;
      for (let i = 0; i < a.length; i++) if (a[i] !== b[i] && ++diff > 1) return false;
      return true;
    }
    const [shorter, longer] = a.length < b.length ? [a, b] : [b, a];
    let i = 0, j = 0, skipped = false;
    while (i < shorter.length && j < longer.length) {
      if (shorter[i] === longer[j]) { i++; j++; continue; }
      if (skipped) return false;
      skipped = true; j++;
    }
    return true;
  }

  function fuzzyTokenMatch(tokens, term) {
    if (term.length < 4) return false;
    for (const token of tokens) {
      if (token.startsWith(term) || term.startsWith(token)) return true;
      if (term.length >= 5 && token.length >= 4 && editDistanceAtMostOne(token, term)) return true;
    }
    return false;
  }

  function buildSearchIndex() {
    if (!manifest) return;
    searchIndex = (manifest.docs || [])
      .filter((d) => d.subject !== "repository")
      .map((d) => {
        const body = plainText(d.content);
        const title = d.title || "";
        const subject = subjectName(d.subject);
        const section = sectionName(d.section);
        const cohort = cohortName(d.cohort);
        const titleN = normalize(title);
        const subjectN = normalize(subject);
        const sectionN = normalize(section);
        const cohortN = normalize(cohort);
        const bodyN = normalize(body);
        return {
          doc: d,
          title,
          subject,
          section,
          cohort,
          body,
          titleN,
          subjectN,
          sectionN,
          cohortN,
          bodyN,
          tokens: tokenize(`${title} ${subject} ${section} ${cohort} ${body}`),
        };
      });
  }

  function matchTerm(item, term) {
    let score = 0;
    let matched = false;
    if (item.titleN.includes(term)) { score += 120; matched = true; }
    if (item.titleN.startsWith(term)) score += 55;
    if (item.subjectN.includes(term)) { score += 70; matched = true; }
    if (item.sectionN.includes(term)) { score += 60; matched = true; }
    if (item.cohortN.includes(term)) { score += 25; matched = true; }
    if (item.bodyN.includes(term)) { score += 12; matched = true; }
    if (!matched && fuzzyTokenMatch(item.tokens, term)) { score += 6; matched = true; }
    return { matched, score };
  }

  function rankedSearch(query) {
    const q = normalize(query);
    const terms = q.split(" ").filter(Boolean);
    const filtered = activeSection === "all" ? searchIndex : searchIndex.filter((item) => item.doc.section === activeSection);
    if (!terms.length) {
      return filtered
        .slice()
        .sort((a, b) => a.subject.localeCompare(b.subject, "ru") || a.section.localeCompare(b.section, "ru") || a.title.localeCompare(b.title, "ru", { numeric: true }))
        .slice(0, 30)
        .map((item) => ({ item, score: 0, matched: 0 }));
    }

    const minMatched = terms.length <= 2 ? terms.length : terms.length - 1;
    return filtered.map((item) => {
      let score = 0;
      let matched = 0;
      for (const term of terms) {
        const result = matchTerm(item, term);
        if (result.matched) matched++;
        score += result.score;
      }
      if (matched < minMatched) return null;
      if (matched === terms.length) score += 100;
      if (q && item.titleN.includes(q)) score += 260;
      if (q && item.subjectN.includes(q)) score += 100;
      if (q && item.sectionN.includes(q)) score += 90;
      if (q && item.bodyN.includes(q)) score += 35;
      if (item.doc.section === "notes") score += 4;
      return { item, score, matched };
    }).filter(Boolean).sort((a, b) => b.score - a.score || a.item.title.localeCompare(b.item.title, "ru", { numeric: true })).slice(0, 60);
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function escapeRegExp(value) { return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

  function highlight(value, terms) {
    let html = escapeHtml(value);
    const useful = terms.filter((t) => t.length >= 2).sort((a, b) => b.length - a.length);
    for (const term of useful) {
      const re = new RegExp(`(${escapeRegExp(term)})`, "giu");
      html = html.replace(re, "<mark>$1</mark>");
    }
    return html;
  }

  function makeSnippet(item, terms) {
    const body = item.body || "";
    if (!body) return "";
    const lower = body.toLowerCase().replaceAll("ё", "е");
    let pos = -1;
    for (const term of terms) {
      const found = lower.indexOf(term);
      if (found >= 0 && (pos < 0 || found < pos)) pos = found;
    }
    if (pos < 0) pos = 0;
    const start = Math.max(0, pos - 75);
    const end = Math.min(body.length, start + 210);
    let snippet = body.slice(start, end).trim();
    if (start > 0) snippet = `…${snippet}`;
    if (end < body.length) snippet += "…";
    return snippet;
  }

  function renderSearch() {
    if (!manifest) {
      searchResults.innerHTML = '<div class="empty">Индекс поиска загружается…</div>';
      return;
    }
    const terms = normalize(lastQuery).split(" ").filter(Boolean);
    const found = rankedSearch(lastQuery);
    selection = Math.min(selection, Math.max(0, found.length - 1));
    updateSearchStatus(found.length);

    if (!found.length) {
      searchResults.innerHTML = `<div class="search-empty"><strong>Ничего не найдено</strong><span>Попробуй часть слова, другое окончание или переключи раздел поиска.</span></div>`;
      return;
    }

    searchResults.innerHTML = found.map(({ item }, i) => {
      const snippet = makeSnippet(item, terms);
      return `<a class="search-item enhanced-search-item ${i === selection ? "selected" : ""}" data-index="${i}" href="${docHref(item.doc.path)}">
        <div class="search-item-topline">
          <div class="search-item-title">${highlight(item.title, terms)}</div>
          <span class="search-section-badge">${escapeHtml(item.section)}</span>
        </div>
        <div class="search-item-path">${escapeHtml(item.cohort)} · ${escapeHtml(item.subject)}</div>
        ${snippet ? `<div class="search-item-snippet">${highlight(snippet, terms)}</div>` : ""}
      </a>`;
    }).join("");

    searchResults.querySelectorAll("a.search-item").forEach((a) => a.addEventListener("click", () => searchDialog.close()));
  }

  function ensureSearchControls() {
    if (document.getElementById("searchFilters")) return;
    const controls = document.createElement("div");
    controls.className = "search-controls";
    controls.innerHTML = `<div id="searchFilters" class="search-filters" aria-label="Фильтр поиска"></div><div id="searchStatus" class="search-status"></div>`;
    searchResults.before(controls);
    const filterEl = controls.querySelector("#searchFilters");
    const options = [{ key: "all", label: "Все" }, ...Object.entries(SECTION_META).map(([key, value]) => ({ key, label: value.label }))];
    filterEl.innerHTML = options.map(({ key, label }) => `<button type="button" class="search-filter ${key === activeSection ? "active" : ""}" data-section="${key}">${escapeHtml(label)}</button>`).join("");
    filterEl.addEventListener("click", (event) => {
      const button = event.target.closest(".search-filter");
      if (!button) return;
      activeSection = button.dataset.section;
      selection = 0;
      filterEl.querySelectorAll(".search-filter").forEach((b) => b.classList.toggle("active", b === button));
      renderSearch();
      searchInput.focus();
    });
  }

  function updateSearchStatus(count) {
    const status = document.getElementById("searchStatus");
    if (!status) return;
    if (!lastQuery.trim()) status.textContent = activeSection === "all" ? "Начни вводить запрос" : `Раздел: ${sectionName(activeSection)}`;
    else status.textContent = `Найдено: ${count}`;
  }

  function enhanceNavigation() {
    nav.querySelectorAll(".nav-section-title").forEach((title) => {
      if (title.dataset.enhanced === "1") return;
      const label = title.textContent.trim();
      const entry = Object.entries(SECTION_META).find(([, meta]) => meta.label === label);
      const meta = entry?.[1] || { label, icon: "•" };
      const section = title.closest(".nav-section");
      const count = section ? [...section.children].filter((child) => child.classList?.contains("nav-link")).length : 0;
      title.dataset.enhanced = "1";
      title.classList.add("nav-section-title-enhanced");
      title.innerHTML = `<span class="nav-section-icon" aria-hidden="true">${escapeHtml(meta.icon)}</span><span class="nav-section-label">${escapeHtml(label)}</span>${count ? `<span class="nav-section-count">${count}</span>` : ""}`;
    });
  }

  const navObserver = new MutationObserver(() => enhanceNavigation());
  navObserver.observe(nav, { childList: true, subtree: true });
  enhanceNavigation();

  ensureSearchControls();

  fetch("content.json", { cache: "no-cache" })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      manifest = data;
      buildSearchIndex();
      if (searchDialog.open) renderSearch();
    })
    .catch((error) => {
      console.warn("Enhanced search index failed:", error);
    });

  searchInput.addEventListener("input", (event) => {
    event.stopImmediatePropagation();
    lastQuery = searchInput.value;
    selection = 0;
    renderSearch();
  }, true);

  searchInput.addEventListener("keydown", (event) => {
    if (!searchDialog.open) return;
    const items = [...searchResults.querySelectorAll(".search-item")];
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      event.stopImmediatePropagation();
      if (!items.length) return;
      selection = (selection + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      renderSearch();
      searchResults.querySelector(".selected")?.scrollIntoView({ block: "nearest" });
      return;
    }
    if (event.key === "Enter" && items.length) {
      event.preventDefault();
      event.stopImmediatePropagation();
      const selected = searchResults.querySelector(".selected") || items[0];
      searchDialog.close();
      location.hash = selected.getAttribute("href");
    }
  }, true);

  document.getElementById("searchBtn")?.addEventListener("click", () => {
    setTimeout(() => {
      lastQuery = searchInput.value;
      selection = 0;
      renderSearch();
    }, 0);
  });

  addEventListener("keydown", (event) => {
    if ((event.key === "/" && !/INPUT|TEXTAREA/.test(document.activeElement?.tagName || "")) || ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k")) {
      setTimeout(() => {
        if (!searchDialog.open) return;
        lastQuery = searchInput.value;
        selection = 0;
        renderSearch();
      }, 0);
    }
  });
})();
