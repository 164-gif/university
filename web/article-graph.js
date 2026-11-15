(() => {
  "use strict";
  const button = document.getElementById("graphBtn");
  if (!button || !window.matchMedia("(min-width: 1024px) and (pointer: fine)").matches) return;

  const SUBJECT_NAMES = {
    physics: "Физика",
    analytical_geometry: "Аналитическая геометрия",
    mathematical_analysis: "Математический анализ",
    obzh: "ОБЖ / БЖД",
    introduction_to_information_security: "Введение в информационную безопасность",
    information_security_fundamentals: "Основы информационной безопасности",
  };
  const SECTION_NAMES = {
    overview: "Главная", notes: "Конспекты", summaries: "Кратко",
    practice: "Практические работы", concepts: "Термины",
    exam: "Подготовка и тесты", laboratory_physics: "Лабораторные",
    checks: "Проверка информации", transcripts: "Расшифровки",
    raw: "Исходники", root: "Общее",
  };
  const SECTION_ORDER = [
    "overview", "notes", "summaries", "practice", "concepts", "exam",
    "laboratory_physics", "checks", "transcripts", "raw", "root"
  ];
  const COLORS = ["#6086df", "#25a895", "#bc82e7", "#d59b48", "#4ba8c9", "#d4779e", "#78a96c", "#ae8d70"];
  const collator = new Intl.Collator("ru", { numeric: true, sensitivity: "base" });
  const root = document.createElement("section");
  root.id = "article-graph";
  root.innerHTML =
    '<div class="graph-panel" role="dialog" aria-modal="true" aria-label="Граф связей статей" tabindex="-1" hidden>' +
      '<div class="graph-header"><div><strong>Граф знаний</strong><small class="graph-stats"></small></div>' +
        '<button type="button" class="graph-close icon-btn" aria-label="Закрыть граф">×</button></div>' +
      '<div class="graph-controls">' +
        '<input class="graph-search" type="search" placeholder="Поиск статей…" aria-label="Поиск статьи в графе">' +
        '<select class="graph-scope" aria-label="Режим отображения">' +
          '<option value="map">Карта предметов</option><option value="subject">Статьи предмета</option>' +
          '<option value="cohort">Предметы моего курса</option><option value="near">Связи текущей статьи</option>' +
        '</select>' +
        '<button class="graph-back" type="button" hidden>← Все предметы</button>' +
        '<label class="graph-links"><input type="checkbox" class="graph-show-links"> Все ссылки</label>' +
        '<button class="graph-fit" type="button" title="Вписать граф в окно">Вписать</button>' +
      '</div>' +
      '<div class="graph-area"><canvas class="graph-canvas" aria-label="Карта знаний: выберите узел для просмотра. Двойной щелчок открывает статью."></canvas>' +
        '<div class="graph-legend">Сплошные линии — структура предметов · Цветные линии — ссылки между статьями</div>' +
        '<div class="graph-tips">Колесо — масштаб · Перетаскивание — перемещение · Двойной щелчок — открыть статью</div>' +
      '</div><aside class="graph-details" aria-live="polite"></aside></div>';
  document.body.append(root);

  const $ = selector => root.querySelector(selector);
  const panel = $(".graph-panel"), canvas = $(".graph-canvas"), ctx = canvas.getContext("2d");
  const search = $(".graph-search"), scopeControl = $(".graph-scope"), details = $(".graph-details");
  const stats = $(".graph-stats"), back = $(".graph-back"), linksControl = $(".graph-show-links");
  const esc = value => String(value ?? "").replace(/[&<>"']/g, char =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  const norm = path => String(path || "").replace(/\\/g, "/").split("/").reduce((parts, part) => {
    if (part === "..") parts.pop();
    else if (part && part !== ".") parts.push(part);
    return parts;
  }, []).join("/");
  const stem = path => path.split("/").pop().replace(/\.(md|txt)$/i, "");
  const subjectName = key => SUBJECT_NAMES[key] || key.replaceAll("_", " ");
  const sectionName = key => SECTION_NAMES[key] || key.replaceAll("_", " ");
  const groupKey = doc => doc.cohort + "::" + doc.subject;
  const docId = path => "doc:" + path;
  const groupId = key => "group:" + key;
  const sectionId = (key, section) => "section:" + key + ":" + section;
  const docHash = path => "#/doc/" + encodeURIComponent(path);
  const activePath = () => {
    const match = location.hash.match(/^#\/doc\/([^?]+)/);
    if (!match) return "";
    try { return decodeURIComponent(match[1]); } catch { return ""; }
  };

  let docs = [], byPath = new Map(), references = [], adjacency = new Map(), groups = [];
  let mode = "map", activeGroup = "", selectedPath = "", chosen = "", hovered = "";
  let nodes = [], nodeById = new Map(), structure = [], shownReferences = [];
  let query = "", showLinks = false, scale = 1, ox = 0, oy = 0, w = 1, h = 1, dragging = null;
  let previousFocus = null;
  let pendingFrame = 0;

  function resolveDoc(input, source) {
    let raw = input.trim().split("|")[0].split("#")[0];
    try { raw = decodeURIComponent(raw); } catch {}
    raw = raw.replace(/\.(md|txt)$/i, "");
    if (!raw) return null;
    const srcParts = source.split("/");
    const dir = source.slice(0, source.lastIndexOf("/"));
    const subjectRoot = srcParts[0] === "cohorts" ? srcParts.slice(0, 4).join("/") : srcParts.slice(0, 2).join("/");
    const guesses = raw.includes("/") ? [norm(dir + "/" + raw), norm(raw), norm(subjectRoot + "/" + raw)] : [];
    for (const guess of guesses) {
      for (const candidate of [guess, guess + ".md", guess + ".txt"]) {
        if (byPath.has(candidate)) return candidate;
      }
    }
    const matches = docs.filter(doc =>
      stem(doc.path).toLocaleLowerCase("ru") === raw.toLocaleLowerCase("ru") ||
      doc.title.toLocaleLowerCase("ru") === raw.toLocaleLowerCase("ru"));
    const current = byPath.get(source);
    return (matches.find(doc => doc.cohort === current?.cohort && doc.subject === current?.subject) ||
      matches.find(doc => doc.cohort === current?.cohort) || matches[0])?.path || null;
  }

  function build(data) {
    docs = (data.docs || []).filter(doc => doc.subject !== "repository");
    byPath = new Map(docs.map(doc => [norm(doc.path), doc]));
    adjacency = new Map(docs.map(doc => [doc.path, new Set()]));
    const seen = new Set();
    for (const doc of docs) {
      const text = doc.content || "";
      const targets = [];
      for (const match of text.matchAll(/(?<!!)\[\[([^\]\n]+)\]\]/g)) targets.push(match[1]);
      for (const match of text.matchAll(/(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)/g)) {
        if (/\.(md|txt)(?:#.*)?$/i.test(match[1])) targets.push(match[1]);
      }
      for (const target of targets) {
        const other = resolveDoc(target, doc.path);
        if (!other || other === doc.path || !byPath.has(other)) continue;
        const id = [doc.path, other].sort().join("\0");
        if (seen.has(id)) continue;
        seen.add(id);
        references.push([doc.path, other]);
        adjacency.get(doc.path).add(other);
        adjacency.get(other).add(doc.path);
      }
    }
    const index = new Map();
    for (const doc of docs) {
      const key = groupKey(doc);
      if (!index.has(key)) index.set(key, {
        key, cohort: doc.cohort, subject: doc.subject, label: subjectName(doc.subject), docs: []
      });
      index.get(key).docs.push(doc);
    }
    groups = [...index.values()].sort((a, b) =>
      collator.compare(a.cohort, b.cohort) || collator.compare(a.label, b.label));
  }

  function addNode(node) {
    nodes.push(node);
    nodeById.set(node.id, node);
  }
  function link(from, to, type = "structure") {
    structure.push({ from, to, type });
  }
  function sectionSort(a, b) {
    const ia = SECTION_ORDER.indexOf(a), ib = SECTION_ORDER.indexOf(b);
    return (ia < 0 ? 100 : ia) - (ib < 0 ? 100 : ib) || collator.compare(sectionName(a), sectionName(b));
  }
  function filteredDocs(list) {
    if (!query) return list;
    const q = query.toLocaleLowerCase("ru");
    return list.filter(doc =>
      (doc.title + " " + doc.path + " " + subjectName(doc.subject) + " " + sectionName(doc.section || "root"))
        .toLocaleLowerCase("ru").includes(q));
  }
  function listSections(documents) {
    const sections = new Map();
    for (const doc of documents) {
      const key = doc.section || "root";
      if (!sections.has(key)) sections.set(key, []);
      sections.get(key).push(doc);
    }
    return [...sections].sort((a, b) => sectionSort(a[0], b[0]))
      .map(([name, list]) => [name, list.sort((a, b) => collator.compare(a.title, b.title))]);
  }

  // The overview is a deterministic mind map: each subject owns a separate radial branch.
  // Its section cards are summaries, not invented links between articles.
  function layoutMap(onlyCohort = false) {
    const current = byPath.get(activePath());
    const visibleGroups = groups.filter(group =>
      (!onlyCohort || !current || group.cohort === current.cohort) &&
      filteredDocs(group.docs).length > 0);
    const n = visibleGroups.length;
    addNode({
      id: "hub", type: "hub", title: "University Notes",
      subtitle: visibleGroups.reduce((sum, group) => sum + filteredDocs(group.docs).length, 0) + " материалов",
      x: -104, y: -29, w: 208, h: 58
    });
    const radius = Math.max(285, n * 48), branchLength = n > 12 ? 360 : 275;
    visibleGroups.forEach((group, index) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * index) / Math.max(n, 1);
      const color = COLORS[index % COLORS.length], count = filteredDocs(group.docs).length;
      const cx = Math.cos(angle) * radius, cy = Math.sin(angle) * radius;
      const groupNode = {
        id: groupId(group.key), type: "subject", groupKey: group.key, color,
        title: group.label, subtitle: count + " статей",
        x: cx - 95, y: cy - 27, w: 190, h: 54
      };
      addNode(groupNode); link("hub", groupNode.id);
      const sections = listSections(filteredDocs(group.docs));
      const sectionCx = Math.cos(angle) * (radius + branchLength);
      const sectionCy = Math.sin(angle) * (radius + branchLength);
      sections.forEach(([name, documents], si) => {
        const cols = sections.length <= 2 ? 1 : 2;
        const row = Math.floor(si / cols), column = si % cols;
        const originX = sectionCx - (cols === 2 ? 161 : 78);
        const originY = sectionCy - (Math.ceil(sections.length / cols) * 46) / 2;
        const node = {
          id: sectionId(group.key, name), type: "section", groupKey: group.key,
          section: name, color,
          title: sectionName(name), subtitle: documents.length + " материалов",
          x: originX + column * 168, y: originY + row * 46, w: 154, h: 37
        };
        addNode(node); link(groupNode.id, node.id);
      });
    });
  }

  // Inside one subject, columns show an explicit hierarchy instead of a force simulation.
  function layoutSubject() {
    const group = groups.find(item => item.key === activeGroup) ||
      groups.find(item => item.key === groupKey(byPath.get(activePath()) || {})) || groups[0];
    if (!group) return;
    activeGroup = group.key;
    const sections = listSections(filteredDocs(group.docs));
    let y = 40;
    const pieces = [];
    for (const [name, documents] of sections) {
      const cols = documents.length > 45 ? 4 : documents.length > 14 ? 3 : documents.length > 4 ? 2 : 1;
      const rows = Math.ceil(documents.length / cols);
      const height = Math.max(86, rows * 42 + 33);
      pieces.push({ name, documents, cols, y, height });
      y += height + 25;
    }
    addNode({
      id: groupId(group.key), type: "subject", groupKey: group.key, color: COLORS[0],
      title: group.label, subtitle: filteredDocs(group.docs).length + " статей",
      x: -275, y: Math.max(40, y / 2 - 28), w: 212, h: 56
    });
    for (const piece of pieces) {
      const section = {
        id: sectionId(group.key, piece.name), type: "section", groupKey: group.key,
        section: piece.name, color: COLORS[0], title: sectionName(piece.name),
        subtitle: piece.documents.length + " материалов",
        x: 0, y: piece.y + piece.height / 2 - 24, w: 175, h: 48
      };
      addNode(section); link(groupId(group.key), section.id);
      piece.documents.forEach((doc, index) => {
        const column = index % piece.cols, row = Math.floor(index / piece.cols);
        const node = {
          id: docId(doc.path), type: "doc", path: doc.path, groupKey: group.key, color: COLORS[0],
          title: doc.title, subtitle: "", x: 245 + column * 226,
          y: piece.y + 10 + row * 42, w: 213, h: 34
        };
        addNode(node); link(section.id, node.id);
      });
    }
  }

  // Explicit inbound/outbound references form two neat lanes in local view.
  function layoutNear() {
    const focus = byPath.get(selectedPath) || byPath.get(activePath());
    if (!focus) { layoutMap(); return; }
    selectedPath = focus.path;
    const inbound = [], outbound = [];
    for (const [from, to] of references) {
      if (from === focus.path) outbound.push(to);
      else if (to === focus.path) inbound.push(from);
    }
    const fromSet = new Set(inbound), toSet = new Set(outbound);
    const left = [...fromSet].sort((a, b) => collator.compare(byPath.get(a).title, byPath.get(b).title));
    const right = [...toSet].filter(path => !fromSet.has(path))
      .sort((a, b) => collator.compare(byPath.get(a).title, byPath.get(b).title));
    addNode({
      id: docId(focus.path), type: "doc", path: focus.path, title: focus.title, subtitle: "",
      x: -116, y: -23, w: 232, h: 46, color: COLORS[0]
    });
    [
      { paths: left, x: -505, name: "Ссылаются на статью" },
      { paths: right, x: 280, name: "Ссылки из статьи" }
    ].forEach((lane, laneIndex) => {
      if (!lane.paths.length) return;
      addNode({
        id: "lane:" + laneIndex, type: "label", title: lane.name, subtitle: "",
        x: lane.x, y: -Math.max(150, lane.paths.length * 25) - 53, w: 230, h: 38
      });
      const height = lane.paths.length * 46;
      lane.paths.forEach((path, index) => {
        const doc = byPath.get(path);
        if (!doc || (query && !filteredDocs([doc]).length)) return;
        addNode({
          id: docId(path), type: "doc", path, groupKey: groupKey(doc),
          title: doc.title, subtitle: subjectName(doc.subject),
          x: lane.x, y: index * 46 - height / 2, w: 232, h: 39,
          color: COLORS[laneIndex + 1]
        });
        link(laneIndex === 0 ? docId(path) : docId(focus.path),
          laneIndex === 0 ? docId(focus.path) : docId(path), "reference");
      });
    });
  }

  function refresh() {
    nodes = []; nodeById = new Map(); structure = [];
    if (mode === "subject") layoutSubject();
    else if (mode === "near") layoutNear();
    else layoutMap(mode === "cohort");
    const paths = new Set(nodes.filter(node => node.type === "doc").map(node => node.path));
    shownReferences = references.filter(([a, b]) => paths.has(a) && paths.has(b));
    back.hidden = mode !== "subject";
    linksControl.closest("label").hidden = mode !== "subject";
    stats.textContent = nodes.filter(n => n.type === "doc").length
      ? nodes.filter(n => n.type === "doc").length + " статей · " + shownReferences.length + " ссылок"
      : groups.length + " предметов · выберите предмет для просмотра статей";
    if (!nodeById.has(chosen)) chosen = "";
    resize();
    fit();
    updateDetails();
  }

  function colors() {
    const style = getComputedStyle(document.documentElement);
    return {
      bg: style.getPropertyValue("--bg").trim() || "#ffffff",
      panel: style.getPropertyValue("--panel").trim() || "#ffffff",
      secondary: style.getPropertyValue("--panel-2").trim() || "#f2f3f7",
      text: style.getPropertyValue("--text").trim() || "#15181f",
      muted: style.getPropertyValue("--muted").trim() || "#667280",
      border: style.getPropertyValue("--border").trim() || "#dfe3eb",
      accent: style.getPropertyValue("--accent").trim() || "#4665e5"
    };
  }
  function rounded(x, y, width, height, radius) {
    const r = Math.min(radius, height / 2, width / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }
  function trimText(text, maxWidth) {
    const str = String(text);
    if (ctx.measureText(str).width <= maxWidth) return str;
    let low = 0, high = str.length;
    while (low < high) {
      const mid = Math.ceil((low + high) / 2);
      if (ctx.measureText(str.slice(0, mid) + "…").width <= maxWidth) low = mid;
      else high = mid - 1;
    }
    return str.slice(0, low) + "…";
  }
  const center = node => ({ x: node.x + node.w / 2, y: node.y + node.h / 2 });
  function edgeLine(a, b, color, width, curved) {
    if (!a || !b) return;
    const p = center(a), q = center(b);
    ctx.beginPath();
    if (curved && (mode === "subject" || mode === "near")) {
      const startX = p.x < q.x ? p.x + a.w / 2 : p.x - a.w / 2;
      const endX = p.x < q.x ? q.x - b.w / 2 : q.x + b.w / 2;
      const middle = (startX + endX) / 2;
      ctx.moveTo(startX, p.y);
      ctx.bezierCurveTo(middle, p.y, middle, q.y, endX, q.y);
    } else {
      ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y);
    }
    ctx.lineWidth = width; ctx.strokeStyle = color; ctx.stroke();
  }
  function render() {
    if (panel.hidden) return;
    ctx.clearRect(0, 0, w, h);
    const c = colors();
    ctx.save();
    ctx.translate(ox, oy);
    ctx.scale(scale, scale);
    const focus = chosen.startsWith("doc:") ? chosen.slice(4) : selectedPath;
    const related = adjacency.get(focus) || new Set();

    for (const line of structure) {
      const a = nodeById.get(line.from), b = nodeById.get(line.to);
      const color = line.type === "reference" ? "#50b8a4" : c.border;
      edgeLine(a, b, color, line.type === "reference" ? 1.8 : 1.5, true);
    }
    if (mode === "subject") {
      for (const [from, to] of shownReferences) {
        const emphasized = from === focus || to === focus || from === hovered.slice(4) || to === hovered.slice(4);
        if (showLinks || emphasized) {
          edgeLine(nodeById.get(docId(from)), nodeById.get(docId(to)),
            emphasized ? "#25a895" : "rgba(96,139,184,.35)", emphasized ? 2.5 : 1, true);
        }
      }
    }

    for (const node of nodes) {
      const active = chosen === node.id || hovered === node.id;
      const relatedDoc = node.path && related.has(node.path);
      const tint = node.color || c.accent;
      rounded(node.x, node.y, node.w, node.h, node.type === "hub" ? 19 : node.type === "subject" ? 12 : 8);
      ctx.fillStyle = active ? c.secondary : c.panel;
      ctx.fill();
      ctx.lineWidth = active ? 2.4 : node.type === "doc" ? 1.1 : 1.6;
      ctx.strokeStyle = active ? c.accent : relatedDoc ? "#25a895" : node.type === "hub" ? c.accent : node.type === "subject" || node.type === "section" ? tint : c.border;
      ctx.stroke();
      if (node.type !== "hub" && node.type !== "label") {
        ctx.fillStyle = tint;
        rounded(node.x + 1, node.y + 5, 4, node.h - 10, 2);
        ctx.fill();
      }
      const left = node.x + (node.type === "hub" || node.type === "label" ? 12 : 15);
      const textSpace = node.w - (left - node.x) - 10;
      ctx.fillStyle = c.text;
      ctx.font = (node.type === "subject" || node.type === "hub" ? "600 15px " : "500 13px ") + "system-ui";
      ctx.textBaseline = "middle";
      ctx.fillText(trimText(node.title, textSpace), left,
        node.y + (node.subtitle ? node.h * 0.37 : node.h / 2));
      if (node.subtitle) {
        ctx.font = "11px system-ui"; ctx.fillStyle = c.muted;
        ctx.fillText(trimText(node.subtitle, textSpace), left, node.y + node.h * 0.72);
      }
    }
    ctx.restore();
  }
  function requestRender() {
    if (pendingFrame) return;
    pendingFrame = requestAnimationFrame(() => { pendingFrame = 0; render(); });
  }
  function resize() {
    if (panel.hidden) return;
    const rect = canvas.getBoundingClientRect();
    const dpi = Math.min(window.devicePixelRatio || 1, 2);
    w = Math.max(1, rect.width); h = Math.max(1, rect.height);
    canvas.width = Math.round(w * dpi); canvas.height = Math.round(h * dpi);
    ctx.setTransform(dpi, 0, 0, dpi, 0, 0);
    render();
  }
  function fit() {
    if (!nodes.length) return;
    const minX = Math.min(...nodes.map(node => node.x));
    const maxX = Math.max(...nodes.map(node => node.x + node.w));
    const minY = Math.min(...nodes.map(node => node.y));
    const maxY = Math.max(...nodes.map(node => node.y + node.h));
    scale = Math.max(0.1, Math.min(1.5, Math.min((w - 95) / (maxX - minX || 1), (h - 95) / (maxY - minY || 1))));
    ox = w / 2 - (minX + maxX) / 2 * scale;
    oy = h / 2 - (minY + maxY) / 2 * scale;
    render();
  }
  const toWorld = (x, y) => ({ x: (x - ox) / scale, y: (y - oy) / scale });
  function hitTest(x, y) {
    const p = toWorld(x, y);
    for (let i = nodes.length - 1; i >= 0; i--) {
      const node = nodes[i];
      if (p.x >= node.x && p.x <= node.x + node.w && p.y >= node.y && p.y <= node.y + node.h) return node;
    }
    return null;
  }
  function showSubject(key) {
    activeGroup = key;
    mode = "subject"; scopeControl.value = mode;
    selectedPath = "";
    chosen = "";
    refresh();
  }
  function choose(node) {
    if (!node) return;
    if (node.type === "subject" || node.type === "section") {
      showSubject(node.groupKey);
      if (node.type === "section") {
        const target = nodes.find(item => item.type === "section" && item.section === node.section);
        if (target) chosen = target.id;
        render(); updateDetails();
      }
      return;
    }
    chosen = node.id;
    if (node.path) selectedPath = node.path;
    updateDetails();
    render();
  }
  function updateDetails() {
    const selected = nodeById.get(chosen);
    if (!selected) {
      details.innerHTML = '<strong>Карта материалов</strong><p>Предметы распределены по отдельным направлениям. Выберите предмет, затем статью, чтобы увидеть её ссылки.</p>' +
        '<p class="graph-help">Карточки разделов показывают структуру, а цветные линии — реальные ссылки в конспектах.</p>';
      return;
    }
    if (!selected.path) {
      details.innerHTML = '<strong>' + esc(selected.title) + '</strong><p>Раздел: ' + esc(selected.subtitle) + '</p>' +
        '<p class="graph-help">Выберите карточку статьи в правой части схемы.</p>';
      return;
    }
    const doc = byPath.get(selected.path);
    const related = [...(adjacency.get(selected.path) || [])].map(path => byPath.get(path)).filter(Boolean)
      .sort((a, b) => collator.compare(a.title, b.title));
    details.innerHTML = '<strong>' + esc(doc.title) + '</strong>' +
      '<small>' + esc(subjectName(doc.subject)) + ' · ' + esc(sectionName(doc.section || "root")) + '</small>' +
      '<a class="graph-open" href="' + docHash(doc.path) + '">Открыть статью ↗</a>' +
      '<h3>Связанные статьи (' + related.length + ')</h3>' +
      (related.length
        ? '<div class="graph-related">' + related.map(item =>
            '<button type="button" data-path="' + esc(item.path) + '">' + esc(item.title) + '</button>').join("") + '</div>'
        : '<p>Явных ссылок на другие статьи пока нет.</p>');
  }

  function open(value) {
    panel.hidden = !value;
    root.classList.toggle("is-open", value);
    button.setAttribute("aria-expanded", String(value));
    if (value) {
      previousFocus = document.activeElement;
      mode = "map"; scopeControl.value = "map";
      const doc = byPath.get(activePath());
      if (doc) { selectedPath = doc.path; activeGroup = groupKey(doc); }
      refresh();
      panel.focus();
    } else {
      previousFocus?.focus?.();
    }
  }
  button.addEventListener("click", () => open(panel.hidden));
  $(".graph-close").addEventListener("click", () => open(false));
  $(".graph-fit").addEventListener("click", fit);
  back.addEventListener("click", () => {
    mode = "map"; scopeControl.value = mode; chosen = ""; refresh();
  });
  search.addEventListener("input", () => { query = search.value.trim(); refresh(); });
  scopeControl.addEventListener("change", () => {
    mode = scopeControl.value;
    if (mode === "subject") {
      const doc = byPath.get(activePath());
      if (doc && !activeGroup) activeGroup = groupKey(doc);
    }
    if (mode === "near") selectedPath = activePath() || selectedPath;
    chosen = ""; refresh();
  });
  linksControl.addEventListener("change", () => { showLinks = linksControl.checked; render(); });
  details.addEventListener("click", event => {
    const control = event.target.closest("[data-path]");
    if (!control) return;
    const path = control.dataset.path, doc = byPath.get(path);
    if (!doc) return;
    if (!nodeById.has(docId(path))) {
      activeGroup = groupKey(doc);
      mode = "subject"; scopeControl.value = mode;
      refresh();
    }
    selectedPath = path;
    chosen = docId(path);
    updateDetails(); render();
  });
  window.addEventListener("keydown", event => {
    if (event.key === "Escape" && !panel.hidden) { event.preventDefault(); open(false); }
    if (event.key === "Tab" && !panel.hidden) {
      const focusable = [...panel.querySelectorAll('button:not([disabled]):not([hidden]),input:not([disabled]),select:not([disabled]),a[href]')]
        .filter(item => item.getClientRects().length);
      if (focusable.length) {
        const first = focusable[0], last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    }
  });
  window.addEventListener("hashchange", () => {
    if (panel.hidden) return;
    const doc = byPath.get(activePath());
    if (!doc) return;
    selectedPath = doc.path;
    activeGroup = groupKey(doc);
    refresh();
  });
  function point(event) {
    const box = canvas.getBoundingClientRect();
    return { x: event.clientX - box.left, y: event.clientY - box.top };
  }
  canvas.addEventListener("pointerdown", event => {
    const p = point(event);
    dragging = { x: p.x, y: p.y, moved: false, node: hitTest(p.x, p.y) };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", event => {
    const p = point(event);
    if (dragging) {
      const dx = p.x - dragging.x, dy = p.y - dragging.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) dragging.moved = true;
      if (dragging.moved) { ox += dx; oy += dy; render(); }
      dragging.x = p.x; dragging.y = p.y;
    } else {
      const hit = hitTest(p.x, p.y);
      const next = hit?.id || "";
      if (next !== hovered) { hovered = next; requestRender(); }
      canvas.style.cursor = hit ? "pointer" : "grab";
    }
  });
  canvas.addEventListener("pointerup", () => {
    if (dragging?.node && !dragging.moved) choose(dragging.node);
    dragging = null;
  });
  canvas.addEventListener("pointercancel", () => { dragging = null; });
  canvas.addEventListener("dblclick", event => {
    const p = point(event), node = hitTest(p.x, p.y);
    if (node?.path) { location.hash = docHash(node.path); open(false); }
  });
  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    const p = point(event), world = toWorld(p.x, p.y);
    scale = Math.min(4, Math.max(.1, scale * Math.exp(-event.deltaY * .001)));
    ox = p.x - world.x * scale; oy = p.y - world.y * scale;
    render();
  }, { passive: false });
  new ResizeObserver(() => { if (!panel.hidden) resize(); }).observe(canvas);
  new MutationObserver(() => { if (!panel.hidden) render(); }).observe(document.documentElement, {
    attributes: true, attributeFilter: ["data-theme"]
  });

  button.disabled = true;
  fetch("content.json", { cache: "no-cache" })
    .then(response => { if (!response.ok) throw Error("HTTP " + response.status); return response.json(); })
    .then(data => { build(data); button.disabled = false; })
    .catch(error => { button.title = "Граф недоступен: " + error.message; });
})();