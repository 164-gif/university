/* Directed article references and explicitly labeled title-based recommendations.
 * This module never presents structural proximity or lexical similarity as a citation. */
(function (global) {
  "use strict";

  const IMAGE_EXTENSIONS = /\.(?:svg|png|jpe?g|gif|webp|avif|pdf|mp[34]|wav|ogg)$/i;
  const STOP_WORDS = new Set([
    "конспект", "конспекты", "лекция", "лекции", "тема", "темы", "глава", "главы",
    "раздел", "вопрос", "вопросы", "ответ", "ответы", "подготовка", "тест", "тесты",
    "проверка", "задание", "задания", "решение", "решения", "пример", "примеры",
    "теория", "теории", "задача", "задачи", "практика", "практические", "работа", "работы",
    "вариант", "варианты", "лабораторная", "лабораторные", "полный", "полная",
    "полное", "краткий", "краткая", "основы", "введение", "математика", "физика",
    "геометрия", "аналитическая", "курс", "часть", "первый", "второй", "третья",
    "основной", "новая", "новое", "новый", "семинар", "контрольная", "результат",
    "все", "всех", "для", "при", "или", "это", "как", "что", "где", "из", "на",
    "по", "об", "от", "до", "со", "под", "без", "над", "про", "с", "и", "в",
    "самостоятельная", "обзор", "материал", "материалы", "учебный", "предмет"
  ]);

  function normalizePath(path) {
    return String(path || "").replace(/\\/g, "/").split("/").reduce((parts, part) => {
      if (part === "..") parts.pop();
      else if (part && part !== ".") parts.push(part);
      return parts;
    }, []).join("/");
  }
  function noExtension(path) { return path.replace(/\.(?:md|txt)$/i, ""); }
  function stem(path) { return noExtension(path.split("/").pop()); }
  function normalizeTitle(value) {
    return String(value || "").trim().replace(/\s+/g, " ").toLocaleLowerCase("ru");
  }
  function groupOf(doc) { return doc.cohort + "::" + doc.subject; }
  function sorted(values) { return [...values].sort((a, b) => a.localeCompare(b, "ru")); }

  function visibleText(content) {
    const rows = String(content || "").split(/\r?\n/), out = [];
    let fence = "";
    const marker = new RegExp("^ {0,3}([\\x60~])\\1{2,}");
    for (const row of rows) {
      const m = row.match(marker);
      if (m) {
        if (!fence) { fence = m[1]; continue; }
        if (fence === m[1]) { fence = ""; continue; }
      }
      if (!fence) out.push(row);
    }
    return out.join("\n")
      .replace(/<!--[\s\S]*?-->/g, "")
      .replace(/\x60[^\x60\n]*\x60/g, "");
  }

  function extractTargets(content) {
    const text = visibleText(content), targets = [];
    for (const match of text.matchAll(/!?\[\[([^\]\n]+)\]\]/g)) {
      const link = match[1].split("|", 1)[0].trim();
      if (link && !IMAGE_EXTENSIONS.test(link.split("#")[0])) targets.push(link);
    }
    for (const match of text.matchAll(/!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^)\s]+)(?:\s+["'][^"']*["'])?\s*\)/g)) {
      let link = match[1].trim();
      if (link.startsWith("<") && link.endsWith(">")) link = link.slice(1, -1);
      if (/^(?:https?:|mailto:|tel:|data:|javascript:)/i.test(link)) continue;
      if (/^\s*#\/doc\//i.test(link) || /\.(?:md|txt)(?:[#?].*)?$/i.test(link)) {
        targets.push(link);
      }
    }
    return targets;
  }

  function buildRelationships(inputDocs) {
    const docs = inputDocs.filter(doc => doc.subject !== "repository")
      .map(doc => ({ ...doc, path: normalizePath(doc.path) }));
    const byPath = new Map(docs.map(doc => [doc.path, doc]));
    const byName = new Map();
    function index(name, path) {
      const key = normalizeTitle(name);
      if (!key) return;
      if (!byName.has(key)) byName.set(key, new Set());
      byName.get(key).add(path);
    }
    docs.forEach(doc => {
      index(stem(doc.path), doc.path);
      index(doc.title, doc.path);
    });

    function resolve(rawLink, currentPath) {
      let link = String(rawLink || "").split("|")[0].trim();
      if (link.startsWith("#/doc/")) link = link.slice(6);
      else if (/^\/?#\/doc\//i.test(link)) link = link.replace(/^\/?#\/doc\//i, "");
      try { link = decodeURIComponent(link); } catch {}
      link = link.split("#")[0].split("?")[0].trim().replace(/^<|>$/g, "");
      if (!link || IMAGE_EXTENSIONS.test(link)) return { path: null, reason: "not-document" };
      const source = byPath.get(currentPath);
      const parts = currentPath.split("/");
      const parent = parts.slice(0, -1).join("/");
      const root = parts[0] === "cohorts" ? parts.slice(0, 4).join("/") : parts.slice(0, 2).join("/");
      const withoutExt = noExtension(link.replace(/^\/+/, ""));
      const candidates = [
        normalizePath(link), normalizePath(parent + "/" + link),
        normalizePath(root + "/" + link),
        normalizePath(withoutExt), normalizePath(parent + "/" + withoutExt),
        normalizePath(root + "/" + withoutExt)
      ];
      for (const candidate of candidates) {
        for (const path of [candidate, candidate + ".md", candidate + ".txt"]) {
          if (byPath.has(path)) return { path };
        }
      }
      const names = byName.get(normalizeTitle(withoutExt)) ||
        byName.get(normalizeTitle(stem(withoutExt)));
      if (!names?.size) return { path: null, reason: "missing" };
      const matches = [...names].map(path => byPath.get(path));
      const scopes = [
        matches.filter(doc => doc.subject === source?.subject && doc.cohort === source?.cohort),
        matches.filter(doc => doc.cohort === source?.cohort),
        matches
      ];
      for (const scope of scopes) {
        if (scope.length === 1) return { path: scope[0].path };
        if (scope.length > 1) return { path: null, reason: "ambiguous" };
      }
      return { path: null, reason: "missing" };
    }

    const outgoing = new Map(docs.map(doc => [doc.path, new Set()]));
    const incoming = new Map(docs.map(doc => [doc.path, new Set()]));
    const adjacency = new Map(docs.map(doc => [doc.path, new Set()]));
    const unresolved = new Map(docs.map(doc => [doc.path, []]));
    const references = [];

    for (const doc of docs) {
      for (const raw of extractTargets(doc.content)) {
        const target = resolve(raw, doc.path);
        if (!target.path) {
          if (target.reason !== "not-document" &&
              !unresolved.get(doc.path).some(item => item.target === raw)) {
            unresolved.get(doc.path).push({ target: raw, reason: target.reason || "missing" });
          }
          continue;
        }
        if (target.path === doc.path || outgoing.get(doc.path).has(target.path)) continue;
        outgoing.get(doc.path).add(target.path);
        incoming.get(target.path).add(doc.path);
        adjacency.get(doc.path).add(target.path);
        adjacency.get(target.path).add(doc.path);
        references.push([doc.path, target.path]);
      }
    }

    // Related topics are ONLY a search hint, not proof of an explicit citation.
    // Use significant title words in the same course AND subject to prevent noisy cross-topic links.
    function words(title) {
      const normalized = normalizeTitle(title)
        .replace(/(?:№|#)\s*\d+/g, " ")
        .replace(/\d+(?:[.,]\d+)*/g, " ");
      const result = new Map();
      for (const match of normalized.matchAll(/[\p{L}]+/gu)) {
        const word = match[0];
        if (word.length < 5 || STOP_WORDS.has(word)) continue;
        const short = word.length > 7 ? word.slice(0, 7) : word;
        if (!STOP_WORDS.has(short)) result.set(short, word);
      }
      return result;
    }
    const suggestions = new Map(docs.map(doc => [doc.path, []]));
    const perGroup = new Map();
    for (const doc of docs) {
      const group = groupOf(doc);
      if (!perGroup.has(group)) perGroup.set(group, []);
      perGroup.get(group).push({ doc, terms: words(doc.title) });
    }
    for (const candidates of perGroup.values()) {
      for (let i = 0; i < candidates.length; i++) {
        const a = candidates[i];
        if (!a.terms.size) continue;
        for (let j = i + 1; j < candidates.length; j++) {
          const b = candidates[j];
          if (!b.terms.size || adjacency.get(a.doc.path).has(b.doc.path)) continue;
          const overlap = [...a.terms.keys()].filter(term => b.terms.has(term));
          const score = 2 * overlap.length / (a.terms.size + b.terms.size);
          const strongSingle = a.terms.size === 1 && b.terms.size === 1 &&
            overlap.length === 1 && overlap[0].length >= 6;
          if ((overlap.length < 2 || score < 0.5) && !strongSingle) continue;
          const common = overlap.map(term => a.terms.get(term)).slice(0, 3).join(", ");
          const reason = "Совпадают слова в названиях: " + common;
          suggestions.get(a.doc.path).push({ path: b.doc.path, score, reason });
          suggestions.get(b.doc.path).push({ path: a.doc.path, score, reason });
        }
      }
    }
    for (const [path, items] of suggestions) {
      items.sort((a, b) => b.score - a.score ||
        byPath.get(a.path).title.localeCompare(byPath.get(b.path).title, "ru"));
      suggestions.set(path, items.slice(0, 5));
    }

    const crossGroup = new Map();
    for (const [a, b] of references) {
      const from = groupOf(byPath.get(a)), to = groupOf(byPath.get(b));
      if (from === to) continue;
      const key = sorted([from, to]).join("\0");
      const entry = crossGroup.get(key) || { a: sorted([from, to])[0], b: sorted([from, to])[1], count: 0 };
      entry.count++;
      crossGroup.set(key, entry);
    }
    const unresolvedCount = [...unresolved.values()].reduce((sum, list) => sum + list.length, 0);
    return {
      references, outgoing, incoming, adjacency, suggestions, unresolved,
      crossGroups: [...crossGroup.values()],
      metrics: { direct: references.length, unresolved: unresolvedCount,
        suggestions: [...suggestions.values()].reduce((sum, items) => sum + items.length, 0) }
    };
  }

  const api = Object.freeze({ buildRelationships, extractTargets, normalizePath });
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.UniversityRelations = api;
})(typeof window === "undefined" ? {} : window);
