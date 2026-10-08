(() => {
  const nav = document.getElementById("navigation");
  if (!nav) return;

  const stripPrefix = (text) => text.replace(/^\s*\d{1,3}\s*(?:[-—.:]|\))\s*/u, "").trim();
  const fileKey = (link) => {
    const path = link.dataset.path || "";
    const name = path.split("/").pop() || path;
    return name.normalize("NFKD");
  };

  function normalizeSection(section) {
    const links = [...section.querySelectorAll(":scope > a.nav-link[data-path]")];
    if (!links.length) return;

    links.sort((a, b) => fileKey(a).localeCompare(fileKey(b), "ru", { numeric: true, sensitivity: "base" }));
    links.forEach((link, index) => {
      section.appendChild(link);
      if (!link.dataset.originalTitle) link.dataset.originalTitle = stripPrefix(link.textContent || "");
      link.textContent = `${String(index + 1).padStart(2, "0")} — ${link.dataset.originalTitle}`;
    });
  }

  function normalizeNavigation() {
    nav.querySelectorAll(".nav-section").forEach(normalizeSection);
  }

  let tries = 0;
  const timer = setInterval(() => {
    tries += 1;
    if (nav.querySelector("a.nav-link[data-path]")) {
      normalizeNavigation();
      clearInterval(timer);
    } else if (tries >= 50) {
      clearInterval(timer);
    }
  }, 100);
})();
