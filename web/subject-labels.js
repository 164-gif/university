(() => {
  const replacements = new Map([
    ["Russian Statehood Fundamentals", "Основы российской государственности"],
  ]);

  function applySubjectLabels(root = document.body) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const value = node.nodeValue?.trim();
      if (!value || !replacements.has(value)) continue;
      node.nodeValue = node.nodeValue.replace(value, replacements.get(value));
    }
  }

  let scheduled = false;
  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      applySubjectLabels();
    });
  };

  new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
  addEventListener("hashchange", schedule);
  schedule();
})();
