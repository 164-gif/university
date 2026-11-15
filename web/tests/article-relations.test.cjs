const test = require("node:test");
const assert = require("node:assert/strict");
const { buildRelationships, extractTargets } = require("../article-relations.js");

const doc = (path, title, content = "", subject = "analytical_geometry", cohort = "c1") =>
  ({ path, title, content, subject, cohort, section: "notes" });

test("parses wikilinks, Markdown links and site hash routes but skips code and media", () => {
  const main = doc("subjects/analytical_geometry/notes/a.md", "Точка и вектор",
    "[[b|Текст]]\n[b](../notes/c.md#title)\n[site](#/doc/subjects%2Fphysics%2Fnotes%2Fp.md)\n" +
    "![img](plot.png)\n" +
    "\u0060\u0060\u0060md\n[[missing_fake]]\n\u0060\u0060\u0060\n[[missing_real]]");
  const b = doc("subjects/analytical_geometry/notes/b.md", "Скалярное произведение");
  const c = doc("subjects/analytical_geometry/notes/c.md", "Векторное произведение");
  const p = doc("subjects/physics/notes/p.md", "Механика", "", "physics");
  const result = buildRelationships([main, b, c, p]);

  assert.deepEqual(new Set(result.outgoing.get(main.path)), new Set([b.path, c.path, p.path]));
  assert.equal(result.incoming.get(b.path).has(main.path), true);
  assert.equal(result.unresolved.get(main.path).length, 1);
  assert.equal(result.unresolved.get(main.path)[0].target, "missing_real");
  assert.equal(result.crossGroups.length, 1);
  assert.equal(result.crossGroups[0].count, 1);
  assert.equal(extractTargets("![x](img.png)\n[[t]]").includes("t"), true);
});

test("tracks mutual references as two directed links", () => {
  const a = doc("subjects/analytical_geometry/a.md", "A", "[[b]]");
  const b = doc("subjects/analytical_geometry/b.md", "B", "[[a]]");
  const result = buildRelationships([a, b]);
  assert.equal(result.references.length, 2);
  assert.equal(result.incoming.get(a.path).has(b.path), true);
  assert.equal(result.outgoing.get(a.path).has(b.path), true);
  assert.equal(result.adjacency.get(a.path).has(b.path), true);
});

test("ambiguous names remain unresolved rather than connecting arbitrary pages", () => {
  const source = doc("subjects/analytical_geometry/notes/source.md", "Source", "[[same]]");
  const x = doc("subjects/analytical_geometry/notes/same.md", "Same");
  const y = doc("subjects/analytical_geometry/concepts/same.md", "Same");
  const result = buildRelationships([source, x, y]);
  // A local same.md file resolves without ambiguity; fallback names are ambiguous.
  assert.equal(result.outgoing.get(source.path).has(x.path), true);
  const alt = doc("subjects/analytical_geometry/raw/source.md", "Source", "[[same]]");
  const ambiguous = buildRelationships([alt, x, y]);
  assert.equal(ambiguous.outgoing.get(alt.path).size, 0);
  assert.equal(ambiguous.unresolved.get(alt.path)[0].reason, "ambiguous");
});

test("related titles are suggestions only and never merged into actual references", () => {
  const a = doc("subjects/analytical_geometry/notes/1.md", "Скалярное произведение");
  const b = doc("subjects/analytical_geometry/summaries/2.md", "Скалярного произведения");
  const c = doc("subjects/physics/notes/3.md", "Скалярное произведение", "", "physics");
  const result = buildRelationships([a, b, c]);
  assert.equal(result.references.length, 0);
  assert.equal(result.suggestions.get(a.path).some(s => s.path === b.path), true);
  assert.equal(result.suggestions.get(a.path).some(s => s.path === c.path), false);
});

test("existing explicit links are not repeated as suggestions", () => {
  const a = doc("subjects/analytical_geometry/notes/1.md", "Скалярное произведение", "[[2]]");
  const b = doc("subjects/analytical_geometry/summaries/2.md", "Скалярного произведения");
  const result = buildRelationships([a, b]);
  assert.equal(result.references.length, 1);
  assert.equal(result.suggestions.get(a.path).length, 0);
});
