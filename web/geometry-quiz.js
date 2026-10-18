(() => {
  const STYLE_ID = "second-order-curves-quiz-styles";

  const questions = [
    {
      type: "formula",
      prompt: "Запиши общее уравнение кривой второго порядка.",
      correct: "Ax² + 2Bxy + Cy² + 2Dx + 2Ey + F = 0",
      accepted: [
        "ax^2+2bxy+cy^2+2dx+2ey+f=0",
        "ax²+2bxy+cy²+2dx+2ey+f=0"
      ]
    },
    {
      type: "formula",
      prompt: "Запиши квадратичную часть общего уравнения кривой второго порядка.",
      correct: "Ax² + 2Bxy + Cy²",
      accepted: [
        "ax^2+2bxy+cy^2",
        "ax²+2bxy+cy²"
      ]
    },
    {
      type: "formula",
      prompt: "Запиши величину δ, используемую для классификации квадратичной части.",
      correct: "δ = AC − B²",
      accepted: [
        "delta=ac-b^2",
        "delta=ac-b²",
        "δ=ac-b^2",
        "δ=ac-b²",
        "ac-b^2",
        "ac-b²"
      ]
    },
    {
      type: "formula",
      prompt: "Запиши формулу поворота осей для координаты x.",
      correct: "x = x′ cos α − y′ sin α",
      accepted: [
        "x=x'cosalpha-y'sinalpha",
        "x=x′cosα-y′sinα",
        "x=x'cosα-y'sinα"
      ]
    },
    {
      type: "formula",
      prompt: "Запиши формулу поворота осей для координаты y.",
      correct: "y = x′ sin α + y′ cos α",
      accepted: [
        "y=x'sinalpha+y'cosalpha",
        "y=x′sinα+y′cosα",
        "y=x'sinα+y'cosα"
      ]
    },
    {
      type: "formula",
      prompt: "После поворота записывают параллельный перенос. Запиши формулу для x′.",
      correct: "x′ = x″ + x₀",
      accepted: [
        "x'=x''+x_0",
        "x′=x″+x₀",
        "x'=x''+x0"
      ]
    },
    {
      type: "formula",
      prompt: "Запиши вторую формулу параллельного переноса — для y′.",
      correct: "y′ = y″ + y₀",
      accepted: [
        "y'=y''+y_0",
        "y′=y″+y₀",
        "y'=y''+y0"
      ]
    },
    {
      type: "choice",
      prompt: "Какое преобразование используют, чтобы убрать смешанный член xy?",
      options: ["Поворот осей", "Параллельный перенос", "Масштабирование обеих осей", "Замена x на −x"],
      answer: 0,
      correct: "Поворот осей"
    },
    {
      type: "choice",
      prompt: "Для невырожденной кривой какого типа выполняется δ > 0?",
      options: ["Эллиптического", "Гиперболического", "Параболического", "Тип определить нельзя"],
      answer: 0,
      correct: "Эллиптического типа"
    },
    {
      type: "choice",
      prompt: "Для невырожденной кривой какого типа выполняется δ < 0?",
      options: ["Параболического", "Эллиптического", "Гиперболического", "Тип определить нельзя"],
      answer: 2,
      correct: "Гиперболического типа"
    },
    {
      type: "choice",
      prompt: "Для невырожденной кривой какого типа выполняется δ = 0?",
      options: ["Эллиптического", "Параболического", "Гиперболического", "Только окружности"],
      answer: 1,
      correct: "Параболического типа"
    },
    {
      type: "choice",
      prompt: "Что делает параллельный перенос при приведении общего уравнения?",
      options: [
        "Меняет начало координат и упрощает линейные члены",
        "Всегда превращает кривую в окружность",
        "Убирает смешанный член xy без поворота",
        "Меняет степень уравнения"
      ],
      answer: 0,
      correct: "Меняет начало координат и упрощает линейные члены"
    }
  ];

  const palette = [
    ["x", "x"], ["y", "y"], ["x²", "x^2"], ["y²", "y^2"], ["²", "^2"],
    ["₀", "_0"], ["′", "'"], ["″", "''"], ["δ", "δ"], ["α", "α"],
    ["A", "A"], ["B", "B"], ["C", "C"], ["D", "D"], ["E", "E"], ["F", "F"],
    ["+", "+"], ["−", "-"], ["=", "="], ["(", "("], [")", ")"],
    ["cos", "cos"], ["sin", "sin"], ["дробь", "\\frac{}{}", 6]
  ];

  function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .geo-quiz { max-width: 900px; margin: 0 auto; }
      .geo-quiz-head { margin-bottom: 20px; }
      .geo-quiz-head h1 { margin-bottom: 8px; }
      .geo-quiz-subtitle { color: var(--muted, #697386); line-height: 1.55; }
      .geo-quiz-progress { display:flex; align-items:center; gap:12px; margin:18px 0; }
      .geo-quiz-progress-bar { flex:1; height:10px; background:rgba(127,127,127,.18); border-radius:999px; overflow:hidden; }
      .geo-quiz-progress-fill { height:100%; width:0; background:currentColor; transition:width .25s ease; }
      .geo-quiz-progress-text { min-width:72px; text-align:right; font-variant-numeric:tabular-nums; }
      .geo-card { border:1px solid rgba(127,127,127,.28); border-radius:18px; padding:22px; background:rgba(127,127,127,.055); transition:border-color .2s, background .2s; }
      .geo-card.correct { border-color:#2f9e63; background:rgba(47,158,99,.11); }
      .geo-card.wrong { border-color:#d94b4b; background:rgba(217,75,75,.11); }
      .geo-question-number { font-size:.88rem; color:var(--muted, #697386); margin-bottom:8px; }
      .geo-question { font-size:1.16rem; font-weight:650; line-height:1.45; margin-bottom:18px; }
      .geo-formula-input { width:100%; box-sizing:border-box; border:1px solid rgba(127,127,127,.34); border-radius:12px; padding:14px 15px; font:600 1.05rem ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; background:var(--surface, rgba(127,127,127,.06)); color:inherit; }
      .geo-formula-input:focus { outline:2px solid rgba(72,101,229,.45); outline-offset:1px; }
      .geo-palette { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
      .geo-symbol { border:1px solid rgba(127,127,127,.28); border-radius:10px; padding:8px 11px; min-width:40px; background:rgba(127,127,127,.08); color:inherit; font:600 .96rem ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; cursor:pointer; }
      .geo-symbol:hover { background:rgba(72,101,229,.12); border-color:rgba(72,101,229,.45); }
      .geo-symbol:disabled { opacity:.45; cursor:default; }
      .geo-edit-actions { display:flex; gap:8px; margin-top:9px; }
      .geo-mini { border:0; background:transparent; color:var(--muted, #697386); cursor:pointer; padding:5px 2px; }
      .geo-options { display:grid; gap:10px; }
      .geo-option { text-align:left; padding:13px 14px; border:1px solid rgba(127,127,127,.3); border-radius:12px; background:rgba(127,127,127,.06); color:inherit; cursor:pointer; font:inherit; }
      .geo-option:hover { border-color:rgba(72,101,229,.5); background:rgba(72,101,229,.08); }
      .geo-option.selected { outline:2px solid rgba(72,101,229,.6); }
      .geo-option:disabled { cursor:default; }
      .geo-feedback { margin-top:16px; padding:12px 14px; border-radius:12px; display:none; line-height:1.5; }
      .geo-card.correct .geo-feedback, .geo-card.wrong .geo-feedback { display:block; }
      .geo-card.correct .geo-feedback { background:rgba(47,158,99,.12); }
      .geo-card.wrong .geo-feedback { background:rgba(217,75,75,.12); }
      .geo-actions { display:flex; justify-content:flex-end; gap:10px; margin-top:18px; }
      .geo-btn { border:0; border-radius:11px; padding:11px 16px; font:600 1rem inherit; cursor:pointer; background:#4665e5; color:white; }
      .geo-btn.secondary { background:rgba(127,127,127,.16); color:inherit; }
      .geo-btn:disabled { opacity:.45; cursor:not-allowed; }
      .geo-results { text-align:center; padding:28px 8px 4px; }
      .geo-score { font-size:3rem; line-height:1; font-weight:800; margin:12px 0 6px; }
      .geo-score-details { font-size:1.08rem; color:var(--muted, #697386); }
      .geo-review { margin-top:24px; text-align:left; display:grid; gap:9px; }
      .geo-review-row { padding:12px 14px; border-radius:11px; background:rgba(127,127,127,.07); border-left:4px solid #d94b4b; }
      .geo-review-row.ok { border-left-color:#2f9e63; }
      .geo-review-answer { margin-top:5px; color:var(--muted, #697386); font-size:.93rem; }
      @media (max-width: 700px) {
        .geo-card { padding:16px; border-radius:14px; }
        .geo-question { font-size:1.05rem; }
        .geo-symbol { flex:0 0 auto; min-width:42px; padding:10px; }
        .geo-actions { justify-content:stretch; }
        .geo-btn { flex:1; }
      }
    `;
    document.head.appendChild(style);
  }

  function normalize(value) {
    return String(value || "")
      .trim()
      .toLowerCase()
      .replace(/\s+/g, "")
      .replace(/[−–—]/g, "-")
      .replace(/[′’]/g, "'")
      .replace(/″/g, "''")
      .replace(/²/g, "^2")
      .replace(/₀/g, "_0")
      .replace(/\\delta|δ/g, "delta")
      .replace(/\\alpha|α/g, "alpha")
      .replace(/\\cos/g, "cos")
      .replace(/\\sin/g, "sin")
      .replace(/\\left|\\right/g, "")
      .replace(/\\cdot|·|\*/g, "")
      .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, "($1)/($2)")
      .replace(/[{}]/g, "");
  }

  function answerIsCorrect(question, value) {
    const v = normalize(value);
    return question.accepted.some((x) => normalize(x) === v);
  }

  function insertAtCursor(input, text, cursorOffset = text.length) {
    const start = input.selectionStart ?? input.value.length;
    const end = input.selectionEnd ?? input.value.length;
    input.value = input.value.slice(0, start) + text + input.value.slice(end);
    const pos = start + cursorOffset;
    input.focus();
    input.setSelectionRange(pos, pos);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function renderFormulaControls(root, state, question) {
    root.innerHTML = `
      <input class="geo-formula-input" type="text" inputmode="text" autocomplete="off" autocapitalize="off" spellcheck="false" placeholder="Введите формулу…" aria-label="Ответ формулой">
      <div class="geo-palette"></div>
      <div class="geo-edit-actions"><button class="geo-mini" type="button" data-edit="back">⌫ удалить</button><button class="geo-mini" type="button" data-edit="clear">Очистить</button></div>
    `;
    const input = root.querySelector(".geo-formula-input");
    const paletteEl = root.querySelector(".geo-palette");
    for (const [label, token, offset] of palette) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "geo-symbol";
      button.textContent = label;
      button.addEventListener("click", () => insertAtCursor(input, token, offset ?? token.length));
      paletteEl.appendChild(button);
    }
    root.querySelector('[data-edit="back"]').addEventListener("click", () => {
      const start = input.selectionStart ?? input.value.length;
      const end = input.selectionEnd ?? start;
      if (start !== end) {
        input.value = input.value.slice(0, start) + input.value.slice(end);
        input.setSelectionRange(start, start);
      } else if (start > 0) {
        input.value = input.value.slice(0, start - 1) + input.value.slice(start);
        input.setSelectionRange(start - 1, start - 1);
      }
      input.focus();
    });
    root.querySelector('[data-edit="clear"]').addEventListener("click", () => { input.value = ""; input.focus(); });
    input.addEventListener("input", () => { state.currentAnswer = input.value; });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !state.locked) {
        event.preventDefault();
        root.closest(".geo-card")?.querySelector('[data-action="check"]')?.click();
      }
    });
    setTimeout(() => input.focus(), 0);
  }

  function renderChoiceControls(root, state, question) {
    root.innerHTML = `<div class="geo-options"></div>`;
    const list = root.querySelector(".geo-options");
    question.options.forEach((label, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "geo-option";
      button.textContent = label;
      button.addEventListener("click", () => {
        if (state.locked) return;
        state.currentAnswer = index;
        list.querySelectorAll(".geo-option").forEach((el, i) => el.classList.toggle("selected", i === index));
      });
      list.appendChild(button);
    });
  }

  function renderQuestion(container, state) {
    const q = questions[state.index];
    const percent = Math.round((state.index / questions.length) * 100);
    container.innerHTML = `
      <div class="geo-quiz-head">
        <h1>Тест по кривым второго порядка</h1>
        <div class="geo-quiz-subtitle">Тема 14: общее уравнение, квадратичная часть, поворот и перенос координат, классификация по δ. Тема 15 в этот тест не входит.</div>
      </div>
      <div class="geo-quiz-progress">
        <div class="geo-quiz-progress-bar"><div class="geo-quiz-progress-fill" style="width:${percent}%"></div></div>
        <div class="geo-quiz-progress-text">${state.index + 1} / ${questions.length}</div>
      </div>
      <section class="geo-card">
        <div class="geo-question-number">Задание ${state.index + 1}</div>
        <div class="geo-question">${q.prompt}</div>
        <div class="geo-answer-area"></div>
        <div class="geo-feedback"></div>
        <div class="geo-actions">
          <button class="geo-btn" type="button" data-action="check">Проверить</button>
          <button class="geo-btn" type="button" data-action="next" hidden>${state.index + 1 === questions.length ? "Результат" : "Следующий вопрос"}</button>
        </div>
      </section>
    `;

    const card = container.querySelector(".geo-card");
    const answerArea = container.querySelector(".geo-answer-area");
    const feedback = container.querySelector(".geo-feedback");
    const checkButton = container.querySelector('[data-action="check"]');
    const nextButton = container.querySelector('[data-action="next"]');
    state.currentAnswer = q.type === "choice" ? null : "";
    state.locked = false;

    if (q.type === "formula") renderFormulaControls(answerArea, state, q);
    else renderChoiceControls(answerArea, state, q);

    checkButton.addEventListener("click", () => {
      if (state.locked) return;
      const empty = q.type === "formula" ? !String(state.currentAnswer || "").trim() : state.currentAnswer === null;
      if (empty) {
        feedback.style.display = "block";
        feedback.textContent = "Сначала введи или выбери ответ.";
        return;
      }
      const ok = q.type === "formula" ? answerIsCorrect(q, state.currentAnswer) : state.currentAnswer === q.answer;
      state.locked = true;
      state.results.push({ prompt: q.prompt, ok, given: q.type === "choice" ? q.options[state.currentAnswer] : state.currentAnswer, correct: q.correct });
      card.classList.add(ok ? "correct" : "wrong");
      feedback.style.display = "block";
      feedback.innerHTML = ok
        ? `<strong>Верно.</strong>`
        : `<strong>Неверно.</strong><br>Правильный ответ: <span class="geo-correct-answer">${q.correct}</span>`;
      answerArea.querySelectorAll("input,button").forEach((el) => { el.disabled = true; });
      checkButton.hidden = true;
      nextButton.hidden = false;
      nextButton.focus();
    });

    nextButton.addEventListener("click", () => {
      if (!state.locked) return;
      state.index += 1;
      if (state.index >= questions.length) renderResults(container, state);
      else renderQuestion(container, state);
    });
  }

  function renderResults(container, state) {
    const correct = state.results.filter((x) => x.ok).length;
    const total = questions.length;
    const percent = Math.round((correct / total) * 100);
    container.innerHTML = `
      <div class="geo-quiz-head"><h1>Результат теста</h1></div>
      <section class="geo-card">
        <div class="geo-results">
          <div class="geo-score">${percent}%</div>
          <div class="geo-score-details">Верно: <strong>${correct} из ${total}</strong> · Ошибок: <strong>${total - correct}</strong></div>
          <div class="geo-actions" style="justify-content:center;margin-top:20px"><button class="geo-btn" type="button" data-action="retry">Пройти ещё раз</button></div>
        </div>
        <div class="geo-review">
          ${state.results.map((r, i) => `<div class="geo-review-row ${r.ok ? "ok" : ""}"><strong>${i + 1}. ${r.ok ? "Верно" : "Ошибка"}</strong><div>${r.prompt}</div>${r.ok ? "" : `<div class="geo-review-answer">Твой ответ: ${escapeHtml(r.given || "—")}<br>Правильно: ${escapeHtml(r.correct)}</div>`}</div>`).join("")}
        </div>
      </section>
    `;
    container.querySelector('[data-action="retry"]').addEventListener("click", () => {
      state.index = 0;
      state.results = [];
      renderQuestion(container, state);
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c]));
  }

  window.renderSecondOrderCurvesQuiz = function renderSecondOrderCurvesQuiz(container) {
    ensureStyles();
    container.classList.add("geo-quiz");
    const state = { index: 0, results: [], currentAnswer: "", locked: false };
    renderQuestion(container, state);
  };
})();
