(() => {
  const STYLE_ID = "second-order-curves-quiz-styles";

  const F = (prompt, correct, accepted) => ({ type: "formula", prompt, correct, accepted });
  const C = (prompt, options, answer, correct) => ({ type: "choice", prompt, options, answer, correct });

  const questions = [
    F("Запиши уравнение окружности с центром M(a,b) и радиусом R.", "(x−a)² + (y−b)² = R²", ["(x-a)^2+(y-b)^2=r^2", "(x-a)²+(y-b)²=r²"]),
    F("Запиши уравнение окружности с центром в начале координат.", "x² + y² = R²", ["x^2+y^2=r^2", "x²+y²=r²"]),
    F("Запиши расстояние от точки (x,y) до центра окружности M(a,b), равное радиусу R.", "R = √((x−a)² + (y−b)²)", ["r=sqrt((x-a)^2+(y-b)^2)", "r=√((x-a)^2+(y-b)^2)", "r=√((x-a)²+(y-b)²)"]),

    F("Запиши каноническое уравнение эллипса.", "x²/a² + y²/b² = 1", ["x^2/a^2+y^2/b^2=1", "x²/a²+y²/b²=1"]),
    F("Запиши основное фокальное свойство эллипса.", "F₁M + F₂M = 2a", ["f1m+f2m=2a", "f_1m+f_2m=2a"]),
    F("Запиши связь a, b и c для эллипса.", "c² = a² − b²", ["c^2=a^2-b^2", "c²=a²-b²", "b^2=a^2-c^2", "b²=a²-c²"]),
    F("Запиши эксцентриситет эллипса.", "ε = c/a", ["epsilon=c/a", "ε=c/a", "e=c/a"]),

    F("Запиши каноническое уравнение гиперболы с ветвями вдоль Ox.", "x²/a² − y²/b² = 1", ["x^2/a^2-y^2/b^2=1", "x²/a²-y²/b²=1"]),
    F("Запиши связь a, b и c для гиперболы.", "c² = a² + b²", ["c^2=a^2+b^2", "c²=a²+b²"]),
    F("Запиши эксцентриситет гиперболы.", "ε = c/a", ["epsilon=c/a", "ε=c/a", "e=c/a"]),
    F("Запиши основное фокальное свойство гиперболы.", "|F₁M − F₂M| = 2a", ["|f1m-f2m|=2a", "|f_1m-f_2m|=2a", "abs(f1m-f2m)=2a"]),
    F("Запиши уравнения асимптот гиперболы x²/a² − y²/b² = 1.", "y = ±(b/a)x", ["y=±(b/a)x", "y=+-(b/a)x", "y=±b/a*x", "y=±bx/a"]),
    F("Для прямоугольной гиперболы a=b. Запиши её эксцентриситет.", "ε = √2", ["epsilon=sqrt2", "epsilon=√2", "ε=√2", "e=√2"]),

    F("Запиши уравнения директрис эллипса или гиперболы с главной осью Ox.", "x = ±a/ε", ["x=±a/epsilon", "x=±a/ε", "x=+-a/epsilon", "x=±a^2/c", "x=±a²/c"]),
    F("Запиши фокусно-директрисное свойство коники.", "MF/d = ε", ["mf/d=epsilon", "mf/d=ε", "mf=epsilon*d", "mf=εd"]),

    F("Запиши каноническое уравнение параболы, ветви которой направлены вправо.", "y² = 2px", ["y^2=2px", "y²=2px"]),
    F("Запиши координаты фокуса параболы y²=2px.", "F(p/2, 0)", ["f(p/2,0)", "f(p÷2,0)", "f(p/2;0)"]),
    F("Запиши уравнение директрисы параболы y²=2px.", "x = −p/2", ["x=-p/2", "x=−p/2"]),

    F("Запиши касательную к эллипсу x²/a² + y²/b² = 1 в точке M₀(x₀,y₀).", "xx₀/a² + yy₀/b² = 1", ["x*x0/a^2+y*y0/b^2=1", "xx_0/a^2+yy_0/b^2=1", "xx₀/a²+yy₀/b²=1"]),
    F("Запиши касательную к гиперболе x²/a² − y²/b² = 1 в точке M₀(x₀,y₀).", "xx₀/a² − yy₀/b² = 1", ["x*x0/a^2-y*y0/b^2=1", "xx_0/a^2-yy_0/b^2=1", "xx₀/a²-yy₀/b²=1"]),

    F("Запиши общее уравнение кривой второго порядка.", "Ax² + 2Bxy + Cy² + 2Dx + 2Ey + F = 0", ["ax^2+2bxy+cy^2+2dx+2ey+f=0", "ax²+2bxy+cy²+2dx+2ey+f=0"]),
    F("Запиши квадратичную часть общего уравнения кривой второго порядка.", "Ax² + 2Bxy + Cy²", ["ax^2+2bxy+cy^2", "ax²+2bxy+cy²"]),
    F("Запиши величину δ для классификации квадратичной части.", "δ = AC − B²", ["delta=ac-b^2", "δ=ac-b^2", "δ=ac-b²", "ac-b^2"]),

    F("Запиши формулу поворота осей для координаты x.", "x = x′cosα − y′sinα", ["x=x'cosalpha-y'sinalpha", "x=x′cosα-y′sinα"]),
    F("Запиши формулу поворота осей для координаты y.", "y = x′sinα + y′cosα", ["y=x'sinalpha+y'cosalpha", "y=x′sinα+y′cosα"]),
    F("После поворота запиши параллельный перенос для x′.", "x′ = x″ + x₀", ["x'=x''+x_0", "x′=x″+x₀", "x'=x''+x0"]),
    F("После поворота запиши параллельный перенос для y′.", "y′ = y″ + y₀", ["y'=y''+y_0", "y′=y″+y₀", "y'=y''+y0"]),

    C("Что означает δ > 0 для невырожденной кривой второго порядка?", ["Эллиптический тип", "Гиперболический тип", "Параболический тип", "Всегда окружность"], 0, "Эллиптический тип"),
    C("Что означает δ < 0 для невырожденной кривой второго порядка?", ["Параболический тип", "Гиперболический тип", "Эллиптический тип", "Всегда две прямые"], 1, "Гиперболический тип"),
    C("Что означает δ = 0 для невырожденной кривой второго порядка?", ["Эллиптический тип", "Гиперболический тип", "Параболический тип", "Только окружность"], 2, "Параболический тип")
  ];

  const palette = [
    ["x", "x"], ["y", "y"], ["a", "a"], ["b", "b"], ["c", "c"], ["p", "p"], ["R", "R"],
    ["A", "A"], ["B", "B"], ["C", "C"], ["D", "D"], ["E", "E"], ["F", "F"], ["M", "M"], ["d", "d"],
    ["²", "^2"], ["₀", "_0"], ["₁", "_1"], ["₂", "_2"], ["′", "'"], ["″", "''"],
    ["ε", "ε"], ["δ", "δ"], ["α", "α"], ["√", "sqrt()", 5], ["±", "±"], ["| |", "||", 1],
    ["+", "+"], ["−", "-"], ["=", "="], ["/", "/"], ["(", "("], [")", ")"], [",", ","],
    ["0", "0"], ["1", "1"], ["2", "2"], ["cos", "cos"], ["sin", "sin"], ["дробь", "()/()", 1]
  ];

  function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .geo-quiz{max-width:920px;margin:0 auto}.geo-quiz-head{margin-bottom:20px}.geo-quiz-head h1{margin-bottom:8px}.geo-quiz-subtitle{color:var(--muted,#697386);line-height:1.55}.geo-quiz-progress{display:flex;align-items:center;gap:12px;margin:18px 0}.geo-quiz-progress-bar{flex:1;height:10px;background:rgba(127,127,127,.18);border-radius:999px;overflow:hidden}.geo-quiz-progress-fill{height:100%;width:0;background:currentColor;transition:width .25s}.geo-quiz-progress-text{min-width:88px;text-align:right;font-variant-numeric:tabular-nums}.geo-card{border:1px solid rgba(127,127,127,.28);border-radius:18px;padding:22px;background:rgba(127,127,127,.055);transition:.2s}.geo-card.correct{border-color:#2f9e63;background:rgba(47,158,99,.11)}.geo-card.wrong{border-color:#d94b4b;background:rgba(217,75,75,.11)}.geo-question-number{font-size:.88rem;color:var(--muted,#697386);margin-bottom:8px}.geo-question{font-size:1.16rem;font-weight:650;line-height:1.45;margin-bottom:18px}.geo-formula-input{width:100%;box-sizing:border-box;border:1px solid rgba(127,127,127,.34);border-radius:12px;padding:14px 15px;font:600 1.05rem ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--surface,rgba(127,127,127,.06));color:inherit}.geo-formula-input:focus{outline:2px solid rgba(72,101,229,.45);outline-offset:1px}.geo-palette{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}.geo-symbol{border:1px solid rgba(127,127,127,.28);border-radius:10px;padding:8px 11px;min-width:40px;background:rgba(127,127,127,.08);color:inherit;font:600 .96rem ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;cursor:pointer}.geo-symbol:hover{background:rgba(72,101,229,.12);border-color:rgba(72,101,229,.45)}.geo-symbol:disabled{opacity:.45;cursor:default}.geo-edit-actions{display:flex;gap:12px;margin-top:9px}.geo-mini{border:0;background:transparent;color:var(--muted,#697386);cursor:pointer;padding:5px 2px}.geo-options{display:grid;gap:10px}.geo-option{text-align:left;padding:13px 14px;border:1px solid rgba(127,127,127,.3);border-radius:12px;background:rgba(127,127,127,.06);color:inherit;cursor:pointer;font:inherit}.geo-option.selected{outline:2px solid rgba(72,101,229,.6)}.geo-option:disabled{cursor:default}.geo-feedback{margin-top:16px;padding:12px 14px;border-radius:12px;display:none;line-height:1.5}.geo-card.correct .geo-feedback,.geo-card.wrong .geo-feedback{display:block}.geo-card.correct .geo-feedback{background:rgba(47,158,99,.12)}.geo-card.wrong .geo-feedback{background:rgba(217,75,75,.12)}.geo-actions{display:flex;justify-content:flex-end;gap:10px;margin-top:18px}.geo-btn{border:0;border-radius:11px;padding:11px 16px;font:600 1rem inherit;cursor:pointer;background:#4665e5;color:#fff}.geo-btn.secondary{background:rgba(127,127,127,.16);color:inherit}.geo-btn:disabled{opacity:.45;cursor:not-allowed}.geo-results{text-align:center;padding:28px 8px 4px}.geo-score{font-size:3rem;line-height:1;font-weight:800;margin:12px 0 6px}.geo-score-details{font-size:1.08rem;color:var(--muted,#697386)}.geo-review{margin-top:24px;text-align:left;display:grid;gap:9px}.geo-review-row{padding:12px 14px;border-radius:11px;background:rgba(127,127,127,.07);border-left:4px solid #d94b4b}.geo-review-row.ok{border-left-color:#2f9e63}.geo-review-answer{margin-top:5px;color:var(--muted,#697386);font-size:.93rem}@media(max-width:700px){.geo-card{padding:16px;border-radius:14px}.geo-question{font-size:1.05rem}.geo-symbol{min-width:42px;padding:10px}.geo-actions{justify-content:stretch}.geo-btn{flex:1}}
    `;
    document.head.appendChild(style);
  }

  function normalize(value) {
    return String(value || "").trim().toLowerCase()
      .replace(/\s+/g, "")
      .replace(/[−–—]/g, "-")
      .replace(/[′’]/g, "'")
      .replace(/″/g, "''")
      .replace(/²/g, "^2")
      .replace(/₀/g, "_0").replace(/₁/g, "_1").replace(/₂/g, "_2")
      .replace(/ε|\\varepsilon|\\epsilon/g, "epsilon")
      .replace(/δ|\\delta/g, "delta")
      .replace(/α|\\alpha/g, "alpha")
      .replace(/√\s*\(([^()]*)\)/g, "sqrt($1)")
      .replace(/√([a-z0-9^]+)/g, "sqrt($1)")
      .replace(/\\sqrt\{([^{}]*)\}/g, "sqrt($1)")
      .replace(/\\cos/g, "cos").replace(/\\sin/g, "sin")
      .replace(/\\left|\\right/g, "")
      .replace(/\\cdot|·|\*/g, "")
      .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, "($1)/($2)")
      .replace(/[{}]/g, "")
      .replace(/\+\-/g, "±")
      .replace(/abs\(([^()]*)\)/g, "|$1|")
      .replace(/;/g, ",");
  }

  function answerIsCorrect(question, value) {
    const v = normalize(value);
    return question.accepted.some(x => normalize(x) === v);
  }

  function insertAtCursor(input, text, cursorOffset = text.length) {
    const start = input.selectionStart ?? input.value.length;
    const end = input.selectionEnd ?? input.value.length;
    input.value = input.value.slice(0, start) + text + input.value.slice(end);
    const pos = start + cursorOffset;
    input.focus(); input.setSelectionRange(pos, pos);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function renderFormulaControls(root, state) {
    root.innerHTML = `<input class="geo-formula-input" type="text" inputmode="text" autocomplete="off" autocapitalize="off" spellcheck="false" placeholder="Введите формулу…" aria-label="Ответ формулой"><div class="geo-palette"></div><div class="geo-edit-actions"><button class="geo-mini" type="button" data-edit="back">⌫ удалить</button><button class="geo-mini" type="button" data-edit="clear">Очистить</button></div>`;
    const input = root.querySelector(".geo-formula-input");
    const paletteEl = root.querySelector(".geo-palette");
    palette.forEach(([label, token, offset]) => {
      const b = document.createElement("button"); b.type="button"; b.className="geo-symbol"; b.textContent=label;
      b.addEventListener("click", () => insertAtCursor(input, token, offset ?? token.length)); paletteEl.appendChild(b);
    });
    root.querySelector('[data-edit="back"]').addEventListener("click", () => {
      const s=input.selectionStart??input.value.length,e=input.selectionEnd??s;
      if(s!==e){input.value=input.value.slice(0,s)+input.value.slice(e);input.setSelectionRange(s,s)}else if(s>0){input.value=input.value.slice(0,s-1)+input.value.slice(s);input.setSelectionRange(s-1,s-1)}input.focus();
    });
    root.querySelector('[data-edit="clear"]').addEventListener("click",()=>{input.value="";input.focus()});
    input.value = state.answer || "";
    input.addEventListener("input",()=>state.answer=input.value);
    setTimeout(()=>input.focus(),0);
  }

  function renderChoiceControls(root, state, q) {
    root.innerHTML = `<div class="geo-options"></div>`;
    const box=root.firstElementChild;
    q.options.forEach((text,i)=>{const b=document.createElement("button");b.type="button";b.className="geo-option"+(state.answer===i?" selected":"");b.textContent=text;b.addEventListener("click",()=>{state.answer=i;box.querySelectorAll(".geo-option").forEach((x,j)=>x.classList.toggle("selected",j===i))});box.appendChild(b)});
  }

  function mountQuiz(host) {
    ensureStyles();
    const states = questions.map(()=>({answer:null,checked:false,correct:false}));
    let index=0;
    host.innerHTML=`<div class="geo-quiz"><div class="geo-quiz-head"><h1>Тест по формулам кривых второго порядка</h1><div class="geo-quiz-subtitle">Все основные формулы из тем 11–14. Тема 15 не входит. Формулу можно печатать с клавиатуры или собирать кнопками.</div></div><div class="geo-quiz-progress"><div class="geo-quiz-progress-bar"><div class="geo-quiz-progress-fill"></div></div><div class="geo-quiz-progress-text"></div></div><div class="geo-card"><div class="geo-question-number"></div><div class="geo-question"></div><div class="geo-answer"></div><div class="geo-feedback"></div><div class="geo-actions"><button class="geo-btn secondary" data-prev>Назад</button><button class="geo-btn" data-check>Проверить</button><button class="geo-btn" data-next style="display:none">Далее</button></div></div></div>`;
    const card=host.querySelector('.geo-card'), num=host.querySelector('.geo-question-number'), text=host.querySelector('.geo-question'), answer=host.querySelector('.geo-answer'), feedback=host.querySelector('.geo-feedback'), fill=host.querySelector('.geo-quiz-progress-fill'), ptxt=host.querySelector('.geo-quiz-progress-text');
    const prev=host.querySelector('[data-prev]'), check=host.querySelector('[data-check]'), next=host.querySelector('[data-next]');

    function render(){
      if(index>=questions.length){renderResults();return}
      const q=questions[index],s=states[index];
      card.classList.remove('correct','wrong'); if(s.checked) card.classList.add(s.correct?'correct':'wrong');
      num.textContent=`Задание ${index+1} из ${questions.length}`; text.textContent=q.prompt;
      fill.style.width=`${(index/questions.length)*100}%`; ptxt.textContent=`${index+1}/${questions.length}`;
      feedback.innerHTML=s.checked?(s.correct?`Верно. <strong>${q.correct}</strong>`:`Неверно. Правильный ответ: <strong>${q.correct}</strong>`):'';
      if(q.type==='formula') renderFormulaControls(answer,s); else renderChoiceControls(answer,s,q);
      [...answer.querySelectorAll('input,button')].forEach(el=>el.disabled=s.checked);
      prev.disabled=index===0; check.style.display=s.checked?'none':''; next.style.display=s.checked?'':''; next.textContent=index===questions.length-1?'Результат':'Далее';
    }

    function validate(){
      const q=questions[index],s=states[index];
      if(q.type==='formula' && !String(s.answer||'').trim()) return;
      if(q.type==='choice' && (s.answer===null || s.answer===undefined)) return;
      s.correct=q.type==='formula'?answerIsCorrect(q,s.answer):s.answer===q.answer; s.checked=true; render();
    }

    function renderResults(){
      const right=states.filter(s=>s.correct).length, total=questions.length, pct=Math.round(right/total*100);
      fill.style.width='100%';ptxt.textContent=`${total}/${total}`;
      card.className='geo-card';
      card.innerHTML=`<div class="geo-results"><div>Тест завершён</div><div class="geo-score">${pct}%</div><div class="geo-score-details">Верно ${right} из ${total} · Ошибок ${total-right}</div><div class="geo-review">${questions.map((q,i)=>`<div class="geo-review-row ${states[i].correct?'ok':''}"><strong>${i+1}. ${q.prompt}</strong><div class="geo-review-answer">${states[i].correct?'Верно':'Неверно'} · Ваш ответ: ${escapeHtml(states[i].answer??'—')}<br>Правильный ответ: ${escapeHtml(q.correct)}</div></div>`).join('')}</div><div class="geo-actions"><button class="geo-btn" data-restart>Пройти ещё раз</button></div></div>`;
      card.querySelector('[data-restart]').addEventListener('click',()=>location.reload());
    }

    prev.addEventListener('click',()=>{if(index>0){index--;render()}}); check.addEventListener('click',validate); next.addEventListener('click',()=>{if(states[index].checked){index++;render()}});
    answer.addEventListener('keydown',e=>{if(e.key==='Enter'&&!states[index].checked){e.preventDefault();validate()}});
    render();
  }

  function escapeHtml(v){return String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}

  function tryMount(){const host=document.getElementById('second-order-curves-quiz');if(host&&!host.dataset.mounted){host.dataset.mounted='1';mountQuiz(host)}}
  window.mountSecondOrderCurvesQuiz=tryMount;
  document.addEventListener('DOMContentLoaded',tryMount);
  new MutationObserver(tryMount).observe(document.documentElement,{subtree:true,childList:true});
})();