(() => {
  function mountQuiz() {
    const host = document.getElementById("second-order-curves-quiz");
    if (!host || host.dataset.quizMounted === "1" || typeof window.renderSecondOrderCurvesQuiz !== "function") return;
    host.dataset.quizMounted = "1";
    window.renderSecondOrderCurvesQuiz(host);
  }

  function start() {
    const content = document.getElementById("content");
    if (content) {
      new MutationObserver(mountQuiz).observe(content, { childList: true, subtree: true });
    }
    mountQuiz();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
  else start();

  addEventListener("hashchange", () => setTimeout(mountQuiz, 0));
})();
