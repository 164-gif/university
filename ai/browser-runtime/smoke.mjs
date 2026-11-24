import {chromium} from "playwright-core";
import assert from "node:assert/strict";
const browser=await chromium.launch({channel:"chrome",headless:true,args:["--no-sandbox"]});
try{
 const page=await browser.newPage();
 await page.goto("http://127.0.0.1:8123/");
 await page.locator("#aiSearchBtn").click();
 await page.waitForURL("**/ai-search.html");
 await page.waitForFunction(()=>!document.querySelector("#ai-submit").disabled||document.querySelector("#ai-status").textContent.startsWith("ИИ-поиск недоступен"),{},{timeout:120000});
 assert.equal(await page.locator("#ai-submit").isEnabled(),true,await page.locator("#ai-status").innerText());
 await page.locator("#ai-query").fill("Что такое скалярное произведение?");
 await page.locator("#ai-submit").click();
 await page.waitForFunction(()=>document.querySelector("#ai-results li")||document.querySelector("#ai-status").textContent.startsWith("ИИ-поиск недоступен"),{},{timeout:120000});
 assert.ok(await page.locator("#ai-results li").count(),await page.locator("#ai-status").innerText());
 assert.match(await page.locator("#ai-results li a").first().getAttribute("href"),/^\.\/#\/doc\//);
 console.log("AI_BROWSER_SMOKE_OK",await page.locator("#ai-status").innerText());
}finally{await browser.close();}
