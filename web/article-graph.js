(() => {
  "use strict";
  if (!matchMedia("(min-width: 1024px) and (pointer: fine)").matches) return;
  const root = document.createElement("section");
  root.id = "article-graph";
  root.innerHTML = '<button class="graph-launch" type="button">◉ Граф связей</button><div class="graph-panel" hidden><header><strong>Граф знаний</strong><button class="graph-close" type="button" aria-label="Закрыть">×</button></header><div class="graph-controls"><input class="graph-search" type="search" placeholder="Поиск статьи…" aria-label="Поиск статьи"><select class="graph-scope" aria-label="Область графа"><option value="subject">Текущий предмет</option><option value="cohort">Учебный набор</option><option value="all">Весь сайт</option><option value="near">Связи статьи</option></select><button class="graph-fit" type="button">Вписать</button></div><div class="graph-area"><canvas class="graph-canvas"></canvas><div class="graph-tips">Колесо — масштаб · Перетаскивание — перемещение · Двойной щелчок — открыть статью</div></div><aside class="graph-details">Выберите статью на графе</aside></div>';
  document.body.append(root);
  const panel=root.querySelector(".graph-panel"), button=root.querySelector(".graph-launch"), canvas=root.querySelector("canvas"), ctx=canvas.getContext("2d"), detail=root.querySelector(".graph-details"), stat=root.querySelector("header strong");
  const esc = s => String(s||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const norm = s => String(s||"").replace(/\\/g,"/").split("/").reduce((a,b)=>{if(b==="..")a.pop();else if(b&&b!==".")a.push(b);return a;},[]).join("/");
  const stem = s => s.split("/").pop().replace(/\.(md|txt)$/i,"");
  const docPath = () => {try{return decodeURIComponent((location.hash.match(/^#\/doc\/([^?]+)/)||[])[1]||"");}catch{return "";}};
  let docs=[], lookup=new Map(), edges=[], adjacency=new Map(), nodes=[], shown=[], selected="", hover="", query="", scope="subject", positions=new Map(),scale=1, ox=0,oy=0,w=1,h=1,drag=null;
  const findLink = (input,source) => {
    const raw=input.split("|")[0].split("#")[0].replace(/\.(md|txt)$/i,"").trim();
    if(!raw)return null;
    const dir=source.slice(0,source.lastIndexOf("/")), parts=source.split("/");
    const base=source.startsWith("cohorts/")?parts.slice(0,4).join("/"):parts.slice(0,2).join("/");
    const candidates=raw.includes("/")?[norm(dir+"/"+raw),norm(base+"/"+raw),norm(raw)]:[];
    for(const c of candidates)for(const p of [c,c+".md",c+".txt"])if(lookup.has(p))return p;
    const matches=docs.filter(d=>stem(d.path).toLowerCase()===raw.toLowerCase()||d.title.toLowerCase()===raw.toLowerCase());
    const src=lookup.get(source);
    return (matches.find(d=>d.cohort===src?.cohort&&d.subject===src?.subject)||matches.find(d=>d.cohort===src?.cohort)||matches[0])?.path||null;
  };
  function prepare(data){
    docs=(data.docs||[]).filter(d=>d.subject!=="repository");
    lookup=new Map(docs.map(d=>[norm(d.path),d]));
    adjacency=new Map(docs.map(d=>[d.path,new Set()]));
    const seen=new Set();
    for(const d of docs){
      const targets=[];
      for(const m of (d.content||"").matchAll(/(?<!!)\[\[([^\]\n]+)\]\]/g)) targets.push(m[1]);
      for(const m of (d.content||"").matchAll(/(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)/g)) if(/\.(md|txt)(?:#.*)?$/i.test(m[1]))targets.push(m[1]);
      for(const target of targets){
        const dest=findLink(target,d.path);
        if(!dest||dest===d.path)continue;
        const key=[d.path,dest].sort().join("|");if(seen.has(key))continue;
        seen.add(key);edges.push([d.path,dest]);
        adjacency.get(d.path).add(dest);adjacency.get(dest).add(d.path);
      }
    }
  }
  function refresh(){
    const path=docPath(),current=lookup.get(path);
    let list=docs;
    if(scope==="subject"&&current)list=list.filter(d=>d.subject===current.subject&&d.cohort===current.cohort);
    if(scope==="cohort"&&current)list=list.filter(d=>d.cohort===current.cohort);
    if(scope==="near"&&current){const ids=new Set([path,...adjacency.get(path)]);list=list.filter(d=>ids.has(d.path));}
    if(query){
      const q=query.toLocaleLowerCase("ru");
      const ids=new Set(list.filter(d=>(d.title+" "+d.path).toLocaleLowerCase("ru").includes(q)).flatMap(d=>[d.path,...adjacency.get(d.path)]));
      list=list.filter(d=>ids.has(d.path));
    }
    list=list.slice().sort((a,b)=>(adjacency.get(b.path)?.size||0)-(adjacency.get(a.path)?.size||0)).slice(0,320);
    nodes=list;
    const ids=new Set(nodes.map(d=>d.path));
    shown=edges.filter(e=>ids.has(e[0])&&ids.has(e[1]));
    if(!ids.has(selected))selected=ids.has(path)?path:"";
    const groups=[...new Set(nodes.map(d=>d.cohort+"/"+d.subject))];
    nodes.forEach((d,i)=>{
      if(positions.has(d.path))return;
      const g=groups.indexOf(d.cohort+"/"+d.subject),a=g*6.28318/Math.max(groups.length,1);
      const rank=nodes.slice(0,i).filter(x=>x.subject===d.subject&&x.cohort===d.cohort).length;
      const theta=rank*2.399963,r=25*Math.sqrt(rank+1);
      positions.set(d.path,{x:Math.cos(a)*180+Math.cos(theta)*r,y:Math.sin(a)*180+Math.sin(theta)*r});
    });
    for(let step=0;step<(nodes.length>220?25:55);step++){
      const f=new Map(nodes.map(d=>[d.path,{x:0,y:0}]));
      for(const [a,b] of shown){
        const p=positions.get(a),q=positions.get(b),dx=q.x-p.x,dy=q.y-p.y,l=Math.hypot(dx,dy)||1,v=(l-95)*.014;
        f.get(a).x+=dx/l*v;f.get(a).y+=dy/l*v;f.get(b).x-=dx/l*v;f.get(b).y-=dy/l*v;
      }
      for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
        const a=nodes[i].path,b=nodes[j].path,p=positions.get(a),q=positions.get(b),dx=q.x-p.x,dy=q.y-p.y,r2=dx*dx+dy*dy+1;
        if(r2>25000)continue;
        const v=Math.min(3,1500/r2),l=Math.sqrt(r2);
        f.get(a).x-=dx/l*v;f.get(a).y-=dy/l*v;f.get(b).x+=dx/l*v;f.get(b).y+=dy/l*v;
      }
      for(const d of nodes){const p=positions.get(d.path),v=f.get(d.path);p.x+=Math.max(-6,Math.min(6,v.x));p.y+=Math.max(-6,Math.min(6,v.y));}
    }
    stat.textContent="Граф знаний · "+nodes.length+" статей · "+shown.length+" связей";
    fit();showDetails();
  }
  function resize(){
    const r=canvas.getBoundingClientRect(),dpi=Math.min(devicePixelRatio||1,2);w=Math.max(r.width,1);h=Math.max(r.height,1);
    canvas.width=Math.round(w*dpi);canvas.height=Math.round(h*dpi);ctx.setTransform(dpi,0,0,dpi,0,0);render();
  }
  function fit(){
    if(!nodes.length){scale=1;ox=w/2;oy=h/2;render();return;}
    const ps=nodes.map(d=>positions.get(d.path)),xs=ps.map(p=>p.x),ys=ps.map(p=>p.y);
    const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
    scale=Math.max(.12,Math.min(2,Math.min((w-100)/(xmax-xmin+80),(h-100)/(ymax-ymin+80))));
    ox=w/2-(xmin+xmax)/2*scale;oy=h/2-(ymin+ymax)/2*scale;render();
  }
  const point=(p)=>({x:p.x*scale+ox,y:p.y*scale+oy});
  const world=(x,y)=>({x:(x-ox)/scale,y:(y-oy)/scale});
  const hit=(x,y)=>{const p=world(x,y);for(let i=nodes.length-1;i>=0;i--){const d=nodes[i],v=positions.get(d.path);if(Math.hypot(p.x-v.x,p.y-v.y)<Math.max(10,12/scale))return d.path;}return "";};
  function render(){
    ctx.clearRect(0,0,w,h);
    const css=getComputedStyle(panel),fg=css.getPropertyValue("--graph-text").trim()||"#dae3f0";
    if(!nodes.length){ctx.fillStyle=fg;ctx.font="15px sans-serif";ctx.fillText("Статьи не найдены",25,45);return;}
    for(const [a,b] of shown){
      const p=point(positions.get(a)),q=point(positions.get(b));
      ctx.strokeStyle=selected?(a===selected||b===selected?"#2fc5af":"rgba(124,141,162,.14)"):"rgba(124,141,162,.3)";
      ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();
    }
    const related=adjacency.get(selected)||new Set();
    for(const d of nodes){
      const p=point(positions.get(d.path)),isSelected=d.path===selected,isRelated=related.has(d.path),isHover=d.path===hover;
      const radius=isSelected||isHover?7:5;
      ctx.fillStyle=isSelected?"#889cff":isRelated?"#2fc5af":"#74879e";ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);ctx.fill();
      if(nodes.length<90||scale>1.25||isSelected||isHover){
        ctx.font=(isSelected?"600 ":"")+"12px system-ui";ctx.fillStyle=fg;
        ctx.fillText(d.title.length>38?d.title.slice(0,37)+"…":d.title,p.x+10,p.y+4);
      }
    }
  }
  function showDetails(){
    const d=lookup.get(selected);
    if(!d){detail.textContent="Выберите статью на графе";return;}
    const related=[...(adjacency.get(d.path)||[])].map(p=>lookup.get(p)).filter(Boolean);
    detail.innerHTML="<strong>"+esc(d.title)+"</strong><small>"+related.length+" связей</small><a href='#/doc/"+encodeURIComponent(d.path)+"'>Открыть статью ↗</a>"
      +"<div class='graph-related'>"+related.slice(0,20).map(x=>"<button type='button' data-path='"+esc(x.path)+"'>"+esc(x.title)+"</button>").join("")+"</div>";
  }
  function select(path){selected=path;showDetails();render();}
  function open(value){
    panel.hidden=!value;button.setAttribute("aria-expanded",String(value));
    if(value){scope=docPath()?"subject":"all";root.querySelector(".graph-scope").value=scope;resize();refresh();}
  }
  root.querySelector(".graph-launch").addEventListener("click",()=>open(panel.hidden));
  root.querySelector(".graph-close").addEventListener("click",()=>open(false));
  root.querySelector(".graph-fit").addEventListener("click",fit);
  root.querySelector(".graph-search").addEventListener("input",e=>{query=e.target.value.trim();refresh();});
  root.querySelector(".graph-scope").addEventListener("change",e=>{scope=e.target.value;refresh();});
  detail.addEventListener("click",e=>{const b=e.target.closest("[data-path]");if(b)select(b.dataset.path);});
  window.addEventListener("keydown",e=>{if(e.key==="Escape"&&!panel.hidden)open(false);});
  window.addEventListener("hashchange",()=>{if(!panel.hidden){selected=docPath();refresh();}});
  function coord(e){const r=canvas.getBoundingClientRect();return{x:e.clientX-r.left,y:e.clientY-r.top};}
  canvas.addEventListener("pointerdown",e=>{const p=coord(e);drag={...p,id:hit(p.x,p.y),moved:false};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener("pointermove",e=>{
    const p=coord(e);
    if(drag){
      const dx=p.x-drag.x,dy=p.y-drag.y;if(Math.abs(dx)+Math.abs(dy)>2)drag.moved=true;
      if(drag.id){const v=positions.get(drag.id);v.x+=dx/scale;v.y+=dy/scale;}
      else{ox+=dx;oy+=dy;}
      drag.x=p.x;drag.y=p.y;render();
    } else {hover=hit(p.x,p.y);canvas.style.cursor=hover?"pointer":"grab";render();}
  });
  canvas.addEventListener("pointerup",()=>{if(drag?.id&&!drag.moved)select(drag.id);drag=null;});
  canvas.addEventListener("pointercancel",()=>drag=null);
  canvas.addEventListener("dblclick",e=>{const p=coord(e),id=hit(p.x,p.y);if(id){location.hash="#/doc/"+encodeURIComponent(id);open(false);}});
  canvas.addEventListener("wheel",e=>{e.preventDefault();const p=coord(e),v=world(p.x,p.y);scale=Math.max(.1,Math.min(5,scale*Math.exp(-e.deltaY*.001)));ox=p.x-v.x*scale;oy=p.y-v.y*scale;render();},{passive:false});
  new ResizeObserver(()=>{if(!panel.hidden)resize();}).observe(canvas);
  button.disabled=true;
  fetch("content.json",{cache:"no-cache"}).then(r=>{if(!r.ok)throw Error("HTTP "+r.status);return r.json();})
    .then(data=>{prepare(data);button.disabled=false;})
    .catch(e=>{button.title="Граф недоступен: "+e.message;});
})();