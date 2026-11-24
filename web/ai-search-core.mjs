export function rankPassages(query,rows,vectors,dimensions,{cohort,subject="",limit=5}={}){
  if(query.length!==dimensions||vectors.length!==rows.length*dimensions||!Array.from(query).every(Number.isFinite))throw new Error("Некорректные размеры индекса");
  const hits=[];
  for(let i=0;i<rows.length;i++){
    const row=rows[i];
    if((cohort&&row.cohort!==cohort)||(subject&&row.subject!==subject))continue;
    let score=0;
    for(let d=0;d<dimensions;d++)score+=query[d]*vectors[i*dimensions+d];
    if(!Number.isFinite(score))throw new Error("Некорректный вектор");
    hits.push({row,score});
  }
  hits.sort((a,b)=>b.score-a.score||a.row.id.localeCompare(b.row.id));
  const seen=new Set();
  return hits.filter(({row})=>{const key=row.cohort+"\0"+row.subject+"\0"+row.text;if(seen.has(key))return false;seen.add(key);return true;}).slice(0,limit);
}
export function sourceHref(path){
  if(typeof path!=="string"||! /^(subjects|cohorts)\//.test(path)||path.split("/").some(p=>p===".."||p==="."))throw new Error("Некорректный источник");
  return "./#/doc/"+encodeURIComponent(path);
}

const STOP=new Set(("что такое как какой какая какие какое каков какова каковы чем чему сколько почему где когда дай дайте назови назовите опиши объясни расскажи пожалуйста определение формулировка является называется в на по из за для от до у о об к ко и а или ли же бы это его ее их он она они мы вы я ты который которая которые которого которых при со с во").split(" "));
const SYMBOLS="∩∪⊂⊆∈∉≠≤≥±×";
const TEX={cap:"∩",cup:"∪",subseteq:"⊆",subsetneq:"⊂",in:"∈",notin:"∉",neq:"≠",leq:"≤",geq:"≥",pm:"±",times:"×"};
export function stem(word){
 if(!/^[а-я]+$/.test(word)||word.length<4)return word;
 const i=word.search(/[аеиоуыэюя]/);if(i<0)return word;
 const prefix=word.slice(0,i+1);let rv=word.slice(i+1),old=rv;
 rv=rv.replace(/(?:ив|ивши|ившись|ыв|ывши|ывшись)$|(?<=[ая])(?:в|вши|вшись)$/,"");
 if(rv===old){
  rv=rv.replace(/(?:ся|сь)$/,"");let before=rv;
  rv=rv.replace(/(?:ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)$/,"");
  if(rv!==before)rv=rv.replace(/(?:ивш|ывш|ующ)$|(?<=[ая])(?:ем|нн|вш|ющ|щ)$/,"");
  else{
   before=rv;
   rv=rv.replace(/(?:ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю)$|(?<=[ая])(?:ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ны|ть|ешь|нно)$/,"");
   if(rv===before)rv=rv.replace(/(?:иями|ями|ами|ией|иям|ием|иях|ев|ов|ие|ье|еи|ии|ей|ой|ий|ям|ем|ам|ом|ах|ях|ию|ью|ия|ья|а|е|и|й|о|у|ы|ь|ю|я)$/,"");
  }
 }
 const result=prefix+rv.replace(/и$/,"");return result.length<3?word:result;
}
export function lexicalTokens(text){
 const clean=text.normalize("NFKC").toLowerCase().replaceAll("ё","е")
  .replace(/https?:\/\/[^\s)>]+/g," ")
  .replace(/\\(?:frac|vec|left|right|mathrm|text|boxed|begin|end|qquad|quad|underline|operatorname)(?![a-zа-я0-9_])/g," ")
  .replace(/\\(cap|cup|subseteq|subsetneq|in|notin|neq|leq|geq|pm|times)(?![a-zа-я0-9_])/g,(_,m)=>TEX[m]);
 return (clean.match(/[a-zа-я]+|\d+(?:[.,]\d+)?|[∩∪⊂⊆∈∉≠≤≥±×]/g)||[])
 .filter(w=>!STOP.has(w)&&(w.length>=2||/^\d$/.test(w)||SYMBOLS.includes(w))).map(stem);
}
export function near(a,b){
 if(a===b||Math.abs(a.length-b.length)>1)return false;
 if(a.length===b.length){
  const diff=[];for(let i=0;i<a.length;i++)if(a[i]!==b[i])diff.push(i);
  return diff.length===1||(diff.length===2&&diff[1]===diff[0]+1&&a[diff[0]]===b[diff[1]]&&a[diff[1]]===b[diff[0]]);
 }
 if(a.length>b.length)[a,b]=[b,a];let i=0;
 while(i<a.length&&a[i]===b[i])i++;
 return a.slice(i)===b.slice(i+1);
}
const FIELDS=[[1,.75],[1.65,.25],[.65,.25]];
export class LexicalIndex{
 constructor(rows){
  this.rows=rows;this.postings=new Map();this.lengths=[];this.average=[0,0,0];this.byLength=new Map();
  for(let i=0;i<rows.length;i++){
   const r=rows[i],fields=[lexicalTokens(r.text),lexicalTokens(r.headings?.at(-1)||""),lexicalTokens(r.title||"")];
   this.lengths.push(fields.map(f=>f.length));
   fields.forEach((terms,f)=>{
    this.average[f]+=terms.length;
    for(const term of terms){
     if(!this.postings.has(term))this.postings.set(term,new Map());
     const p=this.postings.get(term);if(!p.has(i))p.set(i,[0,0,0]);p.get(i)[f]++;
    }
   });
  }
  this.average=this.average.map(x=>Math.max(1,x/Math.max(1,rows.length)));
  for(const term of this.postings.keys())if(term.length>=4&&/^[a-zа-я]+$/.test(term)){
   if(!this.byLength.has(term.length))this.byLength.set(term.length,[]);
   this.byLength.get(term.length).push(term);
  }
 }
 search(question,{cohort,subject="",limit=50}={}){
  const query=[...new Set(lexicalTokens(question))];
  if(!query.length)return [];
  const eligible=new Set(this.rows.flatMap((r,i)=>(!cohort||r.cohort===cohort)&&(!subject||r.subject===subject)?[i]:[]));
  const scoped=new Map();
  const idsFor=term=>{
   if(!scoped.has(term))scoped.set(term,[...(this.postings.get(term)?.keys()||[])].filter(i=>eligible.has(i)));
   return scoped.get(term);
  };
  const exactTerms=query.filter(t=>idsFor(t).length), scores=new Map(),matches=new Map(),corrections=new Map();
  for(const term of query){
   let options=idsFor(term).length?[[term,1]]:[];
   if(!options.length&&term.length>=4&&/^[a-zа-я]+$/.test(term)&&term!=="не"){
    const candidates=[];
    for(const length of [term.length-1,term.length,term.length+1])
     for(const alt of this.byLength.get(length)||[])if(near(term,alt)&&idsFor(alt).length){
      const docs=idsFor(alt);
      // Disambiguate only with another exact term in the same passage.
      const context=exactTerms.reduce((n,t)=>n+Number(docs.some(i=>this.postings.get(t).has(i))),0);
      candidates.push({alt,context});
     }
    candidates.sort((a,b)=>b.context-a.context||a.alt.localeCompare(b.alt));
    if(candidates.length===1||(candidates.length>1&&candidates[0].context>0&&candidates[0].context>candidates[1].context))
     options=[[candidates[0].alt,.65]];
   }
   for(const [alt,factor] of options){
    const ids=idsFor(alt),idf=Math.log1p((eligible.size-ids.length+.5)/(ids.length+.5));
    for(const i of ids){
     const counts=this.postings.get(alt).get(i);
     const tf=counts.reduce((sum,c,f)=>sum+FIELDS[f][0]*c/(1-FIELDS[f][1]+FIELDS[f][1]*this.lengths[i][f]/this.average[f]),0);
     scores.set(i,(scores.get(i)||0)+factor*idf*2.2*tf/(1.2+tf));
     if(!matches.has(i))matches.set(i,new Set());matches.get(i).add(term);
     if(alt!==term){if(!corrections.has(i))corrections.set(i,[]);corrections.get(i).push([term,alt]);}
    }
   }
  }
  const ranked=[...scores].map(([i,score])=>{
   const coverage=matches.get(i).size/query.length;
   return {row:this.rows[i],score:score*coverage,coverage,corrections:corrections.get(i)||[]};
  }).filter(h=>h.coverage>=.5).sort((a,b)=>b.score-a.score||(a.row.source||"").localeCompare(b.row.source||"")||a.row.line_start-b.row.line_start||a.row.id.localeCompare(b.row.id));
  const seen=new Set();
  return ranked.filter(({row})=>{const k=[row.cohort,row.subject,row.text].join("\0");if(seen.has(k))return false;seen.add(k);return true;}).slice(0,limit);
 }
}
export function hybridRank(lexical,dense,limit=5){
 // BM25F beat the learned model in the previous experiment. Preserve its
 // top hit and only rerank comparable lexical candidates (same coverage).
 if(!lexical.length)return dense.slice(0,limit).map(h=>({...h,kind:"semantic-only"}));
 const denseRanks=new Map(dense.map((h,i)=>[h.row.id,i+1]));
 const first={...lexical[0],kind:"lexical"};
 const pool=lexical.slice(1).map((h,i)=>{
  const r=denseRanks.get(h.row.id);
  return {...h,kind:"hybrid",order:i+2,rrf:.9/(20+i+2)+(r?.valueOf()? .1/(20+r):0)};
 });
 // A weak semantic score must not displace a clearly stronger lexical hit.
 // Partition adjacent lexical results into nonoverlapping score bands.
 const sorted=[];
 for(let i=0;i<pool.length;){
  let j=i+1;
  while(j<pool.length&&pool[j].coverage===pool[i].coverage&&pool[j].score>=.9*pool[i].score)j++;
  sorted.push(...pool.slice(i,j).sort((a,b)=>b.rrf-a.rrf||a.order-b.order));i=j;
 }
 return [first,...sorted].slice(0,limit);
}
