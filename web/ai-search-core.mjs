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
