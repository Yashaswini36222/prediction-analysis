import type { Horizon, ModelType } from '../api/types';
export interface ParsedCommand { horizon?:Horizon; apps:string[]; categories:string[]; excludeMonths:string[]; excludeApps:string[]; model?:ModelType }
const monthNames=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
export function parseCommand(text:string,knownApps:string[],knownCategories:string[],knownMonths:string[]):ParsedCommand&{unknownApps:string[]}{
  const lower=text.toLowerCase();let horizon:Horizon|undefined;
  if(/\b(12|twelve|year)\b|12\s*months/.test(lower))horizon=12; else if(/\b(6|six|half year)\b|6\s*month/.test(lower)||/half year/.test(lower))horizon=6; else if(/\b(3|three|quarter)\b|3\s*month/.test(lower)||/quarter/.test(lower))horizon=3;
  const apps=knownApps.filter(app=>lower.includes(app.toLowerCase())).filter((v,i,a)=>a.indexOf(v)===i);let unknownApps:string[]=[];
  const appClause=text.match(/(?:for|app(?:lication)?s?)\s+([^.]+)/i)?.[1];if(appClause){const candidates=appClause.split(/,|and|only|exclude|remove/).map(s=>s.trim()).filter(Boolean);candidates.forEach(candidate=>{if(!knownApps.some(app=>candidate.toLowerCase().includes(app.toLowerCase())||app.toLowerCase().includes(candidate.toLowerCase()))&&!/^(?:the|my|all|only)$/i.test(candidate))unknownApps.push(candidate);});}
  const categories=knownCategories.filter(cat=>lower.includes(cat.toLowerCase()));
  let model:ModelType|undefined;if(/\bxgboost\b/.test(lower))model='xgboost';else if(/\blstm\b/.test(lower))model='lstm';else if(/\bensemble\b/.test(lower))model='ensemble';
  const excludeMonths:string[]=[];const excludeMatches=text.matchAll(/(?:exclude|remove|without)\s+([^.]+)/gi);for(const match of excludeMatches){const chunk=match[1];knownMonths.forEach(month=>{if(chunk.toLowerCase().includes(month.toLowerCase())||monthNames.some(name=>chunk.toLowerCase().includes(name.toLowerCase())&&month.toLowerCase().startsWith(name.toLowerCase())))excludeMonths.push(month);});const iso=chunk.match(/20\d{2}-\d{2}/g);if(iso)iso.forEach(v=>{const found=knownMonths.find(m=>m.includes(v));if(found)excludeMonths.push(found)});}
  const excludeApps:string[]=[];knownApps.forEach(app=>{if(new RegExp(`(?:exclude|remove)\\s+(?:app\\s*)?${app.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}`, 'i').test(text))excludeApps.push(app)});
  return {horizon,apps,categories,excludeMonths:[...new Set(excludeMonths)],excludeApps:[...new Set(excludeApps)],model,unknownApps:[...new Set(unknownApps)]};
}
/** Example: parseCommand('Predict 6 months for SAP GRC only Incidents', ['SAP GRC'], ['INC','Incident'], ['Jan_2023'])
 * returns horizon 6, apps ['SAP GRC'], categories ['INC','Incident']. */
