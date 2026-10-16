// Core helpers and theme (tokens: docs/ui.md); charts read the same CSS variables.
// Classic script; all files share one global scope and load in order from index.html.
const $=id=>document.getElementById(id);
// ---------- theme (tokens: docs/DESIGN.md); charts read the same CSS variables ----------
const K={};
function pal(){const cs=getComputedStyle(document.documentElement);
 for(const n of['ok','warn','bad','accent','ink','ink-muted','violet','end','idle','viz-raw','viz-clean','viz-grid','viz-empty','viz-ink','line-strong'])K[n.replace(/-/g,'_')]=cs.getPropertyValue('--'+n).trim();
 if(typeof DC!=='undefined'){Object.assign(DC,{agree:K.ok,leak:K.bad,false_reject:K.warn,false_accept:K.violet,silence:K.end,acoustic_only:K.viz_raw});
  Object.assign(SC,{ACTIVE:K.ok,IDLE:K.idle,MAYBE_AWAY:K.warn,AWAY:K.bad,ENDED:K.end});const l=$('tlLeg');if(l&&typeof legend==='function')legend()}}
function resolveTheme(c){return c==='dark'||(c==='auto'&&matchMedia('(prefers-color-scheme: dark)').matches)?'dark':'light'}
function setTheme(t){if(!['auto','light','dark'].includes(t))t='auto';localStorage.setItem('presence-theme',t);
 document.documentElement.dataset.themeChoice=t;document.documentElement.dataset.theme=resolveTheme(t);
 document.querySelectorAll('#theme button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.t===t)));pal()}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{if((localStorage.getItem('presence-theme')||'auto')==='auto'){document.documentElement.dataset.theme=resolveTheme('auto');pal()}});
