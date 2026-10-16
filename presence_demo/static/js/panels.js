// Side panels: speech recognition, cross-session re-ID card, API-key rows, event log, error banner.
// Classic script; all files share one global scope and load in order from index.html.
const ASR=[];
// 影像字幕：單行、逐字更新（partial 原地覆寫）、新句淡入、final 後停留再淡出。放在 canvas 外，不受鏡像影響
const CAP_HOLD_MS=2500;let capTimer=null,capKey=null;
function caption(m){const el=$('cap');if(!el)return;if(cfg.asr_provider==='none'||!m||!m.text){el.hidden=true;return}
 const key=m.session+':'+m.turn_order;if(key!==capKey){capKey=key;el.classList.remove('is-in');void el.offsetWidth}
 el.hidden=false;$('capTxt').textContent=m.text;el.classList.add('is-in');el.classList.toggle('is-final',m.kind==='final');
 clearTimeout(capTimer);capTimer=setTimeout(()=>el.classList.remove('is-in'),m.kind==='final'?CAP_HOLD_MS:CAP_HOLD_MS*3)}
function captionOff(){const el=$('cap');if(el){el.classList.remove('is-in');el.hidden=true}capKey=null}
function onAsr(m){if(m.kind==='status'&&m.status==='terminated')captionOff();if(m.kind==='status'){$('asrSt').textContent={connected:'（連線中）',begin:'（辨識中'+(m.model?' · '+m.model:'')+'）',terminated:'（已結束）'}[m.status]||'';return}
 if(m.kind==='error'){$('asrSt').textContent='（錯誤：'+m.error+'）';return}
 const key=m.session+':'+m.turn_order;let e=ASR.find(x=>x.key===key);if(!e){e={key};ASR.push(e);if(ASR.length>100)ASR.shift()}Object.assign(e,m);
 caption(m);
 $('asrList').innerHTML=ASR.slice().reverse().map(x=>`<div style="padding:3px 0;border-bottom:1px solid var(--line);${x.kind==='partial'?'color:var(--mut)':''}"><small style="color:var(--mut)">#${x.session} · ${x.start_s!=null?x.start_s.toFixed(1)+'–'+(x.end_s??0).toFixed(1)+' s':''}${x.language?' · '+x.language:''}</small> ${x.text.replace(/[<&]/g,c=>c==='<'?'&lt;':'&amp;')}</div>`).join('')}
// 跨 session 使用者判斷：決策、呼叫數與估計成本；比對縮圖只在「顯示比對照片」開啟時顯示（只在記憶體）
let REID={};const RV=[];const RV_MAX=4;
const RVR={match:'同一人',no_reference:'無前一位資料',below_threshold:'不同人',uncertain:'不確定',error:'呼叫失敗',cap:'已達上限',cooldown:'冷卻中',no_face_timeout:'沒有清楚的臉',decision_timeout:'等待逾時',no_key:'未設定 key',breaker:'暫停中'};
function rvClear(){RV.length=0;renderRV()}
function renderRV(){const card=$('reidCard');card.hidden=!(cfg.reid_enabled&&cfg.reid_show_crops);if(card.hidden){RV.length=0}
 const img=(b,l)=>b?`<img src="data:image/jpeg;base64,${b}" alt="${l}" width="72" height="72">`:`<span class="rv-ph">${l}<br>無</span>`;
 $('reidList').innerHTML=RV.length?RV.map(v=>`<div class="rv-row"><figure>${img(v.probe,'這次')}<figcaption>這次 #${v.session}</figcaption></figure><span class="rv-vs">↔</span><figure>${img(v.ref,'參考')}<figcaption>${v.ref_session!=null?'參考 #'+v.ref_session:'參考'}</figcaption></figure>
  <div class="rv-info"><b class="${v.decision==='resume'?'ok':''}">${v.decision==='resume'?'↩ 接續 #'+v.resumed_from_session:'＋ 新對話'}</b>
  <span>分數 <code>${v.score!=null?v.score.toFixed(2):'—'}</code> / 門檻 <code>${(+v.threshold).toFixed(2)}</code></span>
  <span>${RVR[v.reason]||v.reason} · ${new Date(v.t*1000).toLocaleTimeString()}</span></div></div>`).join(''):'<p class="panel-help">還沒有比對紀錄</p>'}
function onReidView(m){if(m.type==='reid_view_clear')return rvClear();if(!cfg.reid_show_crops)return;RV.unshift(m);RV.length=Math.min(RV.length,RV_MAX);renderRV()}
function onReid(m){const el=$('reidSt');if(m.type==='handoff'){REID.handoff=m}else REID.st=m;
 const st=REID.st||{},h=REID.handoff;if(!cfg.reid_enabled&&!st.calls){el.hidden=true;return}el.hidden=false;
 const dec=st.phase==='await_query'||st.phase==='calling'||st.phase==='await_retry'?'<b>辨識使用者中…</b>':h?(h.decision==='resume'?`<b class="ok">↩ 接續 #${h.resumed_from_session}</b>${h.score!=null?' （'+h.score.toFixed(2)+'）':''}`:`<b>＋ 新對話</b> <small>${({no_reference:'無前一位資料',below_threshold:'不同人',uncertain:'不確定',error:'呼叫失敗',cap:'已達上限',cooldown:'冷卻中',no_face_timeout:'沒有清楚的臉',no_key:'未設定 key',breaker:'暫停中'})[h.reason]||h.reason}</small>`):'—';
 el.innerHTML=`<span>跨 session</span>${dec}${h?` <code>${h.conversation_id.slice(0,8)}</code>`:''}<span class="sp">呼叫 ${st.calls||0} · ${st.tokens||0} tok${st.cost_est?` · 估計 ${st.cost_est}`:''} · 參考臉 ${st.refs||0}${st.refs?`（${Math.round((st.ttl_left||0)/60)} 分）`:''}</span>`;
 if(st.key_set!=null&&$('reidKeySt')&&!/環境/.test($('reidKeySt').textContent))keyBadge('reidKey',{set:st.key_set});renderRV()}
function keyRow(mod,id,label,tip,extra=''){const det=$('mod_'+mod);if(!det)return false;const host=det.querySelector('.mbody')||det;
 host.insertAdjacentHTML('beforeend',`<div class="row keyrow"><div class="top"><label for="${id}In">${label}</label><span id="${id}St" class="kbadge">—</span></div>
  <input type="password" id="${id}In" class="control" autocomplete="off" placeholder="貼上 API key">
  <div class="btns"><button class="button button-secondary button-small" id="${id}Set" type="button">設定</button><button class="button button-ghost button-small" id="${id}Clr" type="button">清除 key</button>${extra}</div>
  <div class="tip">${tip}</div></div>`);return true}
function keyBadge(id,j){const e=$(id+'St');if(!e)return;e.textContent=j.set?('已設定'+(j.source==='env'?'（環境變數）':'')):'未設定';e.classList.toggle('is-set',!!j.set)}
function wireKey(id,api){const post=k=>fetch('/api/'+api+'_key'+location.search,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:k})}).then(r=>r.json()).then(j=>keyBadge(id,j));
 $(id+'Set').onclick=()=>{const v=$(id+'In').value;$(id+'In').value='';post(v).then(()=>toast('已設定 API key'))};$(id+'Clr').onclick=()=>post('');
 fetch('/api/'+api+'_status'+location.search).then(r=>r.json()).then(j=>keyBadge(id,j)).catch(()=>{})}
function addReidKey(){if(!keyRow('reid','reidKey','雲端 API key','key 只存在伺服器記憶體（重啟就消失），不寫檔、不寫 log、不會傳回瀏覽器。也可以設定環境變數 TYPESAFE_API_KEY。「清除參考臉」會立即刪除記憶體中保存的所有臉部裁切和這裡顯示的比對縮圖。',
  '<button class="button button-ghost button-small" id="reidWipe" type="button">清除參考臉</button>'))return;
 wireKey('reidKey','reid');
 $('reidWipe').onclick=()=>{if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'reid_wipe'}));rvClear();toast('已清除參考臉')}}
function addAsrKey(){if(!keyRow('asr','asrKey','AssemblyAI API key','key 只存在伺服器記憶體（重啟就消失），不寫檔、不寫 log、不會傳回瀏覽器，這裡只顯示「已設定」。也可以在伺服器設定環境變數 ASSEMBLYAI_API_KEY。'))return;
 wireKey('asrKey','asr')}
$('reidViewClr').onclick=rvClear;

const EVL=[];function addEv(s){EVL.push(s);if(EVL.length>200)EVL.shift();const ul=document.getElementById('evList');const li=document.createElement('li');li.className=s.level||'info';
const kv=Object.entries(s).filter(([k])=>!['type','level','event','ts','t'].includes(k)).map(([k,v])=>k+'='+(typeof v==='object'?JSON.stringify(v):v)).join(' ');
li.textContent=new Date().toLocaleTimeString()+' '+(s.event||s.type)+' '+kv;ul.prepend(li);while(ul.children.length>200)ul.lastChild.remove();document.getElementById('evCount').textContent=EVL.length;document.getElementById('navEvCount').textContent=EVL.length}
function showErr(m){document.getElementById('errText').textContent='⚠ '+m+'（已自動恢復，詳見事件紀錄）';document.getElementById('errBanner').style.display='block'}
document.getElementById('errClose').onclick=()=>document.getElementById('errBanner').style.display='none';
