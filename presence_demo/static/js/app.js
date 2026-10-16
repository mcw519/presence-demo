// WebSocket connection, rendering (state machine, overlay, ROI editing), media capture, resource monitor.
// Classic script; all files share one global scope and load in order from index.html.
// ---------- websocket ----------
let ws,last=null,aMeta=null;
// 瀏覽器端事件回報（server log：event=client）＋連線存活檢查：半開的連線（例如主機休眠後）會被主動關閉並重連
let wsTries=0,wsDownAt=0,lastRx=performance.now(),REPQ=[];
function report(what,info={}){const m={type:'client',what,...info};if(ws&&ws.readyState===1)ws.send(JSON.stringify(m));else if(REPQ.length<20)REPQ.push(m)}
setInterval(()=>{if(ws&&ws.readyState===1&&performance.now()-lastRx>8000){REPQ.push({type:'client',what:'stale_socket',silent_s:Math.round((performance.now()-lastRx)/1000)});lastRx=performance.now();try{ws.close()}catch(e){}}},2000);
let wallPrev=Date.now();setInterval(()=>{const n=Date.now(),gap=(n-wallPrev)/1000;wallPrev=n;if(gap>5)report('client_suspend',{gap_s:Math.round(gap)})},1000);
function connect(){ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws'+location.search);ws.binaryType='arraybuffer';
 ws.onopen=()=>{lastRx=performance.now();if(wsTries)report('reconnect',{tries:wsTries,down_s:Math.round((performance.now()-wsDownAt)/1000),running});wsTries=0;REPQ.splice(0).forEach(m=>ws.send(JSON.stringify(m)));if(rec)setTimeout(()=>wsSend({type:'record',on:true}),0);$('conn').classList.add('is-ok');$('conn').classList.remove('is-err');$('connTxt').textContent='已連線';sendCfg();if(running)ws.send(JSON.stringify({type:'start'}))};
 ws.onclose=()=>{if(!wsTries)wsDownAt=performance.now();wsTries++;$('conn').classList.remove('is-ok');$('conn').classList.add('is-err');$('connTxt').textContent='未連線・重新連線中';setTimeout(connect,1500)};
 ws.onmessage=e=>{lastRx=performance.now();if(typeof e.data!=='string')return onClean(e.data);const s=JSON.parse(e.data);
  if(s.type==='event')return addEv(s);
  if(s.type==='error'){addEv(Object.assign({level:'error'},s));return showErr((s.where?s.where+': ':'')+(s.error||s.message||'error'))}
  if(s.type==='asr')return onAsr(s);
  if(s.type==='reid_view'||s.type==='reid_view_clear')return onReidView(s);
  if(s.type==='handoff'||s.type==='reid')return onReid(s);
  if(rec&&(s.type==='audio'||s.type==='state'))recMeta(s);
  if(s.type==='audio'){aMeta=s;if(s.available!==false)TLH.push({t:performance.now(),raw:s.raw_vad,clean:s.clean_vad,sp:s.speech,dec:s.decision,st:s.state,p:s.p_group});while(TLH.length&&performance.now()-TLH[0].t>31000)TLH.shift();return renderAudio()}
  if(s.type==='stats')return renderStats(s);
  if(s.type==='pong'){lastRtt=performance.now()-s.t;return}
  if(s.type==='state'&&(s.recommended_fps||0)!==recFps){recFps=s.recommended_fps||0;if(cfg.adaptive_fps)restartFrames()}
  if(s.config_ack&&s.config){let ch=false;for(const k in s.config){if(!(k in DEF))continue;const v=BOXES.includes(k)?s.config[k].map(x=>+(+x).toFixed(4)):s.config[k];
    if(JSON.stringify(v)!==JSON.stringify(cfg[k])){cfg[k]=v;ch=true}}if(ch){save();syncUI()}
   if(s.warnings&&s.warnings.length)toast('伺服器調整：'+s.warnings.join('；'))}
  if(!('detections' in s)&&last)s.detections=last.detections;last=s;render(s)}}
// ---------- render ----------
const STATES=['ACTIVE','IDLE','MAYBE_AWAY','AWAY','ENDED'];
const STZH={ACTIVE:'對話中',IDLE:'閒置',MAYBE_AWAY:'可能離開',AWAY:'離開',ENDED:'結束'};
$('sm').innerHTML=STATES.map((x,i)=>`<div class="fsm-node" id="st_${x}" data-s="${x}">${i?`<i class="fsm-edge" id="se_${i}"></i>`:''}<b></b><span>${x.replace('_',' ')}</span><small>${STZH[x]}</small></div>`).join('');
// 軸上節點：目前狀態高亮，最近一次轉換的來源節點與經過的線段標示出來
function renderFsm(s){const trs=s.log.filter(e=>e.event==='transition'),tr=trs[trs.length-1];
 const a=tr?STATES.indexOf(tr.frm):-1,b=STATES.indexOf(s.state);
 STATES.forEach((x,i)=>{$('st_'+x).className='fsm-node '+x+(x===s.state?' is-current':'')+(tr&&i===a&&x!==s.state?' is-from':'');
  if(i)$('se_'+i).className='fsm-edge'+(a>=0&&b>=0&&i>Math.min(a,b)&&i<=Math.max(a,b)?(b<a?' is-path back':' is-path'):'')});
 $('smHist').innerHTML=trs.slice(-6).map((e,i,arr)=>`<span class="fsm-tr${i===arr.length-1?' is-last':''}" title="${e.reason}"><i class="dot ${e.to}"></i><em>${e.t.toFixed(1)}s</em>${e.frm.replace('_',' ')} → ${e.to.replace('_',' ')}<small>${e.reason}</small></span>`).join('')||'<span class="quiet">尚無轉換</span>'}
const pct=(a,b)=>Math.max(0,Math.min(100,b>0?a/b*100:0))+'%';
const PH=[];
function drawP(s){const card=$('pCard');if(s.presence_model!=='hmm'){card.style.display='none';return}card.style.display='';
 if(s.p_group!=null){PH.push([performance.now(),s.p_group]);while(PH.length&&performance.now()-PH[0][0]>120000)PH.shift()}
 $('pVal').textContent=s.p_group!=null?`P = ${s.p_group.toFixed(2)}（影像 ${s.p_tracks!=null?s.p_tracks.toFixed(2):'—'}）`:'—';
 $('pHint').textContent=`群組 P ≥ ${cfg.p_active} 回到 ACTIVE；< ${cfg.p_maybe} 進入 MAYBE_AWAY；< ${cfg.p_away} 持續 ${cfg.away_hold_seconds}s 判定 AWAY`;
 const c=$('pSpark'),d=devicePixelRatio||1,r=c.getBoundingClientRect();c.width=r.width*d;c.height=r.height*d;const g=c.getContext('2d'),W=c.width,H=c.height;
 g.clearRect(0,0,W,H);[[cfg.p_active,K.ok],[cfg.p_maybe,K.warn],[cfg.p_away,K.bad]].forEach(([v,col])=>{g.strokeStyle=col+'aa';g.setLineDash([4*d,4*d]);g.beginPath();g.moveTo(0,H-v*H);g.lineTo(W,H-v*H);g.stroke()});g.setLineDash([]);
 if(PH.length>1){const t0=performance.now()-120000;g.strokeStyle=K.accent;g.lineWidth=2*d;g.beginPath();PH.forEach(([t,v],i)=>{const x=(t-t0)/120000*W,y=H-v*H;i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()}}
function render(s){
 drawP(s);
 renderFsm(s);
 const tr=[...s.log].reverse().find(e=>e.event==='transition');$('reason').textContent=tr?`${tr.frm} → ${tr.to}：${tr.reason}`:'—';
 $('pipe').textContent=s.pipeline?'模型：'+s.pipeline:'';
 $('spk').textContent=s.speaking?'🎙 是':'否';$('sil').textContent=s.silence.toFixed(1)+'s';
 $('roiSt').textContent=s.present?'有人':s.empty?'空':'不確定';$('sess').textContent='#'+s.session;
 $('b1').style.width=pct(s.silence,cfg.idle_after);$('b1t').textContent=`${Math.min(s.silence,cfg.idle_after).toFixed(1)} / ${cfg.idle_after}s`;
 const pm={vision:'影像優先',balanced:'平衡',audio:'聲音優先'}[s.priority_mode]||'';
 const mode=s.path==='ended'?'未進行':s.priority_mode==='vision'?(s.path==='present'?'有人：靜默不會離開':s.path==='empty'?'ROI 空：只看影像':'影像不確定：看聲音')
  :s.present?'有人':s.empty?'ROI 空':'只看聲音';
 $('b2l').textContent=`逾時 → AWAY · ${pm} · ${mode}`;
 if(!s.timeout){$('b2t').textContent=`${s.elapsed.toFixed(1)}s / ∞`;s={...s,timeout:1e9}}else $('b2t').textContent=`${s.elapsed.toFixed(1)} / ${s.timeout}s`;
 const f=s.elapsed/s.timeout;$('b2').style.width=pct(s.elapsed,s.timeout);$('b2').style.background=f>=1?'var(--bad)':f>=cfg.maybe_away_frac?'var(--warn)':'var(--ok)';
 $('b2m').style.left=cfg.maybe_away_frac*100+'%';
 const a=s.state==='AWAY'?s.in_state:0;$('b3').style.width=pct(a,cfg.end_after_away);$('b3t').textContent=s.state==='AWAY'?`${a.toFixed(1)} / ${cfg.end_after_away}s`:'—';
 $('people').innerHTML=s.tracks.length?s.tracks.map(t=>`<div class="person"><b style="color:${col(t)}">#${t.id}</b>
  <span class="badge">${t.participant?'★ 參與者':s.intent_model==='score'?'旁觀者':'候選'}</span><span class="sp">${t.visible_kind==='predicted'?`預測中 vis ${(t.vis_conf??0).toFixed(2)}`:t.visible?'可見':`消失 ${t.missing}s`}${t.dwell_seconds!=null?` · 停留 ${t.dwell_seconds}s`:''}${t.velocity?` · 速度 ${Math.hypot(...t.velocity).toFixed(2)}/s`:''}${t.has_face===false?' · 無臉(身體)':''}${t.facing?' · '+({facing:'正面',side:'側身',away:'背對'}[t.facing]):''}${t.distance!=null?` · ${t.distance} m ${TR[t.trend]||''}`:''}</span>
  <small>相似 ${t.similarity.toFixed(2)} · conf ${t.conf.toFixed(2)}</small>${t.p!=null?`<div class="track" title="此人在場機率 P" style="margin-top:4px;height:6px"><div class="fill" style="width:${(t.p*100).toFixed(0)}%;background:${t.p>=cfg.p_active?'var(--ok)':t.p>=cfg.p_away?'var(--warn)':'var(--bad)'}"></div></div><small>P ${t.p.toFixed(2)}</small>`:''}${t.intent!=null?`<div class="track" title="互動意圖（0–1）。白線＝加入門檻" style="margin-top:4px;height:6px;position:relative"><div class="fill" style="width:${(t.intent*100).toFixed(0)}%;background:${t.intent>=cfg.join_threshold?'var(--ok)':t.intent>=cfg.drop_threshold?'var(--idle)':'var(--mut)'}"></div><div class="mark" style="left:${cfg.join_threshold*100}%"></div></div><small title="${t.intent_cues?Object.entries(t.intent_cues).map(([k,v])=>k+' '+(v==null?'—':v)).join(' · '):''}">意圖 ${t.intent.toFixed(2)}</small>`:''}</div>`).join(''):'<div class="empty">目前沒有追蹤對象</div>';
 $('log').innerHTML=[...s.log].reverse().map(e=>`<div><span style="color:var(--mut)">${e.t.toFixed(1)}</span> <span class="ev ${e.event}">${e.event}</span> ${e.frm?e.frm+'→'+e.to:''} ${e.track?'#'+e.track:''} <span style="color:var(--mut)">${e.reason}</span></div>`).join('');
 draw();
}
const col=t=>!t.visible?'#7c7f92':t.participant?'#3ecf8e':'#f2b54a';  // dark-theme good / warn / ink-3
// ---------- overlay & ROI drag ----------
const ROICOL={face_roi:'#6fd3db',pose_roi:'#bdbbff'};  // mic / periwinkle
const TR={approaching:'↘ 靠近',receding:'↗ 遠離',steady:''};
const ZC={in:'#3ecf8e',near:'#f2b54a',far:'#f2b54a',left:'#ff7a6e'};
const SK=[['l_shoulder','r_shoulder'],['l_shoulder','l_hip'],['r_shoulder','r_hip'],['l_hip','r_hip'],['nose','l_eye'],['nose','r_eye'],['l_eye','l_ear'],['r_eye','r_ear']];
let roiEdit='face_roi';
// 正在編輯的 ROI 一定顯示（即使姿勢偵測沒開），其他照「顯示」設定
const visBoxes=()=>BOXES.filter(k=>k===roiEdit||(k==='face_roi'?cfg.show_face_roi:(cfg.pose_enabled&&cfg.show_pose_roi)));
const ROINAME={face_roi:'臉部',pose_roi:'身體'},ROIDEF={face_roi:[0.2,0.05,0.8,0.95],pose_roi:[0,0,1,1]};
function roiInfo(){$('roiVal').textContent=roiEdit?`${ROINAME[roiEdit]} [${cfg[roiEdit].map(v=>(+v).toFixed(2)).join(', ')}]${roiEdit==='pose_roi'&&!cfg.pose_enabled?' · 姿勢偵測未開啟':''}`:'';$('roiReset').disabled=!roiEdit}
$('roiPick').addEventListener('click',e=>{const b=e.target.closest('button[data-roi]');if(!b)return;roiEdit=b.dataset.roi||null;
 $('roiPick').querySelectorAll('button').forEach(x=>{const on=x===b;x.classList.toggle('is-active',on);x.setAttribute('aria-checked',on)});roiInfo();draw()});
$('roiReset').addEventListener('click',()=>{if(!roiEdit)return;cfg[roiEdit]=[...ROIDEF[roiEdit]];syncUI();save();sendCfg();roiInfo();draw()});
// 純函式（測試用）：畫面座標 ↔ 原始相機座標。鏡像時 x 翻轉，矩形左右邊互換
const roiMath={toRaw:(x,m)=>m?1-x:x,boxToView:(b,m)=>m?[1-b[2],b[1],1-b[0],b[3]]:[...b],
 norm:(a,b,min=0.05)=>{const C=v=>Math.min(1,Math.max(0,v));let r=[C(Math.min(a[0],b[0])),C(Math.min(a[1],b[1])),C(Math.max(a[0],b[0])),C(Math.max(a[1],b[1]))];
  if(r[2]-r[0]<min){r[2]=Math.min(1,r[0]+min);r[0]=r[2]-min}if(r[3]-r[1]<min){r[3]=Math.min(1,r[1]+min);r[1]=r[3]-min}return r.map(v=>+v.toFixed(3))}};
const cv=$('c'),g=cv.getContext('2d');let drag=null;
function fit(){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1;cv.width=r.width*d;cv.height=r.height*d;draw()}
addEventListener('resize',fit);
// 鏡像只在繪圖時換算 x 座標（文字不會反過來）；cfg/伺服器的座標一律是原始相機座標
const MIR=()=>!!cfg.mirror_view,mx=x=>MIR()?1-x:x,mbox=b=>MIR()?[1-b[2],b[1],1-b[0],b[3],...b.slice(4)]:b;
function draw(){const W=cv.width,H=cv.height,d=devicePixelRatio||1;g.clearRect(0,0,W,H);$('v').classList.toggle('mirror',MIR());
 (last?.detections||[]).forEach(x=>{const xb=mbox(x.bbox);g.strokeStyle='#ffffff40';g.lineWidth=1*d;g.strokeRect(xb[0]*W,xb[1]*H,(xb[2]-xb[0])*W,(xb[3]-xb[1])*H)});//[0]*W,x.bbox[1]*H,(x.bbox[2]-x.bbox[0])*W,(x.bbox[3]-x.bbox[1])*H)});
 (last?.tracks||[]).forEach(t=>{const c=col(t);
  if(t.body){const bb=mbox(t.body);g.strokeStyle=c+'99';g.lineWidth=1.5*d;g.setLineDash([4*d,4*d]);g.strokeRect(bb[0]*W,bb[1]*H,(bb[2]-bb[0])*W,(bb[3]-bb[1])*H);g.setLineDash([])}
  if(t.keypoints){const K=t.keypoints;g.strokeStyle='#ffffffcc';g.lineWidth=2*d;
   SK.forEach(([a,b])=>{const p=K[a],q=K[b];if(p&&q&&p[2]>0.4&&q[2]>0.4){g.beginPath();g.moveTo(mx(p[0])*W,p[1]*H);g.lineTo(mx(q[0])*W,q[1]*H);g.stroke()}});
   g.fillStyle=c;Object.values(K).forEach(p=>{if(p[2]>0.4){g.beginPath();g.arc(mx(p[0])*W,p[1]*H,3*d,0,7);g.fill()}})}
  const b=mbox(t.bbox);g.strokeStyle=g.fillStyle=c;g.lineWidth=(t.has_face===false?1.5:3)*d;
  const pred=t.visible_kind==='predicted';if(pred){g.setLineDash([6*d,5*d]);g.lineWidth=2*d}
  g.strokeRect(b[0]*W,b[1]*H,(b[2]-b[0])*W,(b[3]-b[1])*H);g.setLineDash([]);g.font=`${13*d}px sans-serif`;
  let lab=`#${t.id}${t.participant?' ★':''} ${pred?`預測 ${(t.vis_conf??0).toFixed(2)}`:t.has_face===false?'身體':t.similarity.toFixed(2)}`;
  if(t.distance!=null)lab+=` · ${t.distance}m${t.trend==='receding'?'↗':t.trend==='approaching'?'↘':''}`;
  const tw=g.measureText(lab).width;g.fillStyle=t.zone?ZC[t.zone]:c;
  g.fillRect(b[0]*W,b[1]*H-18*d,tw+8*d,18*d);g.fillStyle='#000';g.fillText(lab,b[0]*W+4*d,b[1]*H-5*d)});
 const vb=visBoxes();
 if(vb.includes('face_roi')){const r=mbox(cfg.face_roi),x=r[0]*W,y=r[1]*H,w=(r[2]-r[0])*W,h=(r[3]-r[1])*H;
  g.fillStyle='#0006';g.fillRect(0,0,W,y);g.fillRect(0,y+h,W,H-y-h);g.fillRect(0,y,x,h);g.fillRect(x+w,y,W-x-w,h)}
 vb.forEach(k=>{const r=mbox(cfg[k]),x=r[0]*W+(k==='pose_roi'?2*d:0),y=r[1]*H+(k==='pose_roi'?2*d:0),w=(r[2]-r[0])*W-(k==='pose_roi'?4*d:0),h=(r[3]-r[1])*H-(k==='pose_roi'?4*d:0);
  const ed=k===roiEdit;g.strokeStyle=ROICOL[k];g.lineWidth=(ed?3:1.5)*d;g.globalAlpha=ed?1:0.6;g.setLineDash(k==='pose_roi'?[3*d,5*d]:[8*d,6*d]);g.strokeRect(x,y,w,h);g.setLineDash([]);
  g.fillStyle=ROICOL[k];if(ed)[[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([a,b])=>g.fillRect(a-6*d,b-6*d,12*d,12*d));g.globalAlpha=1;
  g.font=`${11*d}px sans-serif`;g.fillText(k==='face_roi'?'臉部 ROI':'身體 ROI',x+8*d,y+(k==='face_roi'?16:30)*d)});
 if(cfg.distance_enabled){g.font=`${12*d}px sans-serif`;const txt=`對話區 ${cfg.zone_min}–${cfg.zone_max} m · 離開 > ${cfg.leave_distance} m`;
  const tw=g.measureText(txt).width;g.fillStyle='#000a';g.fillRect(W-tw-16*d,H-24*d,tw+12*d,20*d);g.fillStyle='#ececf1';g.fillText(txt,W-tw-10*d,H-10*d)}
}
function pos(e){const r=cv.getBoundingClientRect(),C=v=>Math.min(1,Math.max(0,v));return[C(mx((e.clientX-r.left)/r.width)),C((e.clientY-r.top)/r.height),r]}
// 只有選取中的 ROI 可以拖曳；框外按下＝畫新框（座標都在原始相機座標系處理）
function hit(e){if(!roiEdit)return null;const[px,py,rc]=pos(e),tx=18/rc.width,ty=18/rc.height,k=roiEdit,r=cfg[k];
 for(const[i,j]of[[0,1],[2,1],[0,3],[2,3]])if(Math.abs(px-r[i])<tx&&Math.abs(py-r[j])<ty)return{mode:'corner',i,j,k};
 if(px>r[0]&&px<r[2]&&py>r[1]&&py<r[3])return{mode:'move',k};return{mode:'new',k}}
cv.addEventListener('pointermove',e=>{if(!drag){const h=hit(e);cv.style.cursor=h?({move:'move',corner:'nwse-resize',new:'crosshair'}[h.mode]):'default';return}
 const[px,py]=pos(e),r=[...drag.r],C=v=>Math.min(1,Math.max(0,v));
 if(drag.mode==='new')cfg[drag.k]=roiMath.norm(drag.p,[px,py]);
 else if(drag.mode==='move'){const dx=px-drag.p[0],dy=py-drag.p[1],w=r[2]-r[0],h=r[3]-r[1];
  const x0=Math.min(1-w,Math.max(0,r[0]+dx)),y0=Math.min(1-h,Math.max(0,r[1]+dy));cfg[drag.k]=[x0,y0,x0+w,y0+h]}
 else{r[drag.i]=C(px);r[drag.j]=C(py);
  if(r[2]-r[0]<0.05){if(drag.i===0)r[0]=r[2]-0.05;else r[2]=r[0]+0.05}if(r[3]-r[1]<0.05){if(drag.j===1)r[1]=r[3]-0.05;else r[3]=r[1]+0.05}cfg[drag.k]=r}
 cfg[drag.k]=cfg[drag.k].map(v=>+v.toFixed(3));syncUI();roiInfo();draw()});
cv.addEventListener('pointerdown',e=>{const h=hit(e);if(!h)return;drag={...h,p:pos(e),r:[...cfg[h.k]]};cv.setPointerCapture(e.pointerId);e.preventDefault()});
const endDrag=()=>{if(drag){const k=drag.k;if(drag.mode==='new'){const r=cfg[k];if(r[2]-r[0]<0.051&&r[3]-r[1]<0.051)cfg[k]=drag.r}drag=null;syncUI();roiInfo();save();sendCfg();draw()}};
roiInfo();cv.addEventListener('pointerup',endDrag);cv.addEventListener('pointercancel',endDrag);
// ---------- media: VAD + frames, start/stop ----------
let speech=false,frameTimer,vadTimer,pingTimer,stream=null,ac=null,running=false,autoPaused=false,lastRtt=null,v=$('v'),off=document.createElement('canvas'),og=off.getContext('2d');
const wsSend=o=>{if(ws&&ws.readyState===1)ws.send(JSON.stringify(o))};
function restartFrames(){clearInterval(frameTimer);if(!running)return;
 frameTimer=setInterval(()=>{if(!ws||ws.readyState!==1||!v.videoWidth)return;const s=480/v.videoHeight;off.width=Math.round(v.videoWidth*s);off.height=480;
  og.drawImage(v,0,0,off.width,off.height);wsSend({type:'frame',speech,t:performance.now(),data:off.toDataURL('image/jpeg',0.7)})},1000/curFps())}
const curFps=()=>cfg.adaptive_fps&&recFps>0?recFps:cfg.fps;let recFps=0;
function setUI(){$('start').textContent=running?'停止':'開始';$('start').className=running?'button button-danger':'button button-primary';
 $('cam').textContent=running?'鏡頭/麥克風 開':autoPaused?'已自動暫停':'鏡頭/麥克風 關';$('cam').classList.toggle('is-live',running);
 $('placeholder').style.display=running?'none':'grid';$('monHint').textContent=running?'':'（開始後每秒更新）'}
async function start(){
 if(running)return;
 stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:'user'},audio:{echoCancellation:true,noiseSuppression:true}});
 v.srcObject=stream;running=true;autoPaused=false;setUI();wsSend({type:'start'});
 stream.getTracks().forEach(t=>t.onended=()=>{report('media_ended',{kind:t.kind});if(running){stop(true);setTimeout(()=>start().then(()=>report('media_resumed',{kind:t.kind})).catch(e=>toast('鏡頭/麥克風中斷，請按「開始」：'+e.message)),1000)}});
 ac=new (window.AudioContext||window.webkitAudioContext)({latencyHint:'interactive'});await ac.resume();
 const an=ac.createAnalyser();an.fftSize=1024;ac.createMediaStreamSource(stream).connect(an);
 const buf=new Float32Array(an.fftSize);let floor=0.01,until=0,prev=false;
 vadTimer=setInterval(()=>{an.getFloatTimeDomainData(buf);let e=0;for(const x of buf)e+=x*x;const rms=Math.sqrt(e/buf.length),now=performance.now();
  const thr=Math.max(floor*cfg.vad_k,cfg.vad_min),voiced=rms>thr;
  floor=voiced?0.999*floor+0.001*rms:0.95*floor+0.05*rms;floor=Math.max(floor,1e-4);
  if(voiced)until=now+cfg.vad_hang;speech=now<until;
  const sc=x=>Math.min(100,Math.sqrt(x/0.2)*100)+'%';$('lvl').style.width=sc(rms);$('lvl').style.background=speech?'var(--ok)':'var(--idle)';
  $('thr').style.left=sc(thr);$('flo').style.left=sc(floor);
  if(speech!==prev)wsSend({type:'vad',speech});prev=speech},50);
 pingTimer=setInterval(()=>wsSend({type:'ping',t:performance.now(),rtt:lastRtt}),2000);
 await startAudio(ac).catch(e=>toast('音訊串流無法啟動：'+e.message));
 restartFrames();
}
function stop(auto){
 if(rec)recStop();
 [frameTimer,vadTimer,pingTimer].forEach(clearInterval);
 if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;v.srcObject=null;
 if(ac){ac.close().catch(()=>{});ac=null}player=rawMon=cleanMon=anRaw=anClean=null;
 speech=false;$('lvl').style.width='0';
 if(running)wsSend({type:'stop'});running=false;autoPaused=!!auto;setUI();last=null;draw();
}
$('start').onclick=()=>running?(stop(),toast('已停止：鏡頭與麥克風已關閉')):start().catch(e=>{stop();toast('無法開啟鏡頭/麥克風：'+e.message+'（需 HTTPS 或 localhost）')});
document.addEventListener('visibilitychange',()=>{
 report('visibility',{hidden:document.hidden,running,auto_paused:autoPaused});
 if(document.hidden&&running){report('autopause');stop(true)}
 else if(!document.hidden&&autoPaused){start().then(()=>toast('已自動恢復')).catch(()=>toast('分頁切回來了，按「開始」繼續'))}});
// ---------- resource monitor ----------
const H={cpu:[],sys:[],lat:[],p95:[]},HN=60,push=(a,x)=>{a.push(x);if(a.length>HN)a.shift()};
const fmtB=b=>b>=1<<30?(b/(1<<30)).toFixed(1)+' GB':b>=1<<20?(b/(1<<20)).toFixed(0)+' MB':(b/1024).toFixed(0)+' KB';
function spark(id,series,colors,maxMin){const c=$(id),d=devicePixelRatio||1,r=c.getBoundingClientRect();c.width=r.width*d;c.height=r.height*d;
 const x=c.getContext('2d'),W=c.width,Hh=c.height;const mx=Math.max(maxMin,...series.flat())*1.1||1;
 x.strokeStyle=K.viz_grid;x.beginPath();x.moveTo(0,Hh/2);x.lineTo(W,Hh/2);x.stroke();
 series.forEach((a,i)=>{if(a.length<2)return;x.beginPath();a.forEach((y,j)=>{const px=W*(j+HN-a.length)/(HN-1),py=Hh-y/mx*(Hh-2)-1;j?x.lineTo(px,py):x.moveTo(px,py)});
  x.strokeStyle=colors[i];x.lineWidth=1.5*d;x.stroke();
  if(i===0){x.lineTo(W,Hh);x.lineTo(W*(HN-a.length)/(HN-1),Hh);x.closePath();x.fillStyle=colors[i]+'22';x.fill()}})}
function renderStats(s){const y=s.system,c=s.conn,L=c.latency_ms,i=s.info;
 push(H.cpu,y.proc_cpu);push(H.sys,y.sys_cpu);push(H.lat,L.total.avg);push(H.p95,L.total.p95);
 $('mCpu').textContent=`${y.proc_cpu}% / ${y.sys_cpu}%`;$('mLat').textContent=`${L.total.avg} / ${L.total.p95} ms`;
 $('mMem').textContent=`${fmtB(y.proc_rss)} / ${fmtB(y.mem_used)} (${y.mem_pct}%)`;
 $('mFps').textContent=`${c.fps} / ${fmtB(c.bandwidth_Bps)}/s`;
 spark('spCpu',[H.cpu,H.sys],[K.accent,K.viz_raw],100);spark('spLat',[H.lat,H.p95],[K.ok,K.warn],10);
 $('mCores').innerHTML=(y.per_core||[]).map(p=>`<i style="height:${Math.max(4,p)}%" title="${p}%"></i>`).join('');
 const nm={pose:'姿勢',motion:'移動偵測',decode:'JPEG 解碼',detect:'偵測',embed:'Embedding',tracker:'Tracker',total:'總計'};
 $('mLatT').innerHTML=Object.entries(L).map(([k,v])=>`<tr><td>${nm[k]||k}</td><td class="num">${v.last}</td><td class="num">${v.avg}</td><td class="num">${v.p95}</td></tr>`).join('');
 const kv={'Load avg':y.load?y.load.join(' / '):'—','畫面張數':`收 ${c.frames.received} · 處理 ${c.frames.processed} · 丟棄 ${c.frames.dropped}`,
  '畫面大小':`${c.frame_size.join('×')} · 平均 ${fmtB(c.frame_bytes_avg)}`,'WS 往返':c.rtt_ms!=null?c.rtt_ms+' ms':(lastRtt?lastRtt.toFixed(0)+' ms':'—'),
  '模型':i.pipeline,'Embedding':i.embed_backend?`${i.embed_backend.active}（要求 ${i.embed_backend.requested}）· ${i.embed_backend.model} · ${i.embed_backend.accelerator}${i.embed_backend.note?' · '+i.embed_backend.note:''}`:'—','OpenCV':`${i.opencv} · ${i.dnn_backend} / ${i.dnn_target} · ${i.cv_threads} threads`,'程序':`${y.proc_threads} threads · ${i.cpu_count} cores`,
  '偵測率':c.optim&&c.optim.detect_calls!=null?`有臉 ${c.optim.detect_hits} / 偵測 ${c.optim.detect_calls} 次（${c.optim.detect_calls?Math.round(100*c.optim.detect_hits/c.optim.detect_calls):0}%）`:'—',
  '效能':c.optim?`跳過 ${c.optim.skipped_pct}%（${c.optim.skipped} 張）· embedding 計算 ${c.optim.embed_computed} / 重用 ${c.optim.embed_reused}（${c.optim.embed_reused_pct}%）`:'—',
  '姿勢':`${i.pose_model||'—'} · 呼叫 ${c.optim?c.optim.pose_calls:0} 次 · avg ${L.pose?L.pose.avg:0} ms`,
  '人聲分離':aMeta&&aMeta.available?`RTF ${aMeta.rtf} · 每塊 ${aMeta.proc_ms} ms · 約 ${(aMeta.rtf*100).toFixed(0)}% 單核 · 延遲 ${aMeta.algo_latency_ms}+${lastLat!=null?lastLat.toFixed(0):'—'} ms`:'未啟用',
  '送幀 fps':`${curFps()}${cfg.adaptive_fps?`（自適應，伺服器建議 ${c.optim?c.optim.recommended_fps:'—'}）`:'（固定）'}`};

 $('mKv').innerHTML=Object.entries(kv).map(([k,v])=>`<span>${k}</span><span>${v}</span>`).join('');
}
function addCalib(){const det=$('mod_distance');if(!det)return;
 det.insertAdjacentHTML('beforeend',`<div class="row"><div class="top"><label>距離校正</label><input type="number" id="calM" class="control" min="0.2" max="10" step="0.1" value="1.0"><span class="def">m</span><button class="button button-secondary button-small" id="calB" type="button">我站在這個距離</button></div>
  <div class="tip">站在鏡頭前已知的距離（例如用捲尺量 1.0 m），然後按按鈕。系統會用目前最近那個人的估計值算出校正係數。需要先啟用距離估計並開始偵測。</div></div>`);
 $('calB').onclick=()=>{if(!running){toast('請先按「開始」');return}wsSend({type:'calibrate',meters:+$('calM').value})}}
