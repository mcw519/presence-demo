// Loaded after the real UI scripts, with networking replaced by benchmark_frontend.py.
// Observe actual DOM/canvas work; no mock canvas timing is used.
addEventListener('load',()=>{
 const counts={htmlWrites:0,canvasResizes:0,pixelReadbacks:0,audioRenders:0,stateRenders:0};
 const times=[],renderTimes=[];let timer;
 const wrap=(proto,key,count)=>{const d=Object.getOwnPropertyDescriptor(proto,key);
  Object.defineProperty(proto,key,{...d,set(value){counts[count]++;d.set.call(this,value)}})};
 wrap(Element.prototype,'innerHTML','htmlWrites');
 wrap(HTMLCanvasElement.prototype,'width','canvasResizes');
 wrap(HTMLCanvasElement.prototype,'height','canvasResizes');
 const read=CanvasRenderingContext2D.prototype.getImageData;
 CanvasRenderingContext2D.prototype.getImageData=function(...a){counts.pixelReadbacks++;return read.apply(this,a)};
 const measured=fn=>function(...a){const t=performance.now();try{return fn(...a)}finally{renderTimes.push(performance.now()-t)}};
 const oldAudio=renderAudio,oldRender=render;
 renderAudio=measured(function(...a){counts.audioRenders++;return oldAudio(...a)});
 render=measured(function(...a){counts.stateRenders++;return oldRender(...a)});
 renderStats=measured(renderStats);wave=measured(wave);sg=measured(sg);drawTL=measured(drawTL);
 WR.push(...Array(250).fill(.1));WC.push(...Array(250).fill(.05));
 anRaw=anClean={getByteFrequencyData(a){a.fill(30)}};
 cfg.adaptive_fps=true;cfg.fps=1;running=true;
 const deliver=s=>ws.onmessage({data:JSON.stringify(s)});
 const begin=performance.now();let n=0;
 const finish=()=>{
  clearInterval(timer);running=false;clearInterval(frameTimer);
  const ordered=times.slice().sort((a,b)=>a-b);
  const result={kind:'benchmark',version:BENCH_VERSION,seconds:(performance.now()-begin)/1000,
   messages:n,actualFpsSetting:curFps(),...counts,
   deliveryMsAvg:times.reduce((a,b)=>a+b,0)/times.length,
   deliveryMsP95:ordered[Math.min(ordered.length-1,Math.floor(ordered.length*.95))],
   renderWorkTotalMs:renderTimes.reduce((a,b)=>a+b,0),errors:benchErrors};
  if(typeof FEPerf!=='undefined')result.frontend=FEPerf.snapshot();
  parent.postMessage(result,location.origin);
 };
 timer=setInterval(()=>{
  const start=performance.now();
  deliver({...FIXTURE.audio,raw_db:-60+(n%10),clean_db:-70+(n%7)});
  if(n%5===0)deliver(FIXTURE.state);
  if(n%10===0)deliver(FIXTURE.stats);
  times.push(performance.now()-start);n++;
  if(n>=BENCH_SECONDS*10){clearInterval(timer);setTimeout(finish,250)}
 },100);
});
