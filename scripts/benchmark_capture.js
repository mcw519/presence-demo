// Real browser encoding and production framing, synthetic pixels, fake socket.
addEventListener('load',()=>{
 const source=document.createElement('canvas');source.width=640;source.height=480;
 Object.defineProperties(source,{videoWidth:{value:640},videoHeight:{value:480}});
 const ctx=source.getContext('2d'),pixels=ctx.createImageData(640,480);let seed=42;
 for(let i=0;i<pixels.data.length;i+=4){seed=(Math.imul(seed,1664525)+1013904223)>>>0;
  pixels.data[i]=(i/4)%640/3;pixels.data[i+1]=(i/2560)/2;
  pixels.data[i+2]=(seed>>>24)&255;pixels.data[i+3]=255}
 ctx.putImageData(pixels,0,0);v=source;
 const packets=[],jobs=[];
 ws.send=data=>{
  const observerStart=performance.now();try{
  if(typeof data==='string'){
   const m=JSON.parse(data);if(m.type!=='frame')return;
   const bytes=Uint8Array.from(atob(m.data.split(',')[1]),c=>c.charCodeAt(0));
   packets.push({format:'json',wireBytes:new TextEncoder().encode(data).length,jpeg:bytes});
  }else if(data instanceof Blob){
   jobs.push(data.arrayBuffer().then(buf=>{
    const d=new DataView(buf);if(d.getUint8(0)!==4||d.getUint8(9)!==0)throw Error('bad video header');
    packets.push({format:'binary',wireBytes:buf.byteLength,jpeg:new Uint8Array(buf,10)});
   }));
  }
  }finally{window.benchObserverMs+=performance.now()-observerStart}
 };
 cfg.adaptive_fps=false;cfg.fps=5;running=true;
 const start=performance.now();restartFrames();window.benchCaptureTimer=frameTimer;
 setTimeout(()=>{
  clearInterval(frameTimer);
  setTimeout(async()=>{
   await Promise.all(jobs);
   running=false;
   const hashes=await Promise.all(packets.map(async p=>{
    const digest=await crypto.subtle.digest('SHA-256',p.jpeg);
    return [...new Uint8Array(digest)].map(v=>v.toString(16).padStart(2,'0')).join('');
   }));
   const a=benchCaptureTicks.slice().sort((a,b)=>a-b),sum=a.reduce((a,b)=>a+b,0);
   parent.postMessage({kind:'benchmark',version:BENCH_VERSION,mode:'capture',
    seconds:(performance.now()-start)/1000,frames:packets.length,
    format:packets[0]?.format,jpegBytes:packets[0]?.jpeg.length,
    jpegSha256:[...new Set(hashes)],wireBytes:packets.reduce((n,p)=>n+p.wireBytes,0),
    wireBytesPerFrame:packets.length?packets.reduce((n,p)=>n+p.wireBytes,0)/packets.length:0,
    captureCallbackTotalMs:sum,captureCallbackAvgMs:a.length?sum/a.length:0,
    captureCallbackP95Ms:a.length?a[Math.min(a.length-1,Math.ceil(a.length*.95)-1)]:0,
    frontend:FEPerf.snapshot(),errors:benchErrors},location.origin);
  },500);
 },BENCH_SECONDS*1000);
});
