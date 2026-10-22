// Diagnostic page only. The native getUserMedia function is never called.
// Production AudioWorklets, WebSocket, encoding and rendering use synthetic inputs.
let syntheticSourceContext, syntheticPaint;
Object.defineProperty(navigator.mediaDevices, 'getUserMedia', {value: async () => {
  const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;
  const ctx=canvas.getContext('2d'),face=new Image();face.src='/debug/synthetic-face.jpg';
  await face.decode();
  const paint=()=>{ctx.fillStyle='#505050';ctx.fillRect(0,0,640,480);
    for(const x of [60,360])ctx.drawImage(face,x,60,220,360)};
  paint();syntheticPaint=setInterval(paint,1000/30);
  const video=canvas.captureStream(30);
  const audio=new AudioContext({sampleRate:48000});syntheticSourceContext=audio;
  await audio.resume();
  const buffer=audio.createBuffer(1,48000*5,48000),data=buffer.getChannelData(0);let seed=42;
  for(let i=0;i<data.length;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;data[i]=((seed>>>0)/4294967296-.5)*.06}
  const source=audio.createBufferSource();source.buffer=buffer;source.loop=true;
  const sink=audio.createMediaStreamDestination();source.connect(sink);source.start();
  return new MediaStream([...video.getVideoTracks(),...sink.stream.getAudioTracks()]);
}});
addEventListener('load',()=>{
  Object.assign(cfg,{audio_pipeline:'puresound',asr_provider:'none',reid_enabled:false,reid_consent:false,
    adaptive_fps:false,fps:2,motion_gate:false,pose_enabled:true,monitor:'off'});
  syncUI();sendCfg();
  const note=document.createElement('div');note.textContent='Synthetic inputs only — no microphone, camera or cloud. Click Start; stops after 30 seconds.';
  note.style.cssText='padding:12px;background:#253858;color:white';document.body.prepend(note);
  const output=document.createElement('pre');output.id='synthetic-result';document.body.append(output);
  const originalStart=start;
  start=async()=>{
    await originalStart();
    setTimeout(()=>{
      const result={kind:'browser_synthetic_stream',seconds:30,audio:aMeta,frontend:FEPerf.snapshot()};
      stop();clearInterval(syntheticPaint);syntheticSourceContext?.close();
      output.textContent=JSON.stringify(result,null,2);note.textContent='Synthetic test complete';
    },30000);
  };
});
