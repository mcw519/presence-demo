// Injected before production scripts. Native devices are never requested.
(() => {
  const state = {connections: 0, sameCapability: 1, videoSent: 0, pcmSent: 0,
    framesProcessed: 0, audioReceived: 0, errors: 0, interrupted: 0, session: 0,
    ended: 1, requests: 0, tracks: [], contexts: [], intervals: [], source: null};
  let capability;
  const NativeSocket = window.WebSocket;
  window.WebSocket = class extends NativeSocket {
    constructor(url, protocols) {
      if (new URL(url).hostname !== '127.0.0.1') throw new Error('Nonlocal socket blocked');
      super(url, protocols);
      state.connections++;
      const currentCounters = {frames:0, pcm:0, video:0, audio:0};
      this.acceptanceCounters=currentCounters;state.current=currentCounters;
      const current = protocols.find(p => p.startsWith('page-'));
      if (capability === undefined) capability = current;
      else if (current !== capability) state.sameCapability = 0;
      this.addEventListener('message', event => {
        if (typeof event.data !== 'string') {state.audioReceived++;currentCounters.audio++;return;}
        const message = JSON.parse(event.data);
        if (message.type === 'state' && state.current===currentCounters) {
          state.interrupted = +(message.sensor_status === 'interrupted');
          state.session = message.session;state.inState=message.in_state;
          state.ended = +(message.state === 'ENDED');
        }
        if (message.type === 'stats') {currentCounters.frames=message.conn.frames.processed;state.framesProcessed=Math.max(state.framesProcessed,currentCounters.frames);}
        if (message.type === 'error') state.errors++;
      });
    }
    send(data) {
      if (data instanceof Blob) {state.videoSent++;this.acceptanceCounters.video++;}
      if (data instanceof ArrayBuffer && new Uint8Array(data)[0] === 1) {state.pcmSent++;this.acceptanceCounters.pcm++;}
      return super.send(data);
    }
  };
  addEventListener('error', () => state.errors++);
  addEventListener('unhandledrejection', () => state.errors++);
  Object.defineProperty(navigator.mediaDevices, 'getUserMedia', {value: async () => {
    state.requests++;
    const canvas = document.createElement('canvas');canvas.width=640;canvas.height=480;
    const context=canvas.getContext('2d'), face=new Image();face.src='/_acceptance/face';await face.decode();
    const paint=()=>{context.fillStyle='#505050';context.fillRect(0,0,640,480);context.drawImage(face,180,60,280,360)};
    paint();state.intervals.push(setInterval(paint,1000/15));
    const video=canvas.captureStream(15), audio=new AudioContext({sampleRate:48000});
    state.contexts.push(audio);await audio.resume();
    const buffer=audio.createBuffer(1,48000,48000), data=buffer.getChannelData(0);let seed=42;
    for(let i=0;i<data.length;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;data[i]=(seed/4294967296-.5)*.06}
    const source=audio.createBufferSource();source.buffer=buffer;source.loop=true;
    const destination=audio.createMediaStreamDestination();source.connect(destination);source.start();state.source=source;
    const media=new MediaStream([...video.getTracks(),...destination.stream.getTracks()]);
    state.tracks.push(...media.getTracks());return media;
  }});
  window.acceptanceFixture = {
    configure() {Object.assign(cfg,{audio_pipeline:'puresound',asr_provider:'none',reid_enabled:false,reid_consent:false,
      adaptive_fps:false,fps:2,motion_gate:false,pose_enabled:false,intent_model:'rules',priority_mode:'vision',
      leave_distance:10,face_roi:[0,0,1,1],monitor:'off'});syncUI();sendCfg();},
    async begin() {await start();state.productionContext=ac;},
    pauseVideo() {clearInterval(frameTimer);frameTimer=null;frameRate=0;invalidateFrames();},
    resumeVideo() {restartFrames();},
    reconnect() {ws.close();},
    async finish() {state.productionContext=ac;stop();state.intervals.forEach(clearInterval);state.source?.stop();
      await Promise.all(state.contexts.map(context=>context.close()));},
    snapshot() {return {currentFramesProcessed:state.current?.frames||0,currentPcmSent:state.current?.pcm||0,
      currentVideoSent:state.current?.video||0,currentAudioReceived:state.current?.audio||0,connections:state.connections,sameCapability:state.sameCapability,videoSent:state.videoSent,
      pcmSent:state.pcmSent,framesProcessed:state.framesProcessed,audioReceived:state.audioReceived,
      errors:state.errors,interrupted:state.interrupted,session:state.session,ended:state.ended,inState:state.inState||0,
      requests:state.requests,running:+running,streamLive:+!!stream,audioLive:+!!ac,
      frameTimerLive:+(frameTimer!=null),tracksLive:state.tracks.filter(track=>track.readyState==='live').length,
      sourceContextsLive:state.contexts.filter(context=>context.state!=='closed').length,
      productionContextLive:+!!(state.productionContext && state.productionContext.state!=='closed')};}
  };
})();
