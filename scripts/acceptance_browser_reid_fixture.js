// Child acceptance page only. Identity strings remain inside this closure.
(() => {
  let firstCID, firstSession=0, currentSession=0, handoffs=0, resumed=0, stale=0, observedCalling=0;
  const provisionals=new Map();
  const NativeSocket=window.WebSocket;
  window.WebSocket=class extends NativeSocket {
    constructor(...args) {
      super(...args);observedCalling=0;
      this.addEventListener('message',e=>{
        if(typeof e.data!=='string')return;
        const m=JSON.parse(e.data);
        if(m.type==='state')currentSession=m.session;
        if(m.type==='reid')observedCalling=+(m.phase==='calling');
        if(m.type==='reid'&&m.conversation_id&&m.phase!=='idle')
          provisionals.set(m.session,m.conversation_id);
        if(m.type==='handoff') {
          handoffs++;
          if(!firstCID&&m.decision==='new'){firstCID=m.conversation_id;firstSession=m.session;}
          if(m.decision==='resume') {
            resumed++;
            stale|=+(m.session!==currentSession||m.session===firstSession||m.conversation_id!==firstCID
              ||!provisionals.has(m.session)||m.previous_conversation_id!==provisionals.get(m.session)
              ||m.resumed_from_session!==firstSession);
          }
        }
      });
    }
  };
  window.acceptanceReid={
    configure(consent) {acceptanceFixture.configure();Object.assign(cfg,{reid_enabled:true,reid_consent:consent,
      reid_show_crops:false,reid_cooldown:0,reid_timeout:10,reid_decision_timeout:10});syncUI();sendCfg();},
    consent(on) {cfg.reid_consent=on;syncUI();sendCfg();},
    snapshot() {return {...acceptanceFixture.snapshot(),handoffs,resumed,stale,
      provisionalDifferent:+!!(provisionals.has(currentSession)&&firstCID&&provisionals.get(currentSession)!==firstCID),
      uiCurrent:+!!(REID.handoff&&REID.handoff.session===last?.session),
      uiResumed:+!!(REID.handoff?.decision==='resume'&&REID.handoff.conversation_id===firstCID),
      uiCleared:+!(REID.handoff||REID.st?.conversation_id),
      refs:REID.st?.refs||0,calling:observedCalling};}
  };
})();
