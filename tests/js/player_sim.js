// drive the Play worklet in node: bursty arrivals, check latency stays bounded
globalThis.sampleRate=48000;let P;globalThis.AudioWorkletProcessor=class{constructor(){this.port={postMessage:m=>{last=m},onmessage:null}}};let last;
globalThis.registerProcessor=(n,c)=>{if(n==='play')P=c};
eval(require('fs').readFileSync(process.argv[2],'utf8'));
const p=new P();const send=()=>p.port.onmessage({data:new Float32Array(320).fill(0.1)});
const out=[[new Float32Array(128)]];let maxb=0;
for(let q=0;q<48000/128*20;q++){const tms=q*128/48;
 if(Math.floor(tms/20)!==Math.floor((tms-128/48)/20)){if((tms%5000)<400){}else if((tms%5000)<408){for(let k=0;k<21;k++)send()}else send()}
 p.process([],out);if(tms>1000)maxb=Math.max(maxb,p.depth()/16)}
console.log('max buffer ms',maxb.toFixed(0),'stats',JSON.stringify(last));
