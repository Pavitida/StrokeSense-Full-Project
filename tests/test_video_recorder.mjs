import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source = await readFile(new URL('../src/video_recorder.js', import.meta.url), 'utf8');
const {default:mount} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const elements = Object.fromEntries(['video','#start','#stop','#clear','#status'].map(k => [k,{disabled:false, textContent:'', play:async()=>{}, removeAttribute:()=>{}, load:()=>{}}]));
const states = [];
let trackStopped = false;
Object.defineProperty(globalThis,'navigator',{configurable:true,value:{mediaDevices:{getUserMedia:async()=>({getTracks:()=>[{stop:()=>{trackStopped=true;}}]})}}});
globalThis.window = globalThis;
globalThis.MediaRecorder = class {
  static isTypeSupported(type) { return type.startsWith('video/webm'); }
  constructor(stream, options) { this.mimeType=options.mimeType; this.state='inactive'; }
  start() { this.state='recording'; }
  stop() { this.state='inactive'; this.ondataavailable({data:new Blob(['test video'])}); this.onstop(); }
};
globalThis.FileReader = class { readAsDataURL(blob) { blob.arrayBuffer().then(bytes=>{this.result='data:video/webm;base64,'+Buffer.from(bytes).toString('base64');this.onload();}); } };
const cleanup = mount({parentElement:{querySelector:s=>elements[s]},setStateValue:(k,v)=>states.push([k,v])});
await elements['#start'].onclick();
assert.equal(elements['#stop'].disabled,false);
assert.deepEqual(states.at(-1),['clip',null]);
elements['#stop'].onclick();
await new Promise(resolve=>setImmediate(resolve));
assert(trackStopped);
assert.equal(Buffer.from(states.at(-1)[1].base64,'base64').toString(),'test video');
elements['#clear'].onclick();
assert.deepEqual(states.at(-1),['clip',null]);
cleanup();
navigator.mediaDevices.getUserMedia = async()=>{throw Object.assign(new Error('denied'),{name:'NotAllowedError'});};
const cleanup2=mount({parentElement:{querySelector:s=>elements[s]},setStateValue:()=>{}});
await elements['#start'].onclick();
assert.match(elements['#status'].textContent,/ไม่ได้รับอนุญาต/);
assert.equal(elements['#start'].disabled,false);
cleanup2();
console.log('Recorder start/stop, transfer, clear, cleanup and denied-permission tests passed (mock media).');
