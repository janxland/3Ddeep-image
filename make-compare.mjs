// make-compare.mjs — 多参数视差对比页（同屏 6 面板，共用鼠标；点面板进全屏单文件）
// 用法: node make-compare.mjs <img> <depth.png> <out.html> <文件前缀>
// 参数口径参考 qianqian 裸眼3D教程：幅度从小起、拖影减幅、深度贴图对比、层模式、跟随惯性
import fs from 'node:fs';
import sharp from 'sharp';

const [, , IMG, DEPTH, OUTHTML, PREFIX] = process.argv;
if (!IMG || !DEPTH || !OUTHTML) {
  console.error('用法: node make-compare.mjs <img> <depth.png> <out.html> <文件前缀>');
  process.exit(1);
}
const P = PREFIX || 'v';

const PANELS = [
  { key: 'look2',  name: 'A · 小幅 2%',    tag: '环视 · 千千建议起步档', shift: 0.02, mode: 0, contrast: 1,   lerp: 0.06 },
  { key: 'look4',  name: 'B · 基准 4%',    tag: '环视 · 推荐档',         shift: 0.04, mode: 0, contrast: 1,   lerp: 0.06 },
  { key: 'look7',  name: 'C · 大幅 7%',    tag: '环视 · 边缘易拖影',     shift: 0.07, mode: 0, contrast: 1,   lerp: 0.06 },
  { key: 'gamma5', name: 'D · 深度增强 5%', tag: '环视 · 深度对比 ×1.8',  shift: 0.05, mode: 0, contrast: 1.8, lerp: 0.06 },
  { key: 'float5', name: 'E · 浮层 5%',    tag: '近层随鼠标（贴纸感）',  shift: 0.05, mode: 1, contrast: 1,   lerp: 0.06 },
  { key: 'fast5',  name: 'F · 快跟随 5%',  tag: '惯性小 · 更跟手',       shift: 0.05, mode: 0, contrast: 1,   lerp: 0.16 },
];

// 收缩到 1600 内嵌
const jpg = await sharp(IMG).resize({ width: 1600, height: 1600, fit: 'inside', withoutEnlargement: true })
  .jpeg({ quality: 82 }).toBuffer();
const dep = await sharp(DEPTH).resize({ width: 1600, height: 1600, fit: 'inside', withoutEnlargement: true })
  .greyscale().png({ compressionLevel: 9 }).toBuffer();

const html = `<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>景深壁纸 · 参数对比</title>
<style>
:root{color-scheme:dark}
html,body{margin:0;background:#0b0b0f;color:#eee;font:14px/1.5 system-ui}
header{padding:14px 18px 4px;display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap}
header h1{font-size:16px;margin:0;font-weight:600}
header .tip{color:#888;font-size:12px}
#grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:12px 18px 18px}
@media (max-width:1000px){#grid{grid-template-columns:repeat(2,1fr)}}
.panel{position:relative;border-radius:12px;overflow:hidden;aspect-ratio:2/3;cursor:pointer;
box-shadow:0 2px 14px rgba(0,0,0,.5);outline:1px solid #222;transition:outline-color .2s}
.panel:hover{outline-color:#8ab4ff}
.panel canvas{position:absolute;inset:0;width:100%;height:100%}
.bar{position:absolute;top:0;left:0;right:0;z-index:2;display:flex;justify-content:space-between;gap:8px;
background:linear-gradient(rgba(0,0,0,.72),rgba(0,0,0,0));padding:8px 10px;font-size:12px;pointer-events:none}
.bar b{font-weight:600}
.bar span{color:#bbb;text-align:right}
</style></head><body>
<header><h1>景深壁纸 · 6 参数对比</h1><div class="tip">鼠标全页移动同时驱动 6 格 · 点任意卡片进全屏版 · 空闲 3 秒自动演示</div></header>
<div id="grid"></div>
<script>
var IMG="data:image/jpeg;base64,__IMG__";
var DEP="data:image/png;base64,__DEP__";
var PANELS=[
{name:'A · 小幅 2%',tag:'环视 · 千千建议起步档',file:'__P__.look2.html',shift:0.02,mode:0,contrast:1,lerp:0.06},
{name:'B · 基准 4%',tag:'环视 · 推荐档',file:'__P__.look4.html',shift:0.04,mode:0,contrast:1,lerp:0.06},
{name:'C · 大幅 7%',tag:'环视 · 边缘易拖影',file:'__P__.look7.html',shift:0.07,mode:0,contrast:1,lerp:0.06},
{name:'D · 深度增强 5%',tag:'环视 · 深度对比 ×1.8',file:'__P__.gamma5.html',shift:0.05,mode:0,contrast:1.8,lerp:0.06},
{name:'E · 浮层 5%',tag:'近层随鼠标（贴纸感）',file:'__P__.float5.html',shift:0.05,mode:1,contrast:1,lerp:0.06},
{name:'F · 快跟随 5%',tag:'惯性小 · 更跟手',file:'__P__.fast5.html',shift:0.05,mode:0,contrast:1,lerp:0.16}
];

var VS='attribute vec2 aPos;varying vec2 vUv;\\
void main(){vUv=aPos*0.5+0.5;vUv.y=1.0-vUv.y;gl_Position=vec4(aPos,0.0,1.0);}';
var FS='precision highp float;\\
uniform sampler2D uTex;uniform sampler2D uDepth;\\
uniform vec2 uCoverScale;uniform vec2 uCoverOff;\\
uniform vec2 uMouse;uniform float uShift;uniform float uMode;uniform float uContrast;\\
varying vec2 vUv;\\
void main(){\\
  vec2 base=(vUv-0.5)*(1.0-2.0*uShift)+0.5;\\
  vec2 iuv=base*uCoverScale+uCoverOff;\\
  float d=texture2D(uDepth,iuv).r;\\
  d=clamp((d-0.5)*uContrast+0.5,0.0,1.0);\\
  vec2 disp=(uMode<0.5)?(-uMouse*uShift*(1.0-d)):(uMouse*uShift*d);\\
  vec2 suv=clamp(iuv+disp,vec2(0.001),vec2(0.999));\\
  gl_FragColor=vec4(texture2D(uTex,suv).rgb,1.0);\\
}';

var grid=document.getElementById('grid');
var panels=[];
PANELS.forEach(function(p){
  var box=document.createElement('div');box.className='panel';
  var bar=document.createElement('div');bar.className='bar';
  bar.innerHTML='<b>'+p.name+'</b><span>'+p.tag+'</span>';
  var cv=document.createElement('canvas');
  box.appendChild(bar);box.appendChild(cv);
  box.onclick=function(){location.href=p.file;};
  grid.appendChild(box);
  panels.push(initPanel(cv,p));
});

function initPanel(cv,p){
  var gl=cv.getContext('webgl',{antialias:true});
  function sh(type,src){var s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(s);return s;}
  var prog=gl.createProgram();
  gl.attachShader(prog,sh(gl.VERTEX_SHADER,VS));
  gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,FS));
  gl.linkProgram(prog);
  if(!gl.getProgramParameter(prog,gl.LINK_STATUS))throw gl.getProgramInfoLog(prog);
  gl.useProgram(prog);
  var buf=gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER,buf);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,-1,1,1,-1,1]),gl.STATIC_DRAW);
  var aPos=gl.getAttribLocation(prog,'aPos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos,2,gl.FLOAT,false,0,0);
  gl.uniform1f(gl.getUniformLocation(prog,'uShift'),p.shift);
  gl.uniform1f(gl.getUniformLocation(prog,'uMode'),p.mode);
  gl.uniform1f(gl.getUniformLocation(prog,'uContrast'),p.contrast);
  gl.uniform1i(gl.getUniformLocation(prog,'uTex'),0);
  gl.uniform1i(gl.getUniformLocation(prog,'uDepth'),1);
  return {gl:gl,cv:cv,p:p,prog:prog,
    uMouse:gl.getUniformLocation(prog,'uMouse'),
    uCS:gl.getUniformLocation(prog,'uCoverScale'),
    uCO:gl.getUniformLocation(prog,'uCoverOff'),
    mx:0,my:0,ready:0};
}

function uploadTex(c,im,unit){
  var gl=c.gl;
  gl.activeTexture(gl.TEXTURE0+unit);
  var t=gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D,t);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,im);
  c.ready++;
}

var iw=0,ih=0;
var imC=new Image(),imD=new Image(),loaded=0;
function onImg(){
  loaded++;if(loaded<2)return;
  iw=imC.naturalWidth;ih=imC.naturalHeight;
  panels.forEach(function(c){uploadTex(c,imC,0);uploadTex(c,imD,1);});
  fitAll();
}
imC.onload=imD.onload=onImg;
imC.src=IMG;imD.src=DEP;

function fitAll(){
  var dpr=Math.min(2,window.devicePixelRatio||1);
  panels.forEach(function(c){
    var r=c.cv.parentElement.getBoundingClientRect();
    c.cv.width=Math.max(2,Math.floor(r.width*dpr));
    c.cv.height=Math.max(2,Math.floor(r.height*dpr));
    c.gl.viewport(0,0,c.cv.width,c.cv.height);
    if(!iw)return;
    var s=Math.max(c.cv.width/iw,c.cv.height/ih);
    var fx=c.cv.width/(s*iw),fy=c.cv.height/(s*ih);
    c.gl.useProgram(c.prog);
    c.gl.uniform2f(c.uCS,fx,fy);
    c.gl.uniform2f(c.uCO,(1-fx)/2,(1-fy)/2);
  });
}
window.addEventListener('resize',fitAll);

var tx=0,ty=0,idle=0,last=performance.now();
function onMove(x,y){tx=(x/innerWidth)*2-1;ty=-((y/innerHeight)*2-1);idle=0;}
window.addEventListener('mousemove',function(e){onMove(e.clientX,e.clientY)});
window.addEventListener('touchmove',function(e){
  if(e.touches[0])onMove(e.touches[0].clientX,e.touches[0].clientY);
},{passive:true});

function frame(t){
  var dt=Math.min(0.05,(t-last)/1000);last=t;idle+=dt;
  var kx=idle>3?Math.sin(t/1000*0.7)*0.4:tx;
  var ky=idle>3?Math.cos(t/1000*0.5)*0.3:ty;
  panels.forEach(function(c){
    c.mx+=(kx-c.mx)*c.p.lerp;c.my+=(ky-c.my)*c.p.lerp;
    c.gl.useProgram(c.prog);
    c.gl.uniform2f(c.uMouse,c.mx,c.my);
    if(c.ready>=2)c.gl.drawArrays(c.gl.TRIANGLES,0,6);
  });
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script></body></html>`;

fs.writeFileSync(OUTHTML, html
  .replace('__IMG__', jpg.toString('base64'))
  .replace('__DEP__', dep.toString('base64'))
  .replace(/__P__/g, P));
console.log('✔', OUTHTML, (fs.statSync(OUTHTML).size / 1e6).toFixed(2) + 'MB', '前缀=' + P);
