// make-parallax.mjs — 图 + 深度图 → 自包含鼠标视差 HTML（WebGL，单文件可分发）
// 用法: node make-parallax.mjs <img> <depth.png> <out.html> [shift%] [look|float] [contrast] [lerp]
// 视差口径: 亮=近（DA-V2 输出）。look=环视（近锚定远移动）；float=浮层（近层随鼠标浮起）
// contrast=深度对比拉伸（>1 拉开层次）；lerp=跟随惯性系数（大=快）
import fs from 'node:fs';
import sharp from 'sharp';

const [, , IMG, DEPTH, OUTHTML, SHIFTARG, MODEARG, CONTRASTARG, LERPARG] = process.argv;
if (!IMG || !DEPTH || !OUTHTML) {
  console.error('用法: node make-parallax.mjs <img> <depth.png> <out.html> [shift%] [look|float] [contrast] [lerp]');
  process.exit(1);
}
const SHIFT = Math.min(0.08, Math.max(0.02, (SHIFTARG ? parseFloat(SHIFTARG) : 4) / 100));
const MODE = MODEARG === 'float' ? 1 : 0;
const CONTRAST = CONTRASTARG ? parseFloat(CONTRASTARG) : 1;
const LERP = LERPARG ? parseFloat(LERPARG) : 0.06;

// 收缩到 1600 内嵌（保证单文件 <1.5MB）
const jpg = await sharp(IMG).resize({ width: 1600, height: 1600, fit: 'inside', withoutEnlargement: true })
  .jpeg({ quality: 82 }).toBuffer();
const dep = await sharp(DEPTH).resize({ width: 1600, height: 1600, fit: 'inside', withoutEnlargement: true })
  .greyscale().png({ compressionLevel: 9 }).toBuffer();

const html = `<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D 景深壁纸 · 鼠标视差</title>
<style>
html,body{margin:0;height:100%;background:#000;overflow:hidden}
canvas{position:fixed;inset:0;width:100%;height:100%;cursor:crosshair}
#hint{position:fixed;left:50%;bottom:6vh;transform:translateX(-50%);color:#fff;
font:14px/1.6 system-ui;opacity:.75;transition:opacity 1.2s;text-shadow:0 1px 8px #000;user-select:none;pointer-events:none}
</style></head><body>
<canvas id="c"></canvas><div id="hint">移动鼠标，穿过画面</div>
<script>
var IMG="data:image/jpeg;base64,__IMG__";
var DEP="data:image/png;base64,__DEP__";
var SHIFT=__SHIFT__;
var MODE=__MODE__;
var CONTRAST=__CONTRAST__;
var LERP=__LERP__;

var canvas=document.getElementById('c');
var hint=document.getElementById('hint');
var gl=canvas.getContext('webgl',{antialias:true});

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
var uMouse=gl.getUniformLocation(prog,'uMouse');
var uShiftL=gl.getUniformLocation(prog,'uShift');
var uCS=gl.getUniformLocation(prog,'uCoverScale');
var uCO=gl.getUniformLocation(prog,'uCoverOff');
gl.uniform1f(uShiftL,SHIFT);
gl.uniform1f(gl.getUniformLocation(prog,'uMode'),MODE);
gl.uniform1f(gl.getUniformLocation(prog,'uContrast'),CONTRAST);

var iw=0,ih=0,ready=0;
function loadTex(url,unit){
  var im=new Image();
  im.onload=function(){
    var t=gl.createTexture();
    gl.activeTexture(gl.TEXTURE0+unit);
    gl.bindTexture(gl.TEXTURE_2D,t);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,im);
    if(unit===0){iw=im.naturalWidth;ih=im.naturalHeight;fit();}
    ready++;
  };
  im.src=url;
}
loadTex(IMG,0);
loadTex(DEP,1);
gl.uniform1i(gl.getUniformLocation(prog,'uTex'),0);
gl.uniform1i(gl.getUniformLocation(prog,'uDepth'),1);

function fit(){
  var dpr=Math.min(2,window.devicePixelRatio||1);
  canvas.width=Math.floor(innerWidth*dpr);
  canvas.height=Math.floor(innerHeight*dpr);
  gl.viewport(0,0,canvas.width,canvas.height);
  if(!iw)return;
  var s=Math.max(canvas.width/iw,canvas.height/ih);
  var fx=canvas.width/(s*iw),fy=canvas.height/(s*ih);
  gl.uniform2f(uCS,fx,fy);
  gl.uniform2f(uCO,(1-fx)/2,(1-fy)/2);
}
window.addEventListener('resize',fit);fit();

var mx=0,my=0,tx=0,ty=0,idle=0,last=performance.now();
function onMove(x,y){
  tx=(x/innerWidth)*2-1;
  ty=-((y/innerHeight)*2-1);
  idle=0;hint.style.opacity=0;
}
window.addEventListener('mousemove',function(e){onMove(e.clientX,e.clientY)});
window.addEventListener('touchmove',function(e){
  if(e.touches[0])onMove(e.touches[0].clientX,e.touches[0].clientY);
},{passive:true});

function frame(t){
  var dt=Math.min(0.05,(t-last)/1000);last=t;idle+=dt;
  // 空闲 3 秒后缓慢自动环视，方便一眼看出 3D 效果
  var kx=idle>3?Math.sin(t/1000*0.7)*0.4:tx;
  var ky=idle>3?Math.cos(t/1000*0.5)*0.3:ty;
  mx+=(kx-mx)*LERP;my+=(ky-my)*LERP;
  gl.uniform2f(uMouse,mx,my);
  if(ready>=2)gl.drawArrays(gl.TRIANGLES,0,6);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script></body></html>`;

fs.writeFileSync(OUTHTML, html.replace('__IMG__', jpg.toString('base64')).replace('__DEP__', dep.toString('base64')).replace('__SHIFT__', String(SHIFT)).replace('__MODE__', String(MODE)).replace('__CONTRAST__', String(CONTRAST)).replace('__LERP__', String(LERP)));
console.log('✔', OUTHTML, (fs.statSync(OUTHTML).size / 1e6).toFixed(2) + 'MB',
  'shift=' + (SHIFT * 100) + '%', 'mode=' + (MODE ? 'float' : 'look'), 'contrast=' + CONTRAST, 'lerp=' + LERP);
