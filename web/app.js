const E=ECG,$=s=>document.querySelector(s),S={view:'overview',sel:0,t0:0,fz:0},FN=E.FN;
const COL={N:'#4ade80',S:'#38bdf8',V:'#f87171',F:'#fbbf24',Q:'#c084fc'},f4=v=>isFinite(v)?(Math.abs(v)>=100?v.toFixed(0):v.toPrecision(3)):'–';
const pill=c=>`<span class="pill" style="background:${COL[c]}">${c}</span>`,pc=v=>(100*v).toFixed(1)+'%',kpi=(a,b)=>`<div class="kpi"><b>${a}</b><span>${b}</span></div>`;
const PAGES={
 overview:{label:'Overview',group:'Synthetic demo',meta:'Browser-based analysis',eyebrow:'RESEARCH OVERVIEW',title:'ECG research, made interpretable.',subtitle:'Explore signal processing, rough-set feature reduction and fuzzy-rule classification in one research workspace.'},
 analyze:{label:'ECG Analysis',group:'Synthetic demo',meta:'Browser-based analysis',eyebrow:'SIGNAL WORKSPACE',title:'Read the signal. Inspect every beat.',subtitle:'Generate a controlled ECG example or upload a signal to inspect preprocessing, detected R-peaks and experimental class assignments.'},
 features:{label:'Feature Extraction',group:'Synthetic demo',meta:'Browser-based analysis',eyebrow:'BEAT CHARACTERISATION',title:'From waveform to measurable features.',subtitle:'Inspect beat-level descriptors and understand which attributes are retained by the current rough-set reduct.'},
 rough:{label:'Rough Set Analysis',group:'Synthetic demo',meta:'Browser-based analysis',eyebrow:'FEATURE REDUCTION',title:'Find the smallest informative feature set.',subtitle:'Explore discretisation, dependency degree, the QuickReduct search path and core attributes.'},
 fuzzy:{label:'Fuzzy Logic Engine',group:'Synthetic demo',meta:'Browser-based analysis',eyebrow:'INTERPRETABLE CLASSIFICATION',title:'See how fuzzy rules reach a decision.',subtitle:'Inspect learned membership functions, rule antecedents, firing strengths and class-level aggregation.'},
 eval:{label:'Model Evaluation',group:'Synthetic demo',meta:'Browser-based analysis',eyebrow:'EXPERIMENTAL RESULTS',title:'Compare models with the right context.',subtitle:'Review synthetic held-out results, confusion matrices and per-class performance. These metrics are not clinical evidence.'},
 backend:{label:'Research Backend',group:'MIT-BIH · Python',meta:'Local Python API · real MIT-BIH data',eyebrow:'PYTHON RESEARCH SERVICE',title:'Connect the interface to real experiments.',subtitle:'Check the local Python API, audit MIT-BIH files, launch reproducible experiments and inspect saved research results.'},
 inference:{label:'Model Inference',group:'MIT-BIH · Python',meta:'Local Python API · real MIT-BIH data',eyebrow:'UPLOADED ECG · PYTHON MODEL',title:'Run a saved research model on an ECG signal.',subtitle:'Inspect beat-level model outputs, local feature sensitivity, fuzzy rules and training-range drift diagnostics.'}
};
const NAV_ICON={
 overview:'<svg viewBox="0 0 24 24"><rect x="3.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.5"/></svg>',
 analyze:'<svg viewBox="0 0 24 24"><path d="M3 12h4l2.4-6 4.1 12 2.5-6H21"/></svg>',
 features:'<svg viewBox="0 0 24 24"><path d="M4 19V5M4 19h16M8 15l3-4 3 2 5-7"/><circle cx="8" cy="15" r="1"/><circle cx="11" cy="11" r="1"/><circle cx="14" cy="13" r="1"/></svg>',
 rough:'<svg viewBox="0 0 24 24"><path d="M12 3 4 7.5 12 12l8-4.5L12 3Z"/><path d="m4 12 8 4.5 8-4.5M4 16.5 12 21l8-4.5"/></svg>',
 fuzzy:'<svg viewBox="0 0 24 24"><path d="M3 18c3-10 5-10 8 0s5 10 10-12"/><path d="M3 6h18" stroke-dasharray="2 3"/></svg>',
 eval:'<svg viewBox="0 0 24 24"><path d="M4 19V5M4 19h16"/><rect x="7" y="11" width="3" height="6" rx=".7"/><rect x="12" y="8" width="3" height="9" rx=".7"/><rect x="17" y="4" width="3" height="13" rx=".7"/></svg>',
 backend:'<svg viewBox="0 0 24 24"><path d="M4 5h16v11H4zM8 20h8M12 16v4"/><path d="M8 9h8M8 12h5"/></svg>',
 inference:'<svg viewBox="0 0 24 24"><path d="M3 12h3l2.1-5 3.1 10 2.2-6 1.5 3H21"/><path d="M5 4h14v16H5z"/></svg>'
};
$('#nav').innerHTML=[...new Set(Object.values(PAGES).map(v=>v.group))].map(g=>`<div class="nav-group" role="group" aria-label="${g}"><div class="nav-group-label"><span>${g.toUpperCase()}</span></div>${Object.entries(PAGES).filter(([,v])=>v.group===g).map(([k,v])=>`<button class="nav-item ${k==='overview'?'on':''}" type="button" data-k="${k}" data-tip="${v.label}" aria-label="${v.label}" aria-current="${k==='overview'?'page':'false'}">${NAV_ICON[k]}<span class="nav-label">${v.label}</span></button>`).join('')}</div>`).join('');
$('#nav').onclick=e=>{const b=e.target.closest('button[data-k]');if(b)go(b.dataset.k)};
function setTheme(theme){document.documentElement.dataset.theme=theme;try{localStorage.setItem('pulselab-theme',theme)}catch(_){};const dark=theme==='dark';$('#theme-toggle').setAttribute('aria-label',dark?'Switch to light mode':'Switch to dark mode');$('#theme-toggle').setAttribute('aria-pressed',String(dark));document.querySelector('meta[name="theme-color"]').content=dark?'#101114':'#f5f5f7';if(S.view==='analyze'&&S.rec)draw();if(S.view==='fuzzy'&&S.M)A.fuzzy()}
$('#theme-toggle').onclick=()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');
const closeSidebar=()=>{document.body.classList.remove('sidebar-open');$('#sidebar-toggle').setAttribute('aria-expanded','false')};
$('#sidebar-toggle').onclick=()=>{const open=!document.body.classList.contains('sidebar-open');document.body.classList.toggle('sidebar-open',open);$('#sidebar-toggle').setAttribute('aria-expanded',String(open))};
$('#sidebar-backdrop').onclick=closeSidebar;
const isDesktop=()=>matchMedia('(min-width: 761px)').matches;
function setSidebarCollapsed(collapsed,persist=true){
 document.body.classList.toggle('sidebar-collapsed',collapsed);
 const b=$('#sidebar-collapse');if(b){b.setAttribute('aria-expanded',String(!collapsed));b.setAttribute('aria-label',collapsed?'Expand sidebar':'Collapse sidebar');b.title=(collapsed?'Expand sidebar':'Collapse sidebar')+' (Ctrl+B)'}
 hideNavTip();
 if(persist){try{localStorage.setItem('pulselab-sidebar',collapsed?'collapsed':'expanded')}catch(_){}}
 setTimeout(()=>{if(S.view==='analyze'&&S.rec)draw();if(S.view==='fuzzy'&&S.M)A.fuzzy()},320);
}
const navTip=document.createElement('div');navTip.className='nav-tip';navTip.setAttribute('role','tooltip');document.body.appendChild(navTip);
function showNavTip(el){if(!isDesktop()||!document.body.classList.contains('sidebar-collapsed'))return;const r=el.getBoundingClientRect();navTip.textContent=el.dataset.tip||'';navTip.style.left=(r.right+12)+'px';navTip.style.top=(r.top+r.height/2-navTip.offsetHeight/2)+'px';navTip.classList.add('show')}
function hideNavTip(){navTip.classList.remove('show')}
$('#nav').addEventListener('mouseover',e=>{const b=e.target.closest('.nav-item');if(b)showNavTip(b)});
$('#nav').addEventListener('mouseout',e=>{if(e.target.closest('.nav-item'))hideNavTip()});
$('#nav').addEventListener('focusin',e=>{const b=e.target.closest('.nav-item');if(b&&b.matches(':focus-visible'))showNavTip(b)});
$('#nav').addEventListener('focusout',hideNavTip);
$('.sidebar-scroll').addEventListener('scroll',hideNavTip,{passive:true});
$('#sidebar-collapse').onclick=()=>setSidebarCollapsed(!document.body.classList.contains('sidebar-collapsed'));
addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&!e.shiftKey&&!e.altKey&&e.key.toLowerCase()==='b'&&isDesktop()){e.preventDefault();setSidebarCollapsed(!document.body.classList.contains('sidebar-collapsed'))}});
{let saved=null;try{saved=localStorage.getItem('pulselab-sidebar')}catch(_){}
 // First visit on a narrow laptop screen starts as the compact rail; an explicit choice always wins.
 setSidebarCollapsed(saved?saved==='collapsed':(innerWidth<1180),false)}
// Preserve form controls when a view is re-rendered (refreshes must not reset the user's choices).
function keepForm(ids,render){const st={};ids.forEach(id=>{const el=document.getElementById(id);if(!el)return;st[id]=el.type==='checkbox'?el.checked:el.type==='file'?(el.files&&el.files[0]):el.value});render();
 ids.forEach(id=>{const el=document.getElementById(id);if(!el||st[id]===undefined)return;if(el.type==='checkbox')el.checked=st[id];else if(el.type==='file'){try{const dt=new DataTransfer();dt.items.add(st[id]);el.files=dt.files}catch(_){}}else if(el.tagName==='SELECT'){if([...el.options].some(o=>o.value===st[id]))el.value=st[id]}else el.value=st[id]})}
const BACKEND_FORM=['api-split','api-quick','api-download','api-tolerance','api-peak','api-baselines','api-save-models'];
const INFERENCE_FORM=['inference-file','inference-fs'];
function updateEnvironmentChip(){const chip=$('#environment-chip');if(!chip)return;const label=chip.querySelector('span:last-child'),dot=chip.querySelector('.dataset-chip-dot');const isBackend=['backend','inference'].includes(S.view);if(label)label.textContent=isBackend?(S.backendHealth?'PYTHON API CONNECTED':'PYTHON API STATUS'):'SIMULATED DATA';if(dot)dot.style.background=isBackend?(S.backendHealth?'var(--green)':'var(--amber)'):'#d59a3d';}
function go(k){if(!PAGES[k])return;S.view=k;document.querySelectorAll('#nav button').forEach(b=>{const active=b.dataset.k===k;b.classList.toggle('on',active);b.setAttribute('aria-current',active?'page':'false')});const meta=PAGES[k];$('#breadcrumb-current').textContent=meta.label;$('#page-eyebrow').textContent=meta.eyebrow;$('#page-title').textContent=meta.title;$('#page-subtitle').textContent=meta.subtitle;document.title=`${meta.label} · PulseLab`;{const pm=$('#page-meta-text');if(pm)pm.textContent=meta.meta}updateEnvironmentChip();$('#v').innerHTML=R[k]();(A[k]||(()=>{}))();closeSidebar()}
function plot(cv,ser,o={}){const d=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;cv.width=w*d;cv.height=h*d;const g=cv.getContext('2d');g.scale(d,d);
 const win=o.win||8,i0=Math.round((o.t0||0)*E.FS),i1=Math.min(ser[0].y.length,i0+Math.round(win*E.FS));let mn=1e9,mx=-1e9;
 for(const s of ser)for(let i=i0;i<i1;i++){mn=Math.min(mn,s.y[i]);mx=Math.max(mx,s.y[i])}const pad=(mx-mn)*.1||1;mn-=pad;mx+=pad;
 const X=i=>(i-i0)/(i1-i0)*w,Y=v=>h-(v-mn)/(mx-mn)*h;g.lineWidth=1.2;
 for(const s of ser){g.strokeStyle=s.c;g.beginPath();for(let i=i0;i<i1;i++)i===i0?g.moveTo(X(i),Y(s.y[i])):g.lineTo(X(i),Y(s.y[i]));g.stroke()}
 for(const m of o.marks||[])if(m.i>=i0&&m.i<i1){g.fillStyle=m.c;g.beginPath();g.arc(X(m.i),Y(ser[0].y[m.i]),m.sel?6:4,0,6.3);g.fill();g.font='11px sans-serif';g.fillText(m.t||'',X(m.i)-3,Y(ser[0].y[m.i])-9)}
 g.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--muted').trim()||'#8a98b3';g.font='11px sans-serif';for(let t=Math.ceil(i0/E.FS);t<=i1/E.FS;t++)g.fillText(t+'s',X(t*E.FS)+2,h-3)}
// ---------- analysis ----------
function analyze(sig,fs,ann,label){const p=E.process(sig,fs),pr=p.rows.map(r=>E.predict(S.M.mB,r.f));S.rec={p,pr,ann,label,fs:E.FS};S.sel=0;S.t0=0;if(ann)S.rec.det=E.detStats(p.rp,ann)}
function parseCSV(txt){const L=txt.split(/\r?\n/).map(l=>l.trim().split(/[,;\s\t]+/).map(Number)).filter(r=>r.length&&r.every(Number.isFinite));if(!L.length)throw Error('No numeric rows found');
 const c=L[0].length>=2&&L.slice(1,50).every((r,i)=>r[0]>L[i][0])?1:0;return L.map(r=>r[c])}
function sample(sc,seed,noise){const r=E.synth(seed,30,E.SCEN[sc],noise);return r}
const R={
overview:()=>{const M=S.M;return`<div class="card"><h2>Hybrid Rough-Fuzzy intelligent system for ECG arrhythmia detection</h2>
<p>Pipeline: <b>ECG → median-baseline + zero-phase band-pass (0.5–40 Hz) → Pan-Tompkins R-peaks → beat segmentation (−250/+400 ms) → 25 beat features → Rough-Set reduct (QuickReduct) → Fuzzy rule base (Low/Med/High, product t-norm) → AAMI class (N, S, V, F, Q)</b>.</p>
<div class="grid">${kpi(M.nTrain,'training beats')}${kpi(M.nTest,'held-out test beats')}${kpi(FN.length+' → '+M.rough.reduct.length,'features → rough-set reduct')}${kpi(M.mB.rules.length,'fuzzy rules (model B)')}${kpi(pc(M.evB.acc),'accuracy · B Rough+Fuzzy')}${kpi(pc(M.evA.acc),'accuracy · A Fuzzy (all)')}</div></div>
<div class="card"><h2>Important — what this web app is</h2><p class="mut">This is a client-side research demonstration (see docs/METHODOLOGY.md). The MIT-BIH database is not bundled, so these browser models train on <b>synthetic ECG</b> when the page loads. For real Python experiments, use the Research Backend workspace. Metrics describe this synthetic benchmark, <b>not</b> MIT-BIH performance. The browser uses 25 single-lead features and Haar wavelets; the Python scaffold has a different feature definition. A CSV upload is processed for exploratory inspection, but class predictions still come from the synthetic-trained model — the upload does not retrain or validate it. Academic prototype; not for diagnosis or clinical decisions.</p></div>
<div class="card"><h2>AAMI classes</h2>${E.CLS.map(c=>`<div>${pill(c)} ${E.CN[c]}</div>`).join('')}</div>`},
analyze:()=>`<div class="card"><div class="row"><select id="sc">${Object.keys(E.SCEN).map(k=>`<option>${k}</option>`).join('')}</select>
seed <input id="sd" type="number" value="${S.seed||5}"> noise <select id="nz"><option value=".02">low</option><option value=".05" selected>medium</option><option value=".12">high</option></select>
<button class="btn pri" id="go">Generate &amp; analyse</button><button class="btn" id="dl">Download sample CSV</button></div>
<div class="row">Upload CSV/TXT: <input type="file" id="fl" accept=".csv,.txt"> sampling rate (Hz) <input id="fs" type="number" value="360"><span class="mut" id="msg"></span></div></div>
<div id="res">${S.rec?resHTML():''}</div>`,
features:()=>{if(!S.rec)return'<div class="card">Run an analysis first.</div>';const{p}=S.rec,r=p.rows[S.sel],red=S.M.rough.reduct;
 return`<div class="card"><div class="row">Beat # <input type="number" id="bi" min="0" max="${p.rows.length-1}" value="${S.sel}"> <span class="mut">of ${p.rows.length} beats · predicted ${pill(E.CLS[S.rec.pr[S.sel].k])}</span></div>
 <canvas id="bc" style="height:150px"></canvas></div><div class="card"><div class="sc" style="max-height:480px"><table><tr><th>Feature</th><th>Domain</th><th>Value</th><th>In reduct</th></tr>
 ${FN.map((n,i)=>`<tr><td>${n}</td><td>${E.DOM(n)}</td><td>${f4(r.f[i])}</td><td>${red.includes(i)?'✓':''}</td></tr>`).join('')}</table></div></div>`},
rough:()=>{const r=S.M.rough;return`<div class="card"><h2>Rough-set attribute reduction</h2><div class="grid">${kpi(r.n,'class-balanced objects')}${kpi(r.bins,'equal-frequency bins')}${kpi(f4(r.gammaAll),'γ(C) all features')}${kpi(f4(r.gammaR),'γ(R) reduct')}${kpi(r.reduct.length,'reduct size')}${kpi(r.core.length,'core attributes')}</div>
<div class="row" style="margin-top:12px">tolerance <input id="tl" type="number" step=".01" min="0" max=".3" value="${r.tol}"> bins <input id="bn" type="number" min="3" max="5" value="${r.bins}"><button class="btn pri" id="rt">Retrain</button><span class="mut" id="rm"></span></div></div>
<div class="card"><h2>Reduct (QuickReduct order, γ after adding)</h2><table><tr><th>#</th><th>Feature</th><th>Domain</th><th>γ(R)</th></tr>${r.reduct.map((f,i)=>{const s=r.steps.find(x=>x.f===f);return`<tr><td>${i+1}</td><td>${FN[f]}${r.core.includes(f)?' (core)':''}</td><td>${E.DOM(FN[f])}</td><td>${s?f4(s.g):'–'}</td></tr>`}).join('')}</table></div>
<div class="card"><h2>Single-attribute dependency γ({a})</h2>${FN.map((n,i)=>`<div class="row" style="margin:3px 0"><span style="width:130px">${n}${r.reduct.includes(i)?' ✓':''}</span><div style="flex:1"><div class="bar" style="width:${100*r.single[i]/Math.max(...r.single,1e-9)}%"></div></div><span class="mut" style="width:50px">${f4(r.single[i])}</span></div>`).join('')}</div>`},
fuzzy:()=>{const m=S.M.mB;return`<div class="card"><h2>Membership functions (Low/Med/High, quantile-centred)</h2><div class="row"><select id="ff">${m.cols.map((c,i)=>`<option value="${i}" ${i==S.fz?'selected':''}>${FN[c]}</option>`).join('')}</select><span class="mut">dashed line = selected beat in Analysis</span></div><canvas id="mc"></canvas></div>
<div class="card"><h2>Rule base (${m.rules.length} rules)</h2><div class="row"><select id="rc"><option value="-1">all classes</option>${E.CLS.map((c,i)=>`<option value="${i}">${c}</option>`).join('')}</select></div><div class="sc" id="rl"></div></div>`},
eval:()=>{const M=S.M,cmT=(ev)=>`<table><tr><th>true \\ pred</th>${E.CLS.map(c=>`<th>${c}</th>`).join('')}</tr>${ev.cm.map((r,i)=>`<tr><td>${pill(E.CLS[i])}</td>${r.map((v,j)=>`<td style="background:rgba(56,189,248,${i===j?v/Math.max(1,ev.pc[i].n)*.5:v/Math.max(1,ev.pc[i].n)*.6})">${v}</td>`).join('')}</tr>`).join('')}</table>`;
 return`<div class="card"><h2>Model comparison (held-out synthetic records, inter-record split)</h2><table><tr><th>Model</th><th>Features</th><th>Rules</th><th>Accuracy</th><th>Macro-F1</th></tr>
<tr><td>A: Fuzzy (all features)</td><td>${FN.length}</td><td>${M.mA.rules.length}</td><td>${pc(M.evA.acc)}</td><td>${pc(M.evA.macroF1)}</td></tr>
<tr><td>B: Rough Set + Fuzzy</td><td>${M.rough.reduct.length}</td><td>${M.mB.rules.length}</td><td>${pc(M.evB.acc)}</td><td>${pc(M.evB.macroF1)}</td></tr></table><p class="mut">${M.nTest} test beats. Synthetic data — not MIT-BIH results.</p></div>
<div class="card"><h2>Confusion matrix — B</h2>${cmT(M.evB)}</div><div class="card"><h2>Confusion matrix — A</h2>${cmT(M.evA)}</div>
<div class="card"><h2>Per-class (B)</h2><table><tr><th>Class</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr>${M.evB.pc.map((c,i)=>`<tr><td>${pill(E.CLS[i])} ${E.CN[E.CLS[i]]}</td><td>${c.n}</td><td>${pc(c.p)}</td><td>${pc(c.r)}</td><td>${pc(c.f1)}</td></tr>`).join('')}</table></div>`}};
function resHTML(){const{p,pr,det,label}=S.rec,cnt=E.CLS.map((c,i)=>pr.filter(q=>q.k===i).length),hr=p.rp.length>1?60*E.FS/((p.rp[p.rp.length-1]-p.rp[0])/(p.rp.length-1)):0,dur=p.x.length/E.FS;
 return`<div class="card"><div class="grid">${kpi(label,'source')}${kpi(p.rp.length,'R-peaks detected')}${kpi(hr.toFixed(0)+' bpm','mean heart rate')}${det?kpi(pc(det.se)+' / '+pc(det.ppv),'R-peak sensitivity / PPV'):''}${E.CLS.map((c,i)=>kpi(cnt[i],c+' · '+E.CN[c])).join('')}</div></div>
<div class="card"><div class="mut">Raw</div><canvas id="c1" style="height:110px"></canvas><div class="mut" style="margin-top:8px">Filtered &amp; normalised, beats coloured by predicted class</div><canvas id="c2"></canvas>
<input type="range" id="t0" min="0" max="${Math.max(0,dur-8).toFixed(1)}" step=".5" value="${S.t0}"></div>
<div class="card"><h2>Beat-by-beat prediction (click a row)</h2><div class="sc"><table><tr><th>#</th><th>t (s)</th><th>Class</th><th>Score</th><th>Margin</th><th>Flag</th></tr>${pr.map((q,i)=>`<tr class="cl ${i===S.sel?'sel':''}" data-i="${i}"><td>${i}</td><td>${(p.rows[i].r/E.FS).toFixed(2)}</td><td>${pill(E.CLS[q.k])}</td><td>${f4(q.sc[q.k])}</td><td>${f4(q.margin)}</td><td>${q.none?'no rule fired':q.margin<.15?'uncertain':''}</td></tr>`).join('')}</table></div></div><div id="ex"></div>`}
function draw(){const{p,pr}=S.rec;plot($('#c1'),[{y:p.x,c:getComputedStyle(document.documentElement).getPropertyValue('--muted-2').trim()||'#8a98b3'}],{t0:S.t0});plot($('#c2'),[{y:p.norm,c:getComputedStyle(document.documentElement).getPropertyValue('--chart-line').trim()||'#e6ecf5'}],{t0:S.t0,marks:p.rows.map((r,i)=>({i:r.r,c:COL[E.CLS[pr[i].k]],t:E.CLS[pr[i].k],sel:i===S.sel}))});explain()}
const term=(m,[ci,t])=>`${FN[m.cols[ci]]} is ${E.TN[t]}`;
function explain(){if(!S.rec||!$('#ex'))return;const q=S.rec.pr[S.sel],m=S.M.mB;
 $('#ex').innerHTML=`<div class="card"><h2>Beat ${S.sel}: ${pill(E.CLS[q.k])} ${E.CN[E.CLS[q.k]]}</h2>${E.CLS.map((c,i)=>`<div class="row" style="margin:3px 0">${pill(c)}<div style="flex:1"><div class="bar" style="width:${100*q.sc[i]}%;background:${COL[c]}"></div></div><span class="mut" style="width:50px">${f4(q.sc[i])}</span></div>`).join('')}
 <div class="sc" style="margin-top:8px"><table><tr><th>Top fired rules</th><th>Class</th><th>Firing</th><th>Weight</th></tr>${q.fired.slice(0,6).map(x=>`<tr><td>IF ${x.r.ts.map(t=>term(m,t)).join(' AND ')}</td><td>${pill(E.CLS[x.r.cls])}</td><td>${f4(x.f)}</td><td>${f4(x.r.w)}</td></tr>`).join('')||'<tr><td colspan=4>No rule fired — default class N</td></tr>'}</table></div></div>`}
const A={
analyze:()=>{if(S.rec)wire();$('#go').onclick=()=>{S.seed=+$('#sd').value;const r=sample($('#sc').value,S.seed,+$('#nz').value);analyze(r.x,E.FS,r.ann,'synthetic · '+$('#sc').value);$('#res').innerHTML=resHTML();wire()};
 $('#dl').onclick=()=>{const r=sample($('#sc').value,+$('#sd').value,+$('#nz').value),a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['mV\n'+Array.from(r.x,v=>v.toFixed(4)).join('\n')],{type:'text/csv'}));a.download='sample_ecg_360hz.csv';a.click()};
 $('#fl').onchange=e=>{const f=e.target.files[0];if(!f)return;f.text().then(t=>{try{const x=parseCSV(t.replace(/^[^\d\-+.\n]*\n/,''));if(x.length<E.FS*3)throw Error('Need at least 3 seconds of signal');analyze(x,+$('#fs').value||360,null,'upload · '+f.name);$('#res').innerHTML=resHTML();wire();$('#msg').textContent=''}catch(err){$('#msg').textContent='⚠ '+err.message}})}},
features:()=>{if(!S.rec)return;const draw1=()=>{const b=S.rec.p.rows[S.sel].beat;plot($('#bc'),[{y:b,c:'#38bdf8'}],{win:b.length/E.FS,marks:[{i:E.PRE,c:'#f87171',t:'R'}]})};draw1();$('#bi').onchange=e=>{S.sel=Math.max(0,Math.min(S.rec.p.rows.length-1,+e.target.value|0));go('features')}},
rough:()=>{$('#rt').onclick=()=>{$('#rm').textContent='Training…';setTimeout(()=>{S.M=E.train({tol:+$('#tl').value,bins:+$('#bn').value});S.fz=0;if(S.rec)analyze(S.rec.p.x,E.FS,S.rec.ann,S.rec.label);go('rough')},30)}},
fuzzy:()=>{const m=S.M.mB,paint=()=>{const cv=$('#mc'),d=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;cv.width=w*d;cv.height=h*d;const g=cv.getContext('2d');g.scale(d,d);const c=m.cen[S.fz],sp=c[2]-c[0]||1,a=c[0]-sp*.6,z=c[2]+sp*.6,X=v=>(v-a)/(z-a)*w;
  ['#4ade80','#fbbf24','#f87171'].forEach((col,t)=>{g.strokeStyle=col;g.lineWidth=2;g.beginPath();for(let i=0;i<=200;i++){const v=a+(z-a)*i/200,y=h-6-E.mu(c,t,v)*(h-16);i?g.lineTo(X(v),y):g.moveTo(X(v),y)}g.stroke();g.fillStyle=col;g.fillText(E.TN[t],X(c[t])-8,12)});
  if(S.rec){const v=S.rec.p.rows[S.sel].f[m.cols[S.fz]];g.setLineDash([4,4]);g.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--chart-cursor').trim()||'#fff';g.beginPath();g.moveTo(X(v),0);g.lineTo(X(v),h);g.stroke();g.setLineDash([])}},
 rl=()=>{const k=+$('#rc').value;$('#rl').innerHTML=`<table><tr><th>Class</th><th>IF</th><th>Weight</th></tr>${m.rules.filter(r=>k<0||r.cls===k).map(r=>`<tr><td>${pill(E.CLS[r.cls])}</td><td>${r.ts.map(t=>term(m,t)).join(' AND ')}</td><td>${f4(r.w)}</td></tr>`).join('')}</table>`};
 paint();rl();$('#ff').onchange=e=>{S.fz=+e.target.value;paint()};$('#rc').onchange=rl}};
// ---------- Phase 4: local Python API integration ----------
const API_BASE='/api';
S.backendHealth=null;S.backendDataset=null;S.backendExperiments=[];S.backendJob=null;S.backendResult=null;S.backendRpeakResult=null;S.backendAuditResult=null;S.backendError='';
S.inferenceModels=[];S.inferenceResult=null;S.inferenceExplanation=null;S.inferenceError='';S.inferenceBusy=false;S.inferenceSelectedBeat=0;S.inferenceLoadingModels=false;
const escHtml=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtPct=v=>(v!==null&&v!==undefined&&Number.isFinite(Number(v)))?(100*Number(v)).toFixed(1)+'%':'—';
async function apiRequest(path,options={}){
 const response=await fetch(API_BASE+path,{headers:{'Content-Type':'application/json',...(options.headers||{})},...options});
 let body={};try{body=await response.json()}catch(_){body={}}
 if(!response.ok)throw new Error(body.detail||body.message||`API request failed (${response.status})`);
 return body;
}
function backendRender(){
 const h=S.backendHealth,d=S.backendDataset,j=S.backendJob,rows=S.backendExperiments||[],result=S.backendResult;
 const status=h?'Connected':'Not connected',busy=!!j&&['queued','running'].includes(j.status); 
 const deps=h?.dependencies||{};
 const depRows=Object.entries(deps).map(([n,ok])=>`<tr><td>${escHtml(n.replaceAll('_',' '))}</td><td><span class="pill" style="background:${ok?'#b9f3d6':'#ffe2ad'}">${ok?'Ready':'Missing'}</span></td></tr>`).join('');
 const jobHtml=j?`<div class="card"><div class="row"><h2 style="margin:0">Current task</h2><span class="pill" id="backend-job-status" style="background:${j.status==='failed'?'#ffd7dc':j.status==='completed'?'#b9f3d6':'#dbeafe'}">${escHtml(j.status)}</span></div><p id="backend-job-message" class="mut">${escHtml(j.message||'Queued')}</p><div class="bar" style="width:${Math.max(0,Math.min(100,Number(j.progress||0)*100))}%;height:9px"></div><p class="mut">${Math.round(Number(j.progress||0)*100)}% · ${escHtml(j.kind||'')}</p>${j.error?`<p style="color:var(--red)">${escHtml(j.error)}</p>`:''}${j.result?`<pre class="api-json">${escHtml(JSON.stringify(j.result,null,2))}</pre>`:''}</div>`:'';
 const auditReport=S.backendAuditResult?.audit_id?`<div class="card"><h2>Dataset audit: ${escHtml(S.backendAuditResult.audit_id)}</h2><div class="grid">${kpi(S.backendAuditResult.n_records_ok??'—','records readable')}${kpi(S.backendAuditResult.n_records_failed??'—','records failed')}${kpi(S.backendAuditResult.patient_disjoint===true?'Yes':S.backendAuditResult.patient_disjoint===false?'No':'—','patient-disjoint')}</div><div class="row">${(S.backendAuditResult.artifacts||[]).map(a=>`<a class="btn" href="/api/audits/${encodeURIComponent(S.backendAuditResult.audit_id)}/artifacts/${encodeURIComponent(a)}">${escHtml(a)}</a>`).join('')}</div>${S.backendAuditResult.warnings?.length?`<p class="mut">${escHtml(S.backendAuditResult.warnings.join(' · '))}</p>`:''}</div>`:'';
 const history=rows.length?`<div class="sc"><table><tr><th>Run</th><th>Created (UTC)</th><th>Train</th><th>Test</th><th>Best model</th><th>Macro-F1</th></tr>${rows.map(r=>`<tr><td><button class="btn" data-run-id="${escHtml(r.run_id)}">${escHtml(r.run_id)}</button></td><td>${escHtml((r.created_at_utc||'').replace('T',' ').replace('Z',''))}</td><td>${escHtml(r.n_train??'—')}</td><td>${escHtml(r.n_test??'—')}</td><td>${escHtml(r.best_model?.model||'—')}</td><td>${fmtPct(r.best_model?.macro_f1)}</td></tr>`).join('')}</table></div>`:'<p class="mut">No saved MIT-BIH experiment runs found yet.</p>';
 const comparison=result?.model_comparison?.length?`<div class="sc"><table><tr><th>Model</th><th>Accuracy</th><th>Balanced accuracy</th><th>Macro-F1</th><th>Weighted-F1</th><th>Features used</th></tr>${result.model_comparison.map(r=>`<tr><td>${escHtml(r.model)}</td><td>${fmtPct(r.accuracy)}</td><td>${fmtPct(r.balanced_accuracy)}</td><td>${fmtPct(r.macro_f1)}</td><td>${fmtPct(r.weighted_f1)}</td><td>${escHtml(r.n_features_used??'—')}</td></tr>`).join('')}</table></div>`:'';
 const reliabilityRows=Object.entries(result?.metrics||{}).filter(([,m])=>m?.score_reliability?.metrics).map(([name,m])=>({name,...m.score_reliability.metrics,calibrationStatus:m.score_calibration?.status||'not fitted',temperature:m.score_calibration?.temperature??1}));
 const reliabilityHtml=reliabilityRows.length?`<div class="card"><h2>Held-out score reliability diagnostics</h2><p class="mut">Where available, scalar temperature scaling is fitted using grouped out-of-fold training-record scores with fold-specific training-only imputation. Untouched test labels are used only for Brier, log-loss and ECE diagnostics below. A skipped transform leaves scores unscaled; this internal process is not external or clinical validation.</p><div class="sc"><table><tr><th>Model</th><th>Train OOF calibration</th><th>Temperature</th><th>Held-out Brier</th><th>Log loss</th><th>Top-label ECE</th></tr>${reliabilityRows.map(r=>`<tr><td>${escHtml(r.name)}</td><td>${escHtml(r.calibrationStatus)}</td><td>${Number(r.temperature).toFixed(4)}</td><td>${Number(r.multiclass_brier_score).toFixed(4)}</td><td>${Number(r.multiclass_log_loss).toFixed(4)}</td><td>${Number(r.top_label_expected_calibration_error).toFixed(4)}</td></tr>`).join('')}</table></div></div>`:'';
 const patientRows=Object.entries(result?.metrics||{}).filter(([,m])=>m?.patient_cluster_bootstrap).map(([name,m])=>{const b=m.patient_cluster_bootstrap||{},a=b.accuracy_95ci||[],f=b.macro_f1_all_aami_95ci||[];return{name,n:b.n_patients??'—',status:b.status||'not available',accuracy:a.length===2&&a.every(v=>v!=null)?`${Number(a[0]).toFixed(3)}–${Number(a[1]).toFixed(3)}`:'Not estimated',macroF1:f.length===2&&f.every(v=>v!=null)?`${Number(f[0]).toFixed(3)}–${Number(f[1]).toFixed(3)}`:'Not estimated'}});
 const patientHtml=patientRows.length?`<div class="card"><h2>Patient-level robustness</h2><p class="mut">The bootstrap resamples whole held-out patient groups rather than individual beats. Intervals describe variability within this test split; they are not external validation. See per_patient_model_metrics.csv for individual patient rows.</p><div class="sc"><table><tr><th>Model</th><th>Patients</th><th>Status</th><th>Accuracy 95% interval</th><th>Macro-F1 (all AAMI) 95% interval</th></tr>${patientRows.map(r=>`<tr><td>${escHtml(r.name)}</td><td>${escHtml(r.n)}</td><td>${escHtml(r.status)}</td><td>${escHtml(r.accuracy)}</td><td>${escHtml(r.macroF1)}</td></tr>`).join('')}</table></div></div>`:'';
 const artifacts=result?.artifacts?.length?`<div class="row">${result.artifacts.map(a=>`<a class="btn" href="/api/experiments/${encodeURIComponent(result.run_id)}/artifacts/${encodeURIComponent(a)}" target="_blank" rel="noopener">${escHtml(a)}</a>`).join('')}</div>`:'';
 return `<div class="card"><div class="row"><h2 style="margin:0">Python research service</h2><span class="pill" style="background:${h?'#b9f3d6':'#ffe2ad'}">${status}</span><button class="btn" id="backend-refresh">Refresh status</button></div><p class="mut">This browser workspace connects to the Python research engine through a local FastAPI service. Start PulseLab with <code>py -m uvicorn server:app --host 127.0.0.1 --port 8000</code> from the project root.</p>${S.backendError?`<p style="color:var(--red)">${escHtml(S.backendError)}</p>`:''}<div class="grid">${kpi(h?escHtml(h.version):'—','API version')}${kpi(h?escHtml(h.python_version):'—','Python runtime')}${kpi(h?escHtml(h.dataset_records_available):'—','MIT-BIH records available')}${kpi(d?`${d.available_requested}/${d.train_records_requested+d.test_records_requested}`:'—','Selected records present')}</div>${h?`<p class="mut">API ready. Dataset root: <code>${escHtml(h.dataset_root)}</code></p>`:'<p class="mut">The browser is currently running without the integrated backend. This is expected when using Python\'s basic http.server; switch to the Uvicorn command above.</p>'}</div>
 <div class="card"><h2>Local dependency check</h2><div class="sc"><table><tr><th>Dependency</th><th>Status</th></tr>${depRows||'<tr><td colspan="2">Connect to the API to inspect dependencies.</td></tr>'}</table></div><p class="mut">WFDB and PyWavelets are required for MIT-BIH data loading and feature extraction. FastAPI and Uvicorn power this integrated interface.</p></div>
 <div class="card"><h2>MIT-BIH dataset status</h2><div class="grid">${kpi(d?d.train_records_requested:'—','training records requested')}${kpi(d?d.test_records_requested:'—','test records requested')}${kpi(d?d.available_requested:'—','selected records available')}${kpi(d?d.missing_records.length:'—','selected records missing')}</div><p class="mut">${escHtml(d?.note||'Dataset status is unavailable until the API connects.')}</p>${d?.missing_records?.length?`<p class="mut">Missing: ${escHtml(d.missing_records.join(', '))}</p>`:''}<div class="row"><label>Split <select id="api-split"><option value="patient-disjoint">Patient-disjoint (recommended)</option><option value="de-chazal-record-wise">Historical record-wise (overlap warning)</option></select></label><label><input type="checkbox" id="api-quick"> Quick subset (engineering check only)</label><label><input type="checkbox" id="api-download"> Download missing records</label><button class="btn" id="api-audit" ${!h||busy?'disabled':''}>Run dataset audit</button></div></div>
 <div class="card"><h2>Independent R-peak validation</h2><p class="mut">Evaluate the automatic detector against the MIT-BIH reference annotations. These metrics describe peak detection, not arrhythmia classification.</p><div class="row"><label>Matching tolerance (ms) <input id="api-tolerance" type="number" min="25" max="150" step="5" value="75"></label><button class="btn" id="api-rpeak" ${!h||busy?'disabled':''}>Validate test-record R-peaks</button></div>${S.backendRpeakResult?`<div class="grid">${kpi(S.backendRpeakResult.aggregate?.micro_sensitivity==null?'—':fmtPct(S.backendRpeakResult.aggregate.micro_sensitivity),'R-peak sensitivity')}${kpi(S.backendRpeakResult.aggregate?.micro_ppv==null?'—':fmtPct(S.backendRpeakResult.aggregate.micro_ppv),'R-peak PPV')}${kpi(S.backendRpeakResult.aggregate?.micro_f1==null?'—':fmtPct(S.backendRpeakResult.aggregate.micro_f1),'R-peak F1')}${kpi(S.backendRpeakResult.aggregate?.n_records??'—','records evaluated')}</div><p class="mut">${escHtml(S.backendRpeakResult.interpretation||'R-peak detection metrics only.')}</p><div class="row">${['rpeak_validation.json','rpeak_metrics_per_record.csv'].map(a=>`<a class="btn" href="/api/rpeak-validations/${encodeURIComponent(S.backendRpeakResult.validation_id)}/artifacts/${encodeURIComponent(a)}">${a}</a>`).join('')}</div>`:''}</div>
 <div class="card"><h2>Launch MIT-BIH experiment</h2><p class="mut">This starts the Python experiment pipeline. The quick subset is for engineering checks and may not include every AAMI class; use the full patient-disjoint split for the main research evaluation.</p><div class="row"><label>Peak source <select id="api-peak"><option value="detected">Automatic detected peaks (default)</option><option value="annotations">Annotation-guided benchmark</option></select></label><label><input type="checkbox" id="api-baselines" checked> Include RF / SVM baselines</label><label><input type="checkbox" id="api-save-models" checked> Save model artifacts</label></div><div class="row"><button class="btn pri" id="api-run" ${!h||busy?'disabled':''}>Run selected experiment</button><span class="mut">The dataset must be present locally or downloaded by the job.</span></div></div>
 ${jobHtml}${auditReport}
 <div class="card"><div class="row"><h2 style="margin:0">Saved research runs</h2><button class="btn" id="api-history-refresh">Refresh results</button></div>${history}</div>
 ${result?`<div class="card"><div class="row"><h2 style="margin:0">Experiment: ${escHtml(result.run_id)}</h2><span class="pill" style="background:#b9f3d6">Saved result</span></div><div class="grid">${kpi(result.n_train??'—','training beats')}${kpi(result.n_test??'—','test beats')}${kpi(result.n_test_patients??'—','held-out patients')}${kpi(result.n_features??'—','input features')}${kpi(result.metadata?.patient_disjoint===true?'Yes':result.metadata?.patient_disjoint===false?'No':'—','patient-disjoint')}</div><h2>Model comparison</h2>${comparison||'<p class="mut">No comparison table is available for this run.</p>'}${patientHtml}${reliabilityHtml}<h2 style="margin-top:18px">Downloadable artifacts</h2>${artifacts}${result.warnings?.length?`<p class="mut">Warnings: ${escHtml(result.warnings.join(' · '))}</p>`:''}<p class="mut">${escHtml(result.disclaimer||'Research use only; not clinically validated.')}</p></div>`:''}
 <div class="card"><h2>Important separation of results</h2><p class="mut">The existing ECG Analysis, Feature Extraction, Rough Set, Fuzzy and Model Evaluation pages remain a synthetic browser demonstration. This Research Backend page is where real MIT-BIH dataset audits and Python experiment results are managed. A successful API connection does not mean the dataset has been downloaded, and a synthetic benchmark score is never relabelled as MIT-BIH performance.</p><p class="mut"><b>Security:</b> this local research API has no authentication. Keep it bound to 127.0.0.1 and do not expose it to a public network.</p></div>`;
}
async function refreshBackend(){
 S.backendError='';
 try{
  const [health,dataset,history]=await Promise.all([apiRequest('/health'),apiRequest(`/dataset/status?quick=${$('#api-quick')?.checked??false}&split_protocol=${encodeURIComponent($('#api-split')?.value||'patient-disjoint')}`),apiRequest('/experiments?limit=20')]);
  S.backendHealth=health;S.backendDataset=dataset;S.backendExperiments=history.experiments||[];
 }catch(err){S.backendHealth=null;S.backendError=err.message;S.backendDataset=null;}
 updateEnvironmentChip();
 if(S.view==='backend'){ keepForm(BACKEND_FORM,()=>{$('#v').innerHTML=R.backend()}); wireBackend(); }
}
function wireBackend(){
 ['api-quick','api-split'].forEach(id=>{const el=$('#'+id);if(el)el.onchange=refreshBackend});
 const refresh=$('#backend-refresh');if(refresh)refresh.onclick=refreshBackend;
 const historyRefresh=$('#api-history-refresh');if(historyRefresh)historyRefresh.onclick=refreshBackend;
 const audit=$('#api-audit');if(audit)audit.onclick=()=>startBackendJob('audit');
 const rpeak=$('#api-rpeak');if(rpeak)rpeak.onclick=()=>startBackendJob('rpeak');
 const run=$('#api-run');if(run)run.onclick=()=>startBackendJob('experiment');
 document.querySelectorAll('[data-run-id]').forEach(btn=>btn.onclick=()=>loadBackendRun(btn.dataset.runId));
}
async function startBackendJob(kind){
 const split=$('#api-split')?.value||'patient-disjoint',quick=$('#api-quick')?.checked??true,downloadMissing=$('#api-download')?.checked??false;
 const body=kind==='rpeak'?{split_protocol:split,quick,download_missing:downloadMissing,tolerance_ms:Number($('#api-tolerance')?.value||75)}:kind==='audit'?{split_protocol:split,quick,download_missing:downloadMissing}:{split_protocol:split,quick,download_missing:downloadMissing,peak_source:$('#api-peak')?.value||'detected',include_baselines:$('#api-baselines')?.checked??true,save_models:$('#api-save-models')?.checked??true};
 const button=kind==='audit'?$('#api-audit'):kind==='rpeak'?$('#api-rpeak'):$('#api-run');if(button)button.disabled=true;
 try{const route=kind==='audit'?'/dataset/audits':kind==='rpeak'?'/rpeak-validations':'/experiments';const created=await apiRequest(route,{method:'POST',body:JSON.stringify(body)});S.backendJob={job_id:created.job_id,kind:kind==='audit'?'dataset_audit':kind==='rpeak'?'rpeak_validation':'experiment',status:'queued',progress:0,message:'Queued'};S.backendResult=null;renderBackendView();pollBackendJob(created.job_id);}
 catch(err){S.backendError=err.message;renderBackendView();}
 finally{if(button)button.disabled=false;}
}
function renderBackendView(){if(S.view==='backend'){ keepForm(BACKEND_FORM,()=>{$('#v').innerHTML=R.backend()});wireBackend();}}
async function pollBackendJob(jobId){
 try{
  const job=await apiRequest(`/jobs/${encodeURIComponent(jobId)}`);S.backendJob=job;
  const msg=$('#backend-job-message');if(msg)msg.textContent=`${job.message||job.status} · ${Math.round((job.progress||0)*100)}%`;
  const badge=$('#backend-job-status');if(badge){badge.textContent=job.status;badge.style.background=job.status==='failed'?'#ffd7dc':job.status==='completed'?'#b9f3d6':'#dbeafe';}
  const panel=msg?.closest('.card');if(panel){const bar=panel.querySelector('.bar');if(bar)bar.style.width=`${Math.max(0,Math.min(100,(job.progress||0)*100))}%`}
  if(job.status==='completed'){
   if(job.kind==='experiment'&&job.result?.run_id)await loadBackendRun(job.result.run_id,false);
   else if(job.kind==='rpeak_validation'&&job.result?.validation_id)await loadRpeakSummary(job.result.validation_id);
   else S.backendAuditResult=job.result;
   await refreshBackend();return;
  }
  if(job.status==='failed'){renderBackendView();return;}
  if(job.status==='queued'||job.status==='running')setTimeout(()=>pollBackendJob(jobId),1200);
 }catch(err){S.backendError=err.message;renderBackendView();}
}
async function loadBackendRun(runId,rerender=true){
 try{S.backendResult=await apiRequest(`/experiments/${encodeURIComponent(runId)}/summary`);if(rerender)renderBackendView();}
 catch(err){S.backendError=err.message;renderBackendView();}
}
async function loadRpeakSummary(validationId){
 try{S.backendRpeakResult=await apiRequest(`/rpeak-validations/${encodeURIComponent(validationId)}/summary`);}
 catch(err){S.backendError=err.message;}
}
R.backend=backendRender;
A.backend=()=>{wireBackend();refreshBackend();};

// ---------- Phase 6: explainable saved-model ECG inference ----------
function inferenceRender(){
 const models=S.inferenceModels||[];
 const selectedId=$('#inference-model')?.value||models[0]?.model_id||'';
 const result=S.inferenceResult,inferenceExplanation=S.inferenceExplanation;
 const options=models.map(m=>`<option value="${escHtml(m.model_id)}" ${m.model_id===selectedId?'selected':''}>${escHtml(m.model_name)} · ${escHtml(m.run_id)} · ${m.n_channels} ch · ${m.calibration_status==='fitted'?'OOF temp-scaled':'uncalibrated'}</option>`).join('');
 const emptyModels=`<div class="inference-empty"><strong>No saved MIT-BIH models are available yet.</strong><p class="mut">Run a MIT-BIH experiment in Research Backend with “Save model artifacts” enabled. Only trained artifacts with a compatible feature schema appear here.</p><button class="btn" id="inference-go-backend">Open Research Backend</button></div>`;
 const modelOptions=models.length?`<div class="grid"><div class="kpi"><b>${models.length}</b><span>saved models available</span></div><div class="kpi"><b>${escHtml(models[0].feature_schema_version||'—')}</b><span>feature schema</span></div></div>`:emptyModels;
 const dependencyNotice=S.backendHealth&&!S.backendHealth.inference_upload_ready?`<div class="inference-notice"><strong>Inference dependencies are incomplete.</strong><p class="mut">Install the project requirements with <code>pip install -r requirements.txt</code>, then restart the local API. Check the dependency list in Research Backend.</p></div>`:'';
 const resultHtml=result?(()=>{
  const p=result.predictions||[], total=p.length||1;
  const counts=result.class_counts||{};
  const dist=Object.entries(counts).map(([label,count])=>`<div class="inference-class-row"><div class="inference-class-label">${pill(label)}<span>${escHtml(count)}</span></div><div class="inference-class-track"><i style="width:${Math.min(100,100*Number(count)/total)}%"></i></div></div>`).join('');
  const rows=p.map((b,i)=>`<tr class="inference-beat-row ${i===S.inferenceSelectedBeat?'selected':''}" data-inference-beat="${i}" tabindex="0" aria-label="Explain beat ${i+1}"><td>${i+1}</td><td>${Number(b.time_seconds).toFixed(3)}</td><td>${escHtml(b.sample_index_360hz)}</td><td>${pill(b.predicted_class)}</td><td>${b.top_model_score==null?'—':Number(b.top_model_score).toFixed(3)}</td><td>${b.score_margin==null?'—':Number(b.score_margin).toFixed(3)}</td><td>${b.normalised_score_entropy==null?'—':Number(b.normalised_score_entropy).toFixed(3)}</td><td>${b.n_features_outside_training_range==null?'—':escHtml(b.n_features_outside_training_range)}</td><td>${b.low_score_separation||b.high_score_entropy?'<span class="pill warning">Review score</span>':'—'}</td></tr>`).join('');
  const artifacts=(result.artifacts||[]).filter(a=>a!=='beat_features.csv').map(a=>`<a class="btn" href="/api/inferences/${encodeURIComponent(result.inference_id)}/artifacts/${encodeURIComponent(a)}">${escHtml(a)}</a>`).join('');
  const warnings=(result.warnings||[]).map(w=>`<li>${escHtml(w)}</li>`).join('');
  const drift=result.feature_drift;
  const driftHtml=drift?.available?`<div class="card"><div class="row"><h2 style="margin:0">Training-range drift check</h2>${drift.high_drift_flag?'<span class="pill warning">Review recommended</span>':'<span class="pill">No high-drift flag</span>'}</div><p class="mut">${escHtml(drift.method)} · ${drift.n_beats} beats · ${drift.n_features_compared} features</p><div class="grid">${kpi(fmtPct(drift.outside_value_fraction),'feature values outside training q01–q99')}${kpi(drift.beats_with_any_out_of_range_feature,'beats with at least one flagged feature')}</div><p class="mut">This is a descriptive covariate-range check, not proof of a different population or a determination that the signal is invalid. It cannot establish clinical suitability.</p>${(result.artifacts||[]).filter(a=>a.startsWith('feature_drift_')).map(a=>`<a class="btn" href="/api/inferences/${encodeURIComponent(result.inference_id)}/artifacts/${encodeURIComponent(a)}">${escHtml(a)}</a>`).join(' ')}</div>`: (result.feature_drift===null?'<div class="card"><h2>Training-range drift check</h2><p class="mut">This saved model does not contain a Phase 6 training feature reference. Re-run training with Phase 6 to enable this diagnostic.</p></div>':'');
  return `<div class="card"><div class="row"><h2 style="margin:0">Inference result</h2><span class="pill" style="background:#b9f3d6">Completed</span></div><div class="grid">${kpi(escHtml(result.n_predictions),'complete beats classified')}${kpi(escHtml(result.preprocessing?.n_rpeaks_detected??'—'),'R-peaks detected')}${kpi(`${Number(result.input?.duration_seconds||0).toFixed(1)} s`,'signal duration')}${kpi(escHtml(result.input?.sampling_rate_input_hz??'—')+' Hz','input sampling rate')}</div><p class="mut">${escHtml(result.input_filename)} · ${escHtml(result.model?.model_name)} · run ${escHtml(result.model?.run_id)}</p><p class="mut">Scores: ${escHtml(result.score_interpretation)}. ${escHtml(result.score_note||'Score concentration is descriptive only; these are not clinical risk estimates.')}</p>${dist?`<h3>Predicted class distribution</h3><div class="inference-class-distribution">${dist}</div>`:''}${warnings?`<div class="inference-notice"><strong>Interpretation notes</strong><ul>${warnings}</ul></div>`:''}<div class="row">${artifacts}</div><p class="mut">The full uploaded ECG and selected-beat waveform previews are not written to disk. Beat features, predictions and summary metadata are saved locally under <code>experiments/${escHtml(result.inference_id)}/</code>. Selected-beat waveforms remain in a small process-memory cache only.</p></div>${driftHtml}<div class="card"><h2>Beat-by-beat model output</h2><p class="mut">Select a row to inspect its normalised ECG beat waveform, model-specific evidence and single-feature reference sensitivity. Review indicators are not diagnoses.</p><div class="sc"><table><thead><tr><th>Beat</th><th>Time (s)</th><th>Sample @ 360 Hz</th><th>Prediction</th><th>Top score</th><th>Score margin</th><th>Norm. entropy</th><th>Out-of-range features</th><th>Review</th></tr></thead><tbody>${rows}</tbody></table></div></div>${inferenceExplanation?renderInferenceExplanation(inferenceExplanation):'<div class="card"><h2>Beat explanation</h2><p class="mut">Choose a beat above to inspect the decision context.</p></div>'}`;
 })():'';
 return `<div class="card"><div class="row"><h2 style="margin:0">Saved model inference</h2><button class="btn" id="inference-refresh-models" ${S.inferenceLoadingModels?'disabled':''}>${S.inferenceLoadingModels?'Refreshing…':'Refresh models'}</button></div><p class="mut">This workspace runs saved Python models trained from MIT-BIH experiments. It is separate from the synthetic browser demonstration and requires the local FastAPI service.</p>${S.inferenceError?`<div class="inference-error">${escHtml(S.inferenceError)}</div>`:''}${dependencyNotice}${modelOptions}<div class="inference-form"><label for="inference-model">Trained model</label><select id="inference-model" ${!models.length?'disabled':''}>${options||'<option value="">No compatible saved model</option>'}</select><div class="inference-model-meta" id="inference-model-meta">${selectedId&&models.find(m=>m.model_id===selectedId)?(()=>{const m=models.find(x=>x.model_id===selectedId);return `${m.n_features} features · ${m.n_channels} ECG channel(s) · ${escHtml(m.peak_source||'unknown')} peak training mode · ${m.patient_disjoint?'patient-disjoint':'not patient-disjoint'} · ${m.calibration_status==='fitted'?`grouped OOF temperature ${Number(m.score_temperature||1).toFixed(3)}`:'no OOF calibration'}`})():''}</div><label for="inference-file">ECG signal file (.csv or .txt)</label><input id="inference-file" type="file" accept=".csv,.txt,text/csv,text/plain" ${!models.length?'disabled':''}><p class="mut">Use numeric signal columns only, with an optional header. Common time/index columns are skipped when named. If the selected model needs two channels, supply two ECG lead columns. Keep the same lead order used by the experiment.</p><label for="inference-fs">Input sampling rate (Hz)</label><input id="inference-fs" type="number" min="100" max="2000" step="1" value="360" ${!models.length?'disabled':''}><div class="row"><button class="btn pri" id="inference-run" ${!models.length||S.inferenceBusy?'disabled':''}>${S.inferenceBusy?'Processing ECG…':'Run Python model inference'}</button><span class="mut">Upload limit: 20 MB · 3 seconds to 10 minutes</span></div></div></div>${resultHtml}<div class="card"><h2>Research and safety boundary</h2><p class="mut">Predictions are produced by a saved research model after automatic R-peak detection and the Python feature pipeline. Training metrics do not guarantee performance on this uploaded signal. The system is an academic prototype, not a medical device; do not use outputs for diagnosis, treatment or urgent decisions.</p></div>`;
}
function renderInferenceWaveform(waveform){
 if(!waveform?.available||!Array.isArray(waveform.samples)||waveform.samples.length<2){
  return `<div class="inference-waveform-empty"><strong>Beat waveform preview unavailable</strong><p class="mut">${escHtml(waveform?.message||'The selected waveform preview is no longer in the process-memory cache. Re-run inference to create a fresh preview.')}</p></div>`;
 }
 const values=waveform.samples.map(Number).filter(Number.isFinite);
 if(values.length<2)return '';
 let min=Math.min(...values),max=Math.max(...values);if(Math.abs(max-min)<1e-9){max+=1;min-=1}
 const width=760,height=190,left=36,right=14,top=16,bottom=30,plotW=width-left-right,plotH=height-top-bottom;
 const pad=(max-min)*0.08;min-=pad;max+=pad;
 const points=values.map((value,index)=>`${(left+index*(plotW/(values.length-1))).toFixed(2)},${(top+(max-value)*(plotH/(max-min))).toFixed(2)}`).join(' ');
 const rX=left+(Number(waveform.r_peak_sample_in_window||0)/Math.max(values.length-1,1))*plotW;
 const start=Number(waveform.time_ms?.[0]??-250),end=Number(waveform.time_ms?.[waveform.time_ms.length-1]??400);
 return `<div class="inference-waveform-card"><div class="row"><strong>Selected heartbeat waveform</strong><span class="mut">${escHtml(waveform.amplitude_units||'normalised units')}</span></div><p class="mut">Time is relative to the detected R peak. This is the preprocessed, R-amplitude-normalised beat window used to generate features; it is not a diagnostic annotation.</p><div class="inference-waveform-scroll"><svg class="inference-waveform-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Normalised ECG beat waveform with R peak marker"><line x1="${left}" y1="${top+plotH/2}" x2="${width-right}" y2="${top+plotH/2}" class="wave-zero"/><line x1="${rX.toFixed(2)}" y1="${top}" x2="${rX.toFixed(2)}" y2="${top+plotH}" class="wave-r-marker"/><polyline points="${points}" class="wave-trace"/><text x="${left}" y="${height-7}" class="wave-axis-label">${start.toFixed(0)} ms</text><text x="${rX.toFixed(2)}" y="${height-7}" text-anchor="middle" class="wave-axis-label">R peak · 0 ms</text><text x="${width-right}" y="${height-7}" text-anchor="end" class="wave-axis-label">${end.toFixed(0)} ms</text></svg></div></div>`;
}
function renderInferenceExplanation(payload){
 const e=payload.explanation||{}, details=e.details||{};
 const waveformHtml=renderInferenceWaveform(payload.waveform);
 const classScores=payload.class_scores||{};
 const scoreRows=Object.entries(classScores).map(([label,value])=>`<div class="inference-score-row"><span>${pill(label)}</span><strong>${Number(value).toFixed(4)}</strong><div class="inference-score-track"><i style="width:${Math.max(0,Math.min(100,100*Number(value)))}%"></i></div></div>`).join('');
 const rules=(details.top_rules||[]).map((rule,i)=>`<div class="inference-rule"><div><strong>Rule ${i+1}</strong><span>weight ${Number(rule.weight||0).toFixed(3)} · firing ${Number(rule.firing_strength||0).toFixed(3)}</span></div><p>${(rule.antecedents||[]).map(a=>`${escHtml(a.feature)} is ${escHtml(a.term)}`).join(' AND ')||'No antecedents reported'}</p></div>`).join('');
 const ablation=details.local_feature_ablation||{};
 const attributionRows=(ablation.features||[]).map(item=>{
  const max=Math.max(...(ablation.features||[]).map(x=>Math.abs(Number(x.score_delta||0))),1e-9);
  const width=Math.min(100,100*Math.abs(Number(item.score_delta||0))/max);
  const positive=Number(item.score_delta||0)>=0;
  return `<div class="explainability-row"><div class="explainability-row-head"><strong>${escHtml(item.feature)}</strong><span>${Number(item.score_delta)>=0?'+':''}${Number(item.score_delta).toFixed(4)}</span></div><div class="explainability-track"><i class="${positive?'positive':'negative'}" style="width:${width}%"></i></div><p class="mut">Observed ${Number(item.observed_value).toPrecision(4)} · training median ${Number(item.training_median_reference).toPrecision(4)} · reference-replaced score ${Number(item.score_after_reference_replacement).toFixed(4)}</p></div>`;
 }).join('');
 const global=e.global_explainability||{};
 const globalRows=(global.features||[]).slice(0,10).map(x=>`<tr><td>${escHtml(x.feature)}</td><td>${Number(x.importance).toFixed(4)}</td></tr>`).join('');
 const message=e.available?`<p class="mut">This panel combines model-specific rule evidence (when available) with a local sensitivity analysis.</p>`:`<p class="mut">${escHtml(e.message||'A detailed explanation is not available for this model.')}</p>`;
 const limitation=(e.limitations||ablation.limitations||[]).map(x=>`<li>${escHtml(x)}</li>`).join('');
 return `<div class="card" id="inference-explanation"><div class="row"><h2 style="margin:0">Beat ${Number(payload.beat_index)+1} explanation</h2>${pill(payload.predicted_class)}</div>${message}${waveformHtml}<h3>Class score breakdown</h3><div class="inference-score-list">${scoreRows||'<p class="mut">This estimator did not return class scores.</p>'}</div>${rules?`<h3>Strongest fired fuzzy rules</h3>${rules}`:''}${attributionRows?`<h3>Local feature sensitivity</h3><p class="mut">${escHtml(ablation.method||'Single-feature reference ablation')}. Positive deltas mean the observed feature value raises the predicted-class score relative to replacing only that feature with its training median. Negative deltas mean it lowers that score.</p><div class="explainability-list">${attributionRows}</div>`:''}${details.memberships?`<details class="inference-details"><summary>Fuzzy membership values</summary><div class="sc"><table><tr><th>Feature</th><th>Low</th><th>Medium</th><th>High</th></tr>${Object.entries(details.memberships).slice(0,30).map(([name,m])=>`<tr><td>${escHtml(name)}</td><td>${Number(m.Low||0).toFixed(3)}</td><td>${Number(m.Medium||0).toFixed(3)}</td><td>${Number(m.High||0).toFixed(3)}</td></tr>`).join('')}</table></div></details>`:''}${globalRows?`<details class="inference-details"><summary>Global model feature summary</summary><p class="mut">${escHtml(global.method||'Model feature summary')}</p><div class="sc"><table><tr><th>Feature</th><th>Importance / rule usage</th></tr>${globalRows}</table></div></details>`:''}${limitation?`<div class="inference-notice"><strong>Explanation limitations</strong><ul>${limitation}</ul></div>`:''}</div>`;
}
async function refreshInferenceModels(){
 S.inferenceLoadingModels=true;S.inferenceError='';
 if(S.view==='inference'){keepForm(INFERENCE_FORM,()=>{$('#v').innerHTML=inferenceRender()});wireInference()}
 try{const [response,health]=await Promise.all([apiRequest('/models'),apiRequest('/health')]);S.inferenceModels=response.models||[];S.backendHealth=health;}
 catch(err){S.inferenceError=`Could not load saved models: ${err.message}. Start PulseLab with the Uvicorn command from the project root.`;}
 finally{S.inferenceLoadingModels=false;updateEnvironmentChip();if(S.view==='inference'){keepForm(INFERENCE_FORM,()=>{$('#v').innerHTML=inferenceRender()});wireInference()}}
}
function wireInference(){
 const refresh=$('#inference-refresh-models');if(refresh)refresh.onclick=refreshInferenceModels;
 const back=$('#inference-go-backend');if(back)back.onclick=()=>go('backend');
 const model=$('#inference-model');if(model)model.onchange=()=>{const m=S.inferenceModels.find(x=>x.model_id===model.value);const meta=$('#inference-model-meta');if(meta&&m)meta.textContent=`${m.n_features} features · ${m.n_channels} ECG channel(s) · ${m.peak_source||'unknown'} peak training mode · ${m.patient_disjoint?'patient-disjoint':'not patient-disjoint'} · ${m.calibration_status==='fitted'?`grouped OOF temperature ${Number(m.score_temperature||1).toFixed(3)}`:'no OOF calibration'}`};
 const run=$('#inference-run');if(run)run.onclick=submitInference;
 document.querySelectorAll('[data-inference-beat]').forEach(row=>{const load=()=>loadInferenceExplanation(Number(row.dataset.inferenceBeat));row.onclick=load;row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();load()}}});
}
async function submitInference(){
 const file=$('#inference-file')?.files?.[0],modelId=$('#inference-model')?.value,model=S.inferenceModels.find(x=>x.model_id===modelId),fs=Number($('#inference-fs')?.value||360);
 S.inferenceError='';S.inferenceExplanation=null;
 if(!model){S.inferenceError='Select a saved MIT-BIH model first.';renderInferencePage();return}
 if(!file){S.inferenceError='Choose an ECG CSV/TXT file first.';renderInferencePage();return}
 if(!Number.isFinite(fs)||fs<100||fs>2000){S.inferenceError='Sampling rate must be between 100 and 2000 Hz.';renderInferencePage();return}
 if(file.size>20*1024*1024){S.inferenceError='The selected file exceeds the 20 MB upload limit.';renderInferencePage();return}
 S.inferenceBusy=true;S.inferenceResult=null;S.inferenceSelectedBeat=0;renderInferencePage();
 try{
  const form=new FormData();form.append('file',file);form.append('run_id',model.run_id);form.append('model_name',model.model_name);form.append('sampling_rate_hz',String(fs));
  const response=await fetch(API_BASE+'/inference',{method:'POST',body:form});let body={};try{body=await response.json()}catch(_){body={}}
  if(!response.ok)throw new Error(body.detail||`Inference request failed (${response.status})`);
  S.inferenceResult=body;S.inferenceSelectedBeat=0;
 }catch(err){S.inferenceError=err.message}
 finally{S.inferenceBusy=false;renderInferencePage()}
}
async function loadInferenceExplanation(index){
 if(!S.inferenceResult)return;
 S.inferenceSelectedBeat=index;S.inferenceExplanation=null;renderInferencePage();
 const inferenceId=encodeURIComponent(S.inferenceResult.inference_id);
 try{
  const [explanation,waveform]=await Promise.all([
   apiRequest(`/inferences/${inferenceId}/beats/${index}/explanation`),
   apiRequest(`/inferences/${inferenceId}/beats/${index}/waveform`).catch(err=>({available:false,message:err.message}))
  ]);
  S.inferenceExplanation={...explanation,waveform};
 }
 catch(err){S.inferenceError=`Could not load beat explanation: ${err.message}`}
 renderInferencePage();
}
function renderInferencePage(){if(S.view==='inference'){keepForm(INFERENCE_FORM,()=>{$('#v').innerHTML=inferenceRender()});wireInference()}}
R.inference=inferenceRender;
A.inference=()=>{wireInference();refreshInferenceModels();};

function wire(){draw();$('#t0').oninput=e=>{S.t0=+e.target.value;draw()};$('.sc table').onclick=e=>{const tr=e.target.closest('tr.cl');if(!tr)return;S.sel=+tr.dataset.i;document.querySelectorAll('tr.cl').forEach(r=>r.classList.toggle('sel',r===tr));const t=S.rec.p.rows[S.sel].r/E.FS;if(t<S.t0||t>S.t0+8){S.t0=Math.max(0,Math.min(+$('#t0').max,Math.floor(t-2)));$('#t0').value=S.t0}draw()}}
setTheme(document.documentElement.dataset.theme==='dark'?'dark':'light');
addEventListener('keydown',e=>{if(e.key==='Escape')closeSidebar()});
addEventListener('resize',()=>{if(S.view==='analyze'&&S.rec)draw();if(S.view==='fuzzy')A.fuzzy()});
setTimeout(()=>{try{S.M=E.train();const r=sample('mixed',5,.05);analyze(r.x,E.FS,r.ann,'synthetic · mixed');go('overview')}catch(err){console.error(err);const card=document.querySelector('#ld .loading-card p');if(card)card.textContent='The browser demo could not start ('+err.message+'). The Python workspaces still work.';S.M=S.M||null;go('backend')}$('#ld')?.remove()},50);
