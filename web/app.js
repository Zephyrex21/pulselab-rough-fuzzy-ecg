/* PulseLab: real-data, dual-engine frontend. No browser-generated ECG data or classifiers. */
(() => {
  'use strict';
  const $ = (s, root=document) => root.querySelector(s);
  const API = '/api';
  const state = {
    view: 'overview', health: null, mit: null, models: [], experiments: [],
    bidmc: null, selectedRecord: 'bidmc01', preview: null, featureJob: null,
    lastExport: null, inferenceResult: null, busy: false,
  };
  const pages = {
    overview: {label:'Overview', eyebrow:'DUAL RESEARCH FRAMEWORK', title:'Two datasets. One research framework.', subtitle:'Two independent real-data engines connected by a shared Rough-Fuzzy research methodology.', meta:'Local Python engines'},
    mit: {label:'MIT-BIH ECG', eyebrow:'PIPELINE 1 · ECG ARRHYTHMIA', title:'MIT-BIH ECG Research Engine', subtitle:'Patient-disjoint AAMI heartbeat classification using real ECG records, detected R-peaks, Rough Set feature selection and fuzzy rules.', meta:'Real MIT-BIH engine'},
    bidmc: {label:'BIDMC Multimodal', eyebrow:'PIPELINE 2 · CARDIORESPIRATORY', title:'BIDMC Multimodal Research Engine', subtitle:'Synchronized ECG + PPG + impedance respiration processing, multimodal feature fusion and annotated respiratory-rate research.', meta:'Real BIDMC dataset engine'},
  };
  const icons = {
    overview:'<svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>',
    mit:'<svg viewBox="0 0 24 24"><path d="M2 12h4l2.3-6 4.4 12 2.4-7 2 1H22"/></svg>',
    bidmc:'<svg viewBox="0 0 24 24"><path d="M3 12h3l2-5 4 10 3-8 2 3h4"/><path d="M12 3v2M12 19v2"/></svg>'
  };
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (v, digits=1) => Number.isFinite(Number(v)) ? Number(v).toFixed(digits) : '—';
  const pct = v => Number.isFinite(Number(v)) ? `${(100*Number(v)).toFixed(1)}%` : '—';
  const card = (title, body, extra='') => `<section class="card ${extra}"><h2>${title}</h2>${body}</section>`;
  const kpi = (value, label) => `<div class="kpi"><b>${esc(value)}</b><span>${esc(label)}</span></div>`;
  const pill = (text, kind='neutral') => `<span class="pl-pill ${kind}">${esc(text)}</span>`;
  const table = (heads, rows) => `<div class="pl-table-wrap"><table class="pl-table"><thead><tr>${heads.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.length?rows.map(r=>`<tr>${r.map(x=>`<td>${x}</td>`).join('')}</tr>`).join(''):`<tr><td colspan="${heads.length}" class="mut">No records available.</td></tr>`}</tbody></table></div>`;
  async function request(path, options={}) {
    const response = await fetch(API + path, options);
    const text = await response.text();
    let data;
    try { data = text ? JSON.parse(text) : {}; } catch { data = {detail:text}; }
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    return data;
  }
  function setPage(page) {
    state.view = page;
    const meta = pages[page] || pages.overview;
    $('#breadcrumb-current').textContent = meta.label;
    $('#page-eyebrow').textContent = meta.eyebrow;
    $('#page-title').textContent = meta.title;
    $('#page-subtitle').textContent = meta.subtitle;
    $('#page-meta-text').textContent = meta.meta;
    document.querySelectorAll('.nav-item').forEach(btn => btn.classList.toggle('on', btn.dataset.page === page));
    closeSidebar();
    if (page === 'overview') renderOverview();
    if (page === 'mit') renderMIT();
    if (page === 'bidmc') renderBIDMC();
  }
  function buildNav() {
    $('#nav').innerHTML = `<div class="sidebar-section-label">RESEARCH WORKSPACES</div>${Object.keys(pages).map(key=>`<button class="nav-item ${key==='overview'?'on':''}" data-page="${key}" type="button">${icons[key]}<span>${pages[key].label}</span></button>`).join('')}`;
    $('#nav').addEventListener('click', e => { const b=e.target.closest('[data-page]'); if (b) setPage(b.dataset.page); });
  }
  function chartSVG(time, series, labels) {
    const W=900,H=170,L=8,R=8,T=12,B=20, all=series.flat().filter(Number.isFinite);
    if (!all.length) return '<div class="mut">No signal samples to plot.</div>';
    let lo=Math.min(...all), hi=Math.max(...all); if (hi===lo) hi=lo+1;
    const paths=series.map((arr,i)=>{
      const step=Math.max(1,Math.floor(arr.length/850)); const pts=[];
      for(let j=0;j<arr.length;j+=step){const x=L+(W-L-R)*(j/Math.max(1,arr.length-1));const y=T+(H-T-B)*(1-(arr[j]-lo)/(hi-lo));pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);}
      return `<polyline points="${pts.join(' ')}" fill="none" stroke="${['#3984e8','#18a879','#9874df'][i%3]}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>`;
    }).join('');
    return `<div class="pl-chart"><div class="pl-chart-legend">${labels.map((x,i)=>`<span><i style="background:${['#3984e8','#18a879','#9874df'][i%3]}"></i>${esc(x)}</span>`).join('')}</div><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Actual BIDMC physiological signal waveforms"><path d="M${L} ${T}H${W-R}M${L} ${(H+T-B)/2}H${W-R}M${L} ${H-B}H${W-R}" class="pl-gridline"/>${paths}<text x="${L}" y="${H-3}" class="pl-svg-label">${fmt(time[0],1)} s</text><text x="${W-R}" y="${H-3}" text-anchor="end" class="pl-svg-label">${fmt(time[time.length-1],1)} s</text></svg></div>`;
  }
  async function loadCommon() {
    const [health, mit, models, experiments, bidmc] = await Promise.allSettled([
      request('/health'), request('/dataset/status?quick=true'), request('/models'), request('/experiments'), request('/bidmc/status')
    ]);
    state.health = health.status==='fulfilled'?health.value:null;
    state.mit = mit.status==='fulfilled'?mit.value:null;
    state.models = models.status==='fulfilled'?models.value.models||[]:[];
    state.experiments = experiments.status==='fulfilled'?experiments.value.experiments||[]:[];
    state.bidmc = bidmc.status==='fulfilled'?bidmc.value:null;
  }
  function renderOverview() {
    const mitCount=state.health?.dataset_records_available ?? 0;
    const bidmcCount=state.bidmc?.records_found ?? 0;
    const models=state.models.length;
    const exp=state.experiments.length;
    $('#v').innerHTML = `
      <div class="pl-hero-grid">
        <article class="pl-engine-card mit-card"><div class="pl-engine-top"><span class="pl-engine-index">PIPELINE 01</span>${pill('AAMI ARRHYTHMIA','blue')}</div><h2>MIT-BIH ECG Engine</h2><p>Real ECG waveform analysis, R-peak detection, heartbeat segmentation, 47-feature extraction, Rough Set reduction and fuzzy AAMI classification.</p><div class="pl-engine-meta"><span>MIT-BIH Arrhythmia Database</span><span>360 Hz</span><span>5 AAMI classes</span></div><button class="btn pri" data-go="mit">Open ECG engine <span aria-hidden="true">→</span></button></article>
        <article class="pl-engine-card bidmc-card"><div class="pl-engine-top"><span class="pl-engine-index">PIPELINE 02</span>${pill('MULTIMODAL','green')}</div><h2>BIDMC Multimodal Engine</h2><p>Real synchronized ECG + PPG + respiration signals, window-level multimodal feature fusion and dual-annotator respiratory-rate references.</p><div class="pl-engine-meta"><span>BIDMC Dataset</span><span>125 Hz</span><span>3 modalities</span></div><button class="btn pri" data-go="bidmc">Open BIDMC engine <span aria-hidden="true">→</span></button></article>
      </div>
      ${card('Live research workspace',`<div class="grid">${kpi(mitCount,'MIT-BIH records available locally')}${kpi(models,'Saved MIT-BIH models')}${kpi(exp,'MIT-BIH experiment runs')}${kpi(`${bidmcCount}/53`,'BIDMC records indexed')}</div><p class="mut" style="margin-top:12px">These values come from the local Python API and saved research artifacts. No signals or predictions are generated in the browser.</p>`)}
      ${card('Shared Rough-Fuzzy research framework',`<div class="pl-framework"><div><b>Feature engineering</b><span>Dataset-specific, synchronized signal descriptors</span></div><i>→</i><div><b>Rough Set selection</b><span>Training-only attribute reduction</span></div><i>→</i><div><b>Fuzzy inference</b><span>Interpretable rule-based outputs</span></div></div><p class="mut">Datasets, schemas, models and evaluation metrics remain isolated between pipelines. Only validated methodological components are shared.</p>`)}
      ${card('Research and safety boundary',`<p class="mut">PulseLab is an academic research prototype, not a medical device. Its outputs are not clinically validated and must not be used for diagnosis, treatment or urgent decisions. All displayed outputs are sourced from real dataset files, saved experiment artifacts or the local Python API.</p>`)}
    `;
    $('#v').querySelectorAll('[data-go]').forEach(b=>b.addEventListener('click',()=>setPage(b.dataset.go)));
  }
  function renderMIT() {
    const exp=state.experiments[0];
    const best=exp?.best_model;
    const models=state.models;
    const mit=state.mit||{};
    const modelOptions=models.map((m,i)=>`<option value="${esc(m.model_id)}" ${i===0?'selected':''}>${esc(m.model_name)} · ${esc(m.run_id)}</option>`).join('');
    $('#v').innerHTML = `
      ${card('MIT-BIH engine status',`<div class="grid">${kpi(`${mit.available_requested??0}/${mit.train_records_requested+mit.test_records_requested||0}`,'quick split records available')}${kpi(models.length,'saved trained models')}${kpi(exp?exp.n_train:'—','training beats')}${kpi(exp?exp.n_test:'—','testing beats')}</div><p class="mut" style="margin-top:12px">Dataset root: <code>${esc(mit.dataset_root||'data/mitdb')}</code>. The MIT-BIH waveform dataset is not bundled; place records in this folder or use the supported download workflow.</p>${mit.ready_for_selected_experiment?pill('Quick split ready','green'):pill(`${(mit.missing_records||[]).length} records missing locally`,'amber')}`)}
      ${card('Saved patient-disjoint experiment',exp?`<div class="row">${pill(exp.patient_disjoint?'PATIENT-DISJOINT':'CHECK SPLIT',exp.patient_disjoint?'green':'amber')}<span class="mut">${esc(exp.run_id)} · ${esc(exp.created_at_utc||'')}</span></div><div class="grid">${kpi(exp.best_model?.model||'—','highest macro-F1 model')}${kpi(pct(exp.best_model?.accuracy),'accuracy')}${kpi(pct(exp.best_model?.macro_f1),'macro F1')}${kpi(pct(exp.best_model?.balanced_accuracy),'balanced accuracy')}</div><p class="mut" style="margin-top:12px">Model scores are taken from the saved training run. The supplied run was trained using source version 0.6.0; current preprocessing is version 0.6.1. Re-training is recommended before treating this as a matched production inference configuration.</p><div class="row"><button class="btn" id="refresh-mit">Refresh status</button><a class="btn" href="/api/experiments/${encodeURIComponent(exp.run_id)}/summary" target="_blank" rel="noopener">Open run summary</a></div>`:`<p class="mut">No saved experiment manifest was found. Add the trained artifact export to models/ and experiments/ or run an MIT-BIH experiment once records are available.</p>`)}
      ${card('Run the real MIT-BIH research pipeline',`<p class="mut">This launches the Python pipeline on local MIT-BIH WFDB files. It does not use browser-generated ECG signals.</p><div class="row"><label class="mut">Experiment size</label><select id="mit-quick"><option value="true">Quick engineering subset</option><option value="false">Complete patient-disjoint split</option></select><label class="mut">Peak source</label><select id="mit-peaks"><option value="detected">Automatic R-peak detection</option><option value="annotations">Reference annotations (benchmark only)</option></select><button class="btn pri" id="run-mit" ${!mit.ready_for_selected_experiment?'disabled':''}>Start real MIT-BIH experiment</button></div><div id="mit-job" class="mut"></div>${!mit.ready_for_selected_experiment?`<p class="mut">Missing records include: ${esc((mit.missing_records||[]).slice(0,10).join(', '))}${(mit.missing_records||[]).length>10?' …':''}. Use the Research API dataset download flow or provide WFDB records.</p>`:''}`)}
      ${card('Saved-model ECG inference',models.length?`<p class="mut">Run an uploaded ECG signal through a saved MIT-BIH-trained Python model. No browser classifier is involved.</p><div class="pl-form-grid"><label>Trained model<select id="mit-model">${modelOptions}</select></label><label>Sampling rate (Hz)<input id="mit-fs" type="number" min="100" max="2000" value="360"></label><label class="wide">ECG signal CSV/TXT<input id="mit-file" type="file" accept=".csv,.txt"></label></div><div class="row"><button class="btn pri" id="mit-infer">Run Python ECG inference</button><span class="mut">3 seconds–10 minutes · maximum 20 MB</span></div><div id="mit-infer-result"></div>`:`<p class="mut">No compatible saved MIT-BIH model was discovered. The provided model export will appear here after installation if its run manifest and model files are present.`)}
      ${card('Evaluation caution',`<p class="mut">The supplied trained run has strong overall accuracy but low minority-class recall. Accuracy alone is not evidence of reliable arrhythmia recognition. Inspect macro-F1, balanced accuracy, per-class sensitivity, patient-level evaluation and detector false-negative counts before interpreting results.</p>`)}
    `;
    $('#refresh-mit').addEventListener('click', async()=>{await refreshAll();renderMIT();});
    const runBtn=$('#run-mit');
    if(runBtn) runBtn.addEventListener('click', async()=>{
      runBtn.disabled=true; $('#mit-job').textContent='Starting Python experiment…';
      try {const created=await request('/experiments',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({quick:$('#mit-quick').value==='true',download_missing:false,split_protocol:'patient-disjoint',peak_source:$('#mit-peaks').value,include_baselines:true,save_models:true})}); await pollJob(created.job_id,'mit-job'); await refreshAll(); renderMIT();}
      catch(err){$('#mit-job').textContent=err.message;runBtn.disabled=false;}
    });
    const infer=$('#mit-infer');
    if(infer) infer.addEventListener('click', async()=>{
      const file=$('#mit-file').files[0]; if(!file){$('#mit-infer-result').textContent='Choose a signal CSV/TXT file first.';return;}
      const [run_id,model_name]=($('#mit-model').value||'').split('::'); if(!run_id||!model_name){$('#mit-infer-result').textContent='No trained model selected.';return;}
      const form=new FormData();form.append('file',file);form.append('run_id',run_id);form.append('model_name',model_name);form.append('sampling_rate_hz',$('#mit-fs').value||360);
      infer.disabled=true;$('#mit-infer-result').textContent='Running saved Python model…';
      try {const result=await request('/inference',{method:'POST',body:form});$('#mit-infer-result').innerHTML=`<div class="pl-result">${kpi(result.n_predictions,'beats predicted')}${kpi(result.n_detected_peaks,'detected R-peaks')}${kpi(result.mean_heart_rate_bpm?`${fmt(result.mean_heart_rate_bpm,0)} bpm`:'—','mean heart rate')}<p class="mut">Inference ID: ${esc(result.inference_id||'')}</p><a class="btn" href="/api/inferences/${encodeURIComponent(result.inference_id)}/summary" target="_blank" rel="noopener">Open inference summary</a></div>`;}
      catch(err){$('#mit-infer-result').textContent=err.message;}finally{infer.disabled=false;}
    });
  }
  async function renderBIDMC() {
    const b=state.bidmc||{};
    const rows=(b.records||[]).map(r=>[esc(r.record_id),esc(r.patient_id),esc((r.channels||[]).filter(x=>['II','PLETH','RESP'].includes(x)).join(', ')),r.breath_annotations_available?pill('Dual annotations','green'):pill('No annotations','amber'),r.monitor_numerics_available?pill('Numerics','blue'):'—']);
    $('#v').innerHTML = `
      ${card('BIDMC dataset status',`<div class="grid">${kpi(`${b.records_found||0}/53`,'recording CSVs indexed')}${kpi(b.patient_groups??'—','patient groups identified')}${kpi('125 Hz','waveform sampling rate')}${kpi('1 Hz','monitor numerics')}</div><p class="mut" style="margin-top:12px">Dataset path: <code>${esc(b.csv_directory||'data/bidmc/bidmc_csv')}</code></p>${b.ready?pill('Dataset files indexed','green'):pill('Dataset not found / incomplete','amber')}<p class="mut">${esc(b.message||'')}</p><div class="pl-callout">Dataset archive is kept outside the source repository. Extract the PhysioNet ZIP so the project can find <code>bidmc_csv/</code> below <code>data/bidmc/</code>.</div>`)}
      ${card('Multimodal processing architecture',`<div class="pl-modality-grid"><div><strong>ECG · Lead II</strong><span>Cardiac morphology, heart rate and interval variability</span></div><div><strong>PPG · PLETH</strong><span>Pulse morphology, pulse rate and pulse variability</span></div><div><strong>RESP · impedance</strong><span>Respiratory cycles and waveform dynamics</span></div></div><div class="pl-framework"><div><b>Signal conditioning</b><span>Per-modality filtering and quality checks</span></div><i>→</i><div><b>Window features</b><span>32-second windows, 16-second stride</span></div><i>→</i><div><b>Feature fusion</b><span>ECG + PPG + RESP descriptors</span></div><i>→</i><div><b>Rough-Fuzzy</b><span>Feature reduction and fuzzy regression</span></div></div><p class="mut">The model-training state is intentionally pending. No BIDMC predictions or performance metrics are fabricated.</p>`)}
      ${card('Inspect a real BIDMC recording',`<div class="row"><label class="mut" for="bidmc-record">Recording</label><select id="bidmc-record">${(b.records||[]).map(r=>`<option value="${esc(r.record_id)}" ${r.record_id===state.selectedRecord?'selected':''}>${esc(r.record_id)} · ${esc(r.patient_id)}</option>`).join('')||'<option>Dataset not indexed</option>'}</select><button class="btn pri" id="load-bidmc" ${!b.records_found?'disabled':''}>Load real recording</button></div><div id="bidmc-preview" class="mut">Select a record and load its actual waveform channels.</div>`)}
      ${card('Extract synchronized multimodal features',`<p class="mut">Build training-ready windows and dual-annotator respiratory-rate references from actual BIDMC signals. This does not train a model.</p><div class="row"><label class="mut">Window</label><select id="bidmc-window"><option value="32">32 seconds (recommended)</option><option value="64">64 seconds</option></select><label class="mut">Stride</label><select id="bidmc-stride"><option value="16">16 seconds</option><option value="32">32 seconds</option></select><button class="btn pri" id="export-bidmc" ${!b.ready?'disabled':''}>Extract feature dataset</button></div><div id="bidmc-job" class="mut"></div>${state.lastExport?`<div class="pl-callout">Last export: <b>${esc(state.lastExport.export_id)}</b> · ${esc(state.lastExport.n_windows)} windows · ${esc(state.lastExport.n_patient_groups)} patient groups <div class="row" style="margin-top:9px"><a class="btn" href="/api/bidmc/feature-exports/${encodeURIComponent(state.lastExport.export_id)}/files/features_full_multimodal.csv" download>Full multimodal CSV</a><a class="btn" href="/api/bidmc/feature-exports/${encodeURIComponent(state.lastExport.export_id)}/files/features_rr_estimation_no_resp_input.csv" download>RR estimation CSV (leakage controlled)</a><a class="btn" href="/api/bidmc/feature-exports/${encodeURIComponent(state.lastExport.export_id)}/files/feature_export_summary.json" download>Summary JSON</a></div></div>`:''}`)}
      ${card('Respiratory-rate target and leakage controls',`<p class="mut">The target is derived independently from the two manual breath annotators (ann1 and ann2), using their annotated breath sample intervals. The monitor's 1 Hz respiratory-rate channel is retained as a separate reference, not substituted for manual labels.</p><ul class="pl-compact-list"><li><b>Full multimodal table:</b> includes ECG, PPG and RESP features for multimodal exploratory analysis.</li><li><b>RR estimation table:</b> excludes RESP-derived predictors and ECG/PPG-to-RESP correlations because the target is derived from RESP-based manual annotations.</li><li><b>Evaluation:</b> use patient-grouped train/test splits; repeated windows from one patient must never cross split boundaries.</li></ul><p class="mut">The provided CSV and WFDB distributions include 53 recordings. Where MIMIC-II patient IDs cannot be read from accompanying WFDB headers, the loader marks a record-level fallback group instead of guessing.</p>`)}
      ${card('BIDMC model status',`<div class="row">${pill('TRAINING PENDING','amber')}${pill('NO PREDICTIONS AVAILABLE','neutral')}</div><p class="mut">The second engine now supports BIDMC dataset discovery, record loading, waveform preview, dual-annotator target extraction, multimodal feature generation and export. Rough-Fuzzy respiratory-rate regression code is included, but it will only produce predictions after a patient-disjoint Kaggle training run exports a compatible BIDMC model artifact.</p>`)}
    `;
    const loadBtn=$('#load-bidmc');
    if(loadBtn) loadBtn.addEventListener('click', async()=>{
      state.selectedRecord=$('#bidmc-record').value; loadBtn.disabled=true; $('#bidmc-preview').textContent='Reading real BIDMC signal files…';
      try {state.preview=await request(`/bidmc/records/${encodeURIComponent(state.selectedRecord)}/preview?duration_sec=20`);$('#bidmc-preview').innerHTML=renderPreview(state.preview);}
      catch(err){$('#bidmc-preview').textContent=err.message;}finally{loadBtn.disabled=false;}
    });
    const exportBtn=$('#export-bidmc');
    if(exportBtn) exportBtn.addEventListener('click',async()=>{
      exportBtn.disabled=true;$('#bidmc-job').textContent='Queuing BIDMC feature extraction…';
      try {const created=await request('/bidmc/feature-exports',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({records:[],window_sec:Number($('#bidmc-window').value),stride_sec:Number($('#bidmc-stride').value)})});await pollJob(created.job_id,'bidmc-job',true);}
      catch(err){$('#bidmc-job').textContent=err.message;exportBtn.disabled=false;}
    });
    if(state.preview) $('#bidmc-preview').innerHTML=renderPreview(state.preview);
  }
  function renderPreview(p) {
    return `<div class="grid" style="margin:10px 0">${kpi(p.record_id,'record')}${kpi(p.patient_id,'patient group')}${kpi(`${fmt(p.sampling_frequency_hz,0)} Hz`,'sampling rate')}${kpi(`${fmt(p.duration_sec,1)} s`,'preview duration')}</div>${chartSVG(p.time_sec,[p.ecg,p.ppg,p.resp],['ECG II','PPG PLETH','RESP'])}<p class="mut">Real source channels: ${esc((p.channel_names||[]).join(', '))}. Manual annotation events in preview: ann1 ${p.annotations?.ann1_sec?.length||0}, ann2 ${p.annotations?.ann2_sec?.length||0}.</p>`;
  }
  async function pollJob(jobId, targetId, bidmc=false) {
    const target=$('#'+targetId);let done=false;
    while(!done){
      const job=await request(`/jobs/${encodeURIComponent(jobId)}`);
      if(!target) return;
      target.textContent=`${job.status.toUpperCase()} · ${Math.round((job.progress||0)*100)}% · ${job.message||''}`;
      if(job.status==='completed'){
        done=true;
        if(bidmc && job.result){state.lastExport=job.result;target.innerHTML=`<b>Feature extraction complete.</b> ${job.result.n_windows} windows across ${job.result.n_records} records and ${job.result.n_patient_groups} patient groups. Refreshing exports…`;renderBIDMC();}
        else if(job.status==='completed'){target.textContent='Experiment completed. Refreshing saved runs…';}
      } else if(job.status==='failed'){done=true;target.textContent=`FAILED · ${job.error||'Unknown error'}`;}
      else await new Promise(resolve=>setTimeout(resolve,1400));
    }
  }
  async function refreshAll(){ await loadCommon(); }
  function setTheme(theme){document.documentElement.dataset.theme=theme;try{localStorage.setItem('pulselab-theme',theme);}catch{} }
  function initTheme(){let theme='light';try{theme=localStorage.getItem('pulselab-theme')||'light';}catch{}setTheme(theme);$('#theme-toggle').addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));}
  function closeSidebar(){document.body.classList.remove('sidebar-open');$('#sidebar-toggle')?.setAttribute('aria-expanded','false');}
  function initSidebar(){ $('#sidebar-toggle').addEventListener('click',()=>{document.body.classList.toggle('sidebar-open');$('#sidebar-toggle').setAttribute('aria-expanded',document.body.classList.contains('sidebar-open')?'true':'false');});$('#sidebar-backdrop').addEventListener('click',closeSidebar);$('#sidebar-collapse').addEventListener('click',()=>document.body.classList.toggle('sidebar-collapsed')); }
  async function start(){
    buildNav();initTheme();initSidebar();
    try {await loadCommon();} catch(err){console.error(err);}
    setPage('overview');
    $('#ld')?.remove();
    if(!state.health){const banner=document.createElement('div');banner.className='pl-offline';banner.textContent='Local Python API is offline. Start PulseLab with run_server.bat / the documented uvicorn command to access both research engines.';$('#v').prepend(banner);}
  }
  document.addEventListener('DOMContentLoaded',start);
})();
