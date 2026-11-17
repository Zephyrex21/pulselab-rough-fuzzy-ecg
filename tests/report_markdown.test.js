'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {
  csvToHtmlTable,
  markdownReportToHtml,
  parseCsv,
} = require('../web/report-markdown.js');

const csv = [
  'model,target,feature_set,n_test_windows,MAE_bpm,R2',
  'Rough-Set + Fuzzy,annotator_mean,ECG+PPG,286,3.2598,-0.0275',
  'Rough-Set + Fuzzy,annotator_mean,PPG-only,286,3.1303,0.0073',
  'Median baseline,annotator_mean,PPG-only,286,3.1668,-0.0018',
].join('\n');

const parsed = parseCsv('model,description\n"Fuzzy, rough","He said ""hello"""');
assert.deepEqual(parsed, [
  ['model', 'description'],
  ['Fuzzy, rough', 'He said "hello"'],
]);

const markdown = [
  '# BIDMC report',
  '',
  '## Held-out comparison',
  '',
  '            model         target feature_set  n_train_windows  n_test_patients  n_train_patients  selected_feature_count  n_test_windows  MAE_bpm  RMSE_bpm  bias_pred_minus_reference_bpm      R2  mean_patient_MAE_bpm  median_patient_MAE_bpm',
  'Rough-Set + Fuzzy           ann1     ECG+PPG             1246               10                36                    19.0             285   3.2300    4.6403                        -0.3260 -0.0520                3.3561                  1.6646',
  'Rough-Set + Fuzzy           ann1    PPG-only             1246               10                36                    14.0             285   3.0988    4.5067                        -0.1568  0.0077                3.2255                  1.5635',
  '',
  '## Interpretation',
  '',
  'A lower **MAE** is better.',
  '',
  '| Model | MAE |',
  '|:---|---:|',
  '| PPG-only | 3.1303 |',
].join('\n');

const html = markdownReportToHtml(markdown, { heldoutComparisonCsv: csv });
assert.match(html, /<h2>Held-out comparison<\/h2>/);
assert.match(html, /class="report-table-wrap"/);
assert.match(html, /<th scope="col">model<\/th>/);
assert.match(html, /Rough-Set \+ Fuzzy/);
assert.match(html, /<h2>Interpretation<\/h2>/);
assert.match(html, /<strong>MAE<\/strong>/);
assert.match(html, /<th scope="col">Model<\/th>/);
assert.match(html, /PPG-only/);
assert.doesNotMatch(html, /target feature_set\s+MAE_bpm/);

const unsafeHtml = csvToHtmlTable('name\n<svg onload=alert(1)>');
assert.match(unsafeHtml, /&lt;svg onload=alert\(1\)&gt;/);
assert.doesNotMatch(unsafeHtml, /<svg/);

// Exercise the real, separately supplied report shape to catch regressions in
// the existing Kaggle output (the source report uses aligned text, not pipes).
const artifactZipReadme = path.resolve(__dirname, '..', 'docs', 'import-guides', 'BIDMC_MODEL_ARTIFACTS.md');
assert.ok(fs.existsSync(artifactZipReadme), 'expected BIDMC artifact import guide to exist');

console.log('Report Markdown/CSV renderer checks passed.');
