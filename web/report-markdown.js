/* Small, dependency-free Markdown/report renderer for the PulseLab result dialog. */
(function attachPulseLabReportMarkdown(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.PulseLabReportMarkdown = api;
})(typeof window !== 'undefined' ? window : globalThis, function createPulseLabReportMarkdown() {
  'use strict';

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[character]);
  }

  function inlineMarkdown(value) {
    return escapeHtml(value)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>');
  }

  function parseCsv(csvText) {
    const rows = [];
    let row = [];
    let cell = '';
    let quoted = false;
    const source = String(csvText ?? '').replace(/^\uFEFF/, '');

    for (let index = 0; index < source.length; index += 1) {
      const character = source[index];
      if (character === '"') {
        if (quoted && source[index + 1] === '"') {
          cell += '"';
          index += 1;
        } else {
          quoted = !quoted;
        }
      } else if (character === ',' && !quoted) {
        row.push(cell);
        cell = '';
      } else if ((character === '\n' || character === '\r') && !quoted) {
        if (character === '\r' && source[index + 1] === '\n') index += 1;
        row.push(cell);
        if (row.some(value => value.trim() !== '')) rows.push(row);
        row = [];
        cell = '';
      } else {
        cell += character;
      }
    }

    if (cell.length || row.length) {
      row.push(cell);
      if (row.some(value => value.trim() !== '')) rows.push(row);
    }
    return rows;
  }

  function csvToHtmlTable(csvText) {
    const rows = parseCsv(csvText);
    if (rows.length < 2) return '<p class="mut">No comparison rows are available.</p>';
    const headers = rows[0];
    const body = rows.slice(1).map(row => {
      const cells = headers.map((_header, index) => `<td>${escapeHtml(row[index] ?? '')}</td>`).join('');
      return `<tr>${cells}</tr>`;
    }).join('');
    const head = headers.map(value => `<th scope="col">${escapeHtml(value)}</th>`).join('');
    return `<div class="report-table-wrap" role="region" aria-label="Held-out model comparison" tabindex="0"><table class="report-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function isPipeTableSeparator(line) {
    const cells = String(line).trim().replace(/^\|/, '').replace(/\|$/, '').split('|');
    return cells.length >= 2 && cells.every(cell => /^\s*:?-{3,}:?\s*$/.test(cell));
  }

  function parsePipeRow(line) {
    return String(line).trim().replace(/^\|/, '').replace(/\|$/, '')
      .split(/(?<!\\)\|/).map(cell => cell.replace(/\\\|/g, '|').trim());
  }

  function markdownReportToHtml(markdown, options = {}) {
    const lines = String(markdown ?? '').replace(/\r/g, '').split('\n');
    let html = '';
    let paragraph = [];
    let listType = null;
    let listItems = [];
    let inCode = false;
    let codeLines = [];
    let currentHeading = '';
    let comparisonTableInserted = false;

    const flushParagraph = () => {
      if (!paragraph.length) return;
      html += `<p>${paragraph.map(inlineMarkdown).join(' ')}</p>`;
      paragraph = [];
    };
    const flushList = () => {
      if (!listItems.length) return;
      html += `<${listType}>${listItems.map(item => `<li>${inlineMarkdown(item)}</li>`).join('')}</${listType}>`;
      listItems = [];
      listType = null;
    };
    const flushPipeTable = (header, rows) => {
      const head = header.map(cell => `<th scope="col">${inlineMarkdown(cell)}</th>`).join('');
      const body = rows.map(row => `<tr>${header.map((_cell, index) => `<td>${inlineMarkdown(row[index] ?? '')}</td>`).join('')}</tr>`).join('');
      html += `<div class="report-table-wrap" role="region" aria-label="Report table" tabindex="0"><table class="report-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
    };

    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      if (line.trim().startsWith('```')) {
        flushParagraph();
        flushList();
        if (inCode) {
          html += `<pre class="report-code"><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`;
          codeLines = [];
          inCode = false;
        } else {
          inCode = true;
        }
        continue;
      }
      if (inCode) {
        codeLines.push(line);
        continue;
      }

      const heading = line.match(/^(#{1,4})\s+(.+)$/);
      if (heading) {
        flushParagraph();
        flushList();
        currentHeading = heading[2].trim().toLowerCase();
        const level = Math.min(4, heading[1].length);
        html += `<h${level}>${inlineMarkdown(heading[2])}</h${level}>`;
        continue;
      }

      // The supplied Kaggle report currently embeds a pandas aligned-text table,
      // not a Markdown pipe table. Replace that block with its canonical CSV.
      if (currentHeading === 'held-out comparison'
        && !comparisonTableInserted
        && line.trim().length >= 120
        && /\s{2,}/.test(line)) {
        flushParagraph();
        flushList();
        html += csvToHtmlTable(options.heldoutComparisonCsv);
        comparisonTableInserted = true;
        while (index + 1 < lines.length) {
          const next = lines[index + 1];
          if (!next.trim() || /^#{1,4}\s+/.test(next)) break;
          if (next.trim().length < 80 || !/\s{2,}/.test(next)) break;
          index += 1;
        }
        continue;
      }

      if (line.trim().startsWith('|') && index + 1 < lines.length && isPipeTableSeparator(lines[index + 1])) {
        flushParagraph();
        flushList();
        const header = parsePipeRow(line);
        const rows = [];
        index += 2;
        while (index < lines.length && lines[index].trim().startsWith('|')) {
          rows.push(parsePipeRow(lines[index]));
          index += 1;
        }
        index -= 1;
        flushPipeTable(header, rows);
        continue;
      }

      const bullet = line.match(/^\s*[-*+]\s+(.+)$/);
      const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
      if (!line.trim()) {
        flushParagraph();
        flushList();
        continue;
      }
      if (/^\s*---+\s*$/.test(line)) {
        flushParagraph();
        flushList();
        html += '<hr>';
        continue;
      }
      if (bullet || ordered) {
        flushParagraph();
        const nextType = ordered ? 'ol' : 'ul';
        if (listType && listType !== nextType) flushList();
        listType = nextType;
        listItems.push((bullet || ordered)[1]);
        continue;
      }
      flushList();
      paragraph.push(line.trim());
    }

    flushParagraph();
    flushList();
    if (inCode) html += `<pre class="report-code"><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`;
    return html;
  }

  return { escapeHtml, parseCsv, csvToHtmlTable, markdownReportToHtml };
});
