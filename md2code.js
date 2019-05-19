'use strict';

var fs = require('fs');
var spawnSync = require('child_process').spawnSync;
var _ = require('lodash');

var lines = fs.readFileSync('README.md', 'utf8').split('\n').map(s => s + '\n');
var fencepos =
    lines.map((s, i) => [s, i]).filter(([s, i]) => s.indexOf('```') === 0);

var seen = new Set([]);
var replacement = [];
_.chunk(fencepos, 2).forEach(([[_, i], [__, j]]) => {
  var language = lines[i].match(/```([^\s]+)/);
  language = language ? language[1] : language;

  var fname = null;
  if (lines[i + 1].indexOf('# export') === 0) {
    fname = lines[i + 1].match(/# export ([^\s]+)/)[1];
  }
  var contentStart = i + 1 + (fname === null ? 0 : 1);
  var contents = lines.slice(contentStart, j).join('');

  if (language === 'py' || language === 'python') {
    contents =
        spawnSync(
            'yapf', ['--style', '{based_on_style: chromium, COLUMN_LIMIT:80}'],
            {input: contents, encoding: 'utf8'})
            .stdout;
    replacement.push({start: i, end: j, contentStart, contents});
  }

  if (fname) {
    if (seen.has(fname)) {
      fs.appendFileSync(fname, contents);
    } else {
      if (language === 'py' || language === 'python') {
        fs.writeFileSync(
            fname, '# -*- coding: utf-8 -*-\n\n');  // I need emoji!
        fs.appendFileSync(fname, contents);
      } else {
        fs.writeFileSync(fname, contents);
      }
      seen.add(fname);
    }
  }
});

for (var ri = replacement.length - 1; ri >= 0; ri--) {
  var r = replacement[ri];
  for (var k = r.contentStart + 1; k < r.end; k++) {
    lines[k] = '';
  }
  lines[r.contentStart] = r.contents;
}
fs.writeFileSync('README.md', lines.join(''))
