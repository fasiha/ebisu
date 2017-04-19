"use strict";

var fs = require('fs');
var _ = require('lodash');

var lines = fs.readFileSync('README.md', 'utf8').split('\n');
var fencepos =
    lines.map((s, i) => [s, i]).filter(([ s, i ]) => s.indexOf('```') === 0);

var seen = new Set([]);

_.chunk(fencepos, 2).forEach(([ [ _, i ], [ __, j ] ]) => {
  if (lines[i + 1].indexOf('# export') === 0) {
    var fname = lines[i + 1].match(/# export ([^\s]*)/)[1];
    var contents = lines.slice(i + 2, j).join('\n');
    if (seen.has(fname)) {
      fs.appendFileSync(fname, contents);
    } else {
      fs.writeFileSync(fname, contents);
      seen.add(fname);
    }
    fs.appendFileSync(fname, '\n');
  }
})
