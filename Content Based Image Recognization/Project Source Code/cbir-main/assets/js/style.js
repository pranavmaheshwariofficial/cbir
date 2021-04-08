document.getElementById("file").onchange = function() {
  'use strict';
  var fullPath = document.getElementById('file').value;
  var filename = null;
  if (fullPath) {
    var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
    filename = fullPath.substring(startIndex);
    if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) {
      filename = filename.substring(1);
    }
  }
  document.getElementById('file-path').setAttribute("value", filename);
};