PulseLab 0.6.1 patch - extract into the PROJECT ROOT (the folder that contains server.py), choose "Replace files".
Changed: web/index.html web/styles.css web/app.js server.py modules/rpeak_detection.py modules/inference.py
         tests/test_phase7_fixes.py CHANGELOG.md README.md VERSION
Then: stop the server (Ctrl+C), run `python -m pytest -q`, start it again, and hard-refresh the browser (Ctrl+F5).
