@echo off
setlocal

echo ... making documentation

call python -m mkdocs build

echo ... DONE making documentation
echo Output is in site/ folder. Open site/index.html to view.
echo.
echo To serve locally with live reload:
echo     mkdocs serve
echo Then open http://localhost:8000/

endlocal
