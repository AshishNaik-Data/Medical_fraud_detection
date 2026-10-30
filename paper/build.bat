@echo off
REM Build IEEE paper to PDF on Windows
REM Requires MiKTeX: https://miktex.org/download

pdflatex -interaction=nonstopmode ieee_fraud_detection.tex
bibtex ieee_fraud_detection 2>nul
pdflatex -interaction=nonstopmode ieee_fraud_detection.tex
pdflatex -interaction=nonstopmode ieee_fraud_detection.tex

echo.
echo Done. Output: ieee_fraud_detection.pdf
pause
