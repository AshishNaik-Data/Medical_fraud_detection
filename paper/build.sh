#!/bin/bash
# Build IEEE paper to PDF
# Requires: texlive-full or mactex (macOS) or miktex (Windows)
# Install on macOS: brew install --cask mactex
# Install on Windows: https://miktex.org/download

pdflatex -interaction=nonstopmode ieee_fraud_detection.tex
bibtex ieee_fraud_detection 2>/dev/null || true
pdflatex -interaction=nonstopmode ieee_fraud_detection.tex
pdflatex -interaction=nonstopmode ieee_fraud_detection.tex

echo ""
echo "Done. Output: ieee_fraud_detection.pdf"
