#!/usr/bin/env bash

pdoc ../mici --html --output-dir . --template-dir templates --force
rm -r docs
mv mici docs
cp -r ../images .
pandoc -f gfm -t html ../README.md -o index.html -s --css style.css --toc -M pagetitle='Mici - Python implementations of manifold MCMC methods'
