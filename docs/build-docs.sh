#!/usr/bin/env bash

pdoc ../mici --html --output-dir . --template-dir templates --force
rm -r docs
mv mici docs
inkscape -z -e logo.png -w 800 -h 242 ../images/mici-logo-rectangular.svg
