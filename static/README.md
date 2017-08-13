Notes
=======

To remake paper-ready pdfs, delete the `pdfs` folder and run `mkdir pdfs && for f in *.png; do convert $f -colorspace CMYK pdfs/"${f%%.*}".pdf ; done` in this directory. Only works if you have imagemagick installed (`brew install imagemagick`).

To check if a pdf is in CMYK color, run `identify -format '%[colorspace]'`


