parallel --bar -j32 'filename=$(basename {}); num=${filename#sharp_}; num=${num%.png}; output="data/blur/blur_${num}.png";
gimp -i -b "(let* ((img (car (gimp-file-load 1 \"{}\" \"{}\")))
                   (layer (car (gimp-image-get-active-layer img))))
              (plug-in-gauss-iir2 1 img layer 13 13)
              (gimp-file-save 1 img layer \"$output\" \"$output\")
              (gimp-image-delete img)
              (gimp-quit 0))"' ::: data/sharp/sharp_*.png
