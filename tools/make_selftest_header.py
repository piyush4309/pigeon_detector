"""Generate firmware/examples/bird_detect/main/selftest_images.h (320x240 RGB888 arrays).
Then set BIRD_SELFTEST to 1 in app_main.cpp to run the model on these images at boot.
Usage: python make_selftest_header.py out.h name=image.jpg [name=image.jpg ...]"""
import sys
from PIL import Image
out = open(sys.argv[1], "w"); names = []
out.write("#pragma once\n#include <stdint.h>\n")
for a in sys.argv[2:]:
    n, f = a.split("=")
    d = Image.open(f).convert("RGB").resize((320, 240)).tobytes()
    out.write(f"static const uint8_t img_{n}[] = {{" + ",".join(map(str, d)) + "};\n"); names.append(n)
out.write("static const char *selftest_names[] = {" + ",".join(f'"{n}"' for n in names) + "};\n")
out.write("static const uint8_t *selftest_imgs[] = {" + ",".join(f"img_{n}" for n in names) + "};\n")
