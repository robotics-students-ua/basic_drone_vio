# Planar quad — web simulator

Browser port of `2d_quad/quad_2d.py` (plain ES modules, no build step).

- `sim/quad2d.js` — dynamics + cascaded controllers, mirrors `quad_2d.py`
- `sim/world.js` — fixed-step clock, ground contact, pilot input per mode
- `sim/vision.js` — downward camera, stripe-edge landmarks, Gauss–Newton pose (mirrors `vision_1.py`, chapter 4)
- `sim/render.js` — canvas drawing (world view + 1D camera image)
- `sim/main.js` — keyboard / touch UI and the animation loop

ES modules do not load from `file://`, so serve the folder:

```bash
cd 2d_quad/web
python3 -m http.server 8000
# open http://localhost:8000/sim/
# optional: ?mode=crash|acro|stab|acceleration|velocity|position  &vision=on
```
