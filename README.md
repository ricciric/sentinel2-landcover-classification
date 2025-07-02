# Generating images EO with ControlNet + Stable Diffusion

## Objective of the Phase 1

Experimenting on the ControlNet to guide the generation of the images via Stable Diffusion, using multispectrum data from Sentinel-2. I explored three kind of input:

- **True Color (B4, B3, B2) Natural RGB Image**
- **False Color (B8, B4, B3) Vegetation visualized in red**
- **NDVI (got from B8 and B4) Vegetation map on a gray scale**

---

## Area of study

A little agricultural area near Rome, defined by:
'''
json
{"type":"Polygon","coordinates":[[[12.120752,41.996881],[12.147274,41.983548],[12.12719,41.962043],[12.096119,41.978826],[12.120752,41.996881]]]}
'''

## Output

- **True Color: Consistent generation and photorealistic, represent the structures of the original vegetation accurately**
- **False Color: Less readable due to the artistic color used**
- **NDVI: Strong spatial control, abstract aesthetic but great to isolate the vegetation zones**

---

## Objective of the Phase 2
