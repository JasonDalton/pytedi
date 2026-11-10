# Example: farm ponds around Hurt, Virginia

`run_hurt.py` builds a pytedi dam list for the HUC12 watershed that contains Hurt, Virginia
(Pittsylvania County, on the Staunton River in the Roanoke basin), from US public data.

```
pip install -e ".[us]"
python examples/us_hurt_va/run_hurt.py            # about 2 minutes on the first run
python examples/us_hurt_va/run_hurt.py --help     # threshold, dates, Sentinel-2 on/off
```

Outputs go to `examples/us_hurt_va/output/` (git-ignored):

| File | Contents |
|------|----------|
| `hurt_va_damlist.csv` | the dam list, in the columns of `tedimodel().damlist` |
| `hurt_va_candidates.gpkg` | layers `candidates` (pond polygons with all attributes), `outlets`, `nid_dams`, `flowlines`, `aoi` |
| `hurt_va_map.png` | hillshade, flowlines, ponds by source, pond-to-pond links, NID dams |
| `cache/` | downloads, reused on later runs |

## Method

1. **Area of interest.** `huc12_at(lat, lon)` finds the HUC12 in the USGS Watershed Boundary
   Dataset. Any bounding box, polygon, GeoJSON or HUC8/10/12 id also works.
2. **Terrain.** 3DEP 1/3 arc-second (about 10 m) DEM, in EPSG:5070.
3. **Mapped ponds.** NHD high-resolution waterbodies of type LakePond (390) and Reservoir (436),
   flowlines, and river polygons. The NHD services time out on detailed polygons, so the
   bounding box is queried and the result clipped. `nhd_dataset="3dhp"` uses the 3D
   Hydrography Program instead.
4. **Official dams.** The USACE National Inventory of Dams national GeoPackage, converted to
   metric units.
5. **Detected ponds.** Sentinel-2 Level-2A scenes from Element 84's Earth Search (public, no
   credentials) for June to September, under 20% cloud; clouds and shadows masked with the
   scene classification layer; per-pixel median; McFeeters NDWI from the 10 m green and NIR
   bands; water where NDWI > 0; polygons of 500 m2 (0.05 ha) or more.
6. **Merge.** NHD ponds are kept; a Sentinel-2 polygon that overlaps one marks it `both`;
   other Sentinel-2 polygons become `sentinel2` ponds unless they lie in an NHD river polygon.
   Each NID dam is matched to the nearest pond within 150 m; an unmatched dam becomes a `nid`
   pond with its NID surface area.
7. **Network.** The DEM is conditioned and routed (D8) with pysheds. Each pond's outlet is its
   highest-accumulation cell; `catchuptotal_km2` is the area draining there; `ds_damid` is the
   first pond the flow path enters ("-99" for the catchment outlet); `catchuplocal_km2` is the
   total less the totals of the ponds directly upstream.
8. **Volume.** `vol_ML = a * surfacearea_m2 ** b`. The default is the model's own relation,
   a = 0.0001042 and b = 1.3213 (Fowler et al. 2015, farm dams in Victoria, Australia). Against
   Virginia's National Inventory of Dams records for private farm-type dams of 2 to 20 acres it
   gives a median of 0.96 to 1.16 times the listed normal storage. `--volume-relation va_nid_farm`
   uses a = 0.0122006, b = 0.845694, fitted to 476 of those Virginia dams (R2 0.55); it implies
   implausibly deep small ponds (about 4.7 m mean depth at 500 m2) because the NID only lists
   dams that store at least 15 to 50 acre-feet. Both are estimates.
9. **Dam list.** No demands, bypasses or winterfill yet: every dam gets the engine's neutral
   values (group "none", demand factor 0, operating in all years).

## Limits

- NHD ponds are kept even when Sentinel-2 shows no water. NHD outlines often date from
  older topographic mapping, and some ponds have since been drained or overgrown; others are
  shaded by trees or covered in algae, which keeps NDWI below zero.
- Ponds smaller than about 0.05 ha, or under tree canopy, are missed by Sentinel-2.
- Lowering the NDWI threshold finds more ponds but also bright roofs and pads; check the map.
- Catchment areas stop at the AOI boundary. A pond fed from outside the AOI has its
  catchment underestimated.
- No published area-volume relation for US farm ponds was found; the default is Australian,
  checked against Virginia NID dams of 2 acres and up.

## References

Fowler, K., Morden, R., Lowe, L., Nathan, R., 2015. Advances in assessing the impact of
hillside farm dams on streamflow. Australian Journal of Water Resources 19.
https://doi.org/10.1080/13241583.2015.1116182

McFeeters, S.K., 1996. The use of the Normalized Difference Water Index (NDWI) in the
delineation of open water features. International Journal of Remote Sensing 17, 1425-1432.
https://doi.org/10.1080/01431169608948714
