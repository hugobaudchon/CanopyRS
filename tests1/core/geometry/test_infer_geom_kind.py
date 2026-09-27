"""infer_geom_kind, in every combination of the CRS the geometries are saved in and the CRS of the
image they were drawn on: projected (UTM, MTM, Web Mercator) and geographic (WGS84, NAD83), equal
or different, north and south of the equator."""

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS, Transformer
from rasterio.crs import CRS as RasterioCRS
from shapely.affinity import rotate
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Point, Polygon, box

from canopyrs1.core.constants import GeomKind
from canopyrs1.core.geometry.shapes import infer_geom_kind

BOX, MASK, POINT = GeomKind.BOX, GeomKind.MASK, GeomKind.POINT

MONTREAL, SYDNEY = (-73.6, 45.5), (151.2, -33.9)
# The CRSs used around each place: UTM (its zone and the next one), MTM, Web Mercator, WGS84, NAD83.
CRS_AROUND = {
    MONTREAL: ["EPSG:32618", "EPSG:32619", "EPSG:2950", "EPSG:3857", "EPSG:4326", "EPSG:4269"],
    SYDNEY: ["EPSG:32756", "EPSG:32755", "EPSG:3857", "EPSG:4326"],
}
PAIRS = [
    (place, image_crs, saved_crs)
    for place, crss in CRS_AROUND.items()
    for image_crs in crss
    for saved_crs in crss
]


def _drawn_in(crs, place):
    """Return (boxes, crowns) drawn in ``crs`` around ``place``: axis-aligned boxes from 50 cm to
    2 km, a long thin one, and round crowns, in that CRS's own coordinates."""
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(*place)
    u = 1.0 if CRS.from_user_input(crs).is_projected else 1e-5  # about 1 m
    boxes = [
        box(x, y, x + 0.5 * u, y + 0.5 * u),
        box(x, y, x + 10 * u, y + 10 * u),
        box(x + 100 * u, y, x + 500 * u, y + 20 * u),
        box(x, y + 300 * u, x + 2000 * u, y + 2300 * u),
    ]
    crowns = [Point(x + k * 50 * u, y).buffer(8 * u) for k in range(3)]
    return boxes, crowns


def _saved_in(geometries, drawn_crs, saved_crs):
    """Return ``geometries`` reprojected the way a user would, with geopandas."""
    return gpd.GeoSeries(geometries, crs=drawn_crs).to_crs(saved_crs)


# =============================================================================
# Every combination of image CRS and saved CRS
# =============================================================================


@pytest.mark.parametrize("place, image_crs, saved_crs", PAIRS)
def test_boxes_drawn_on_the_image_saved_in_any_crs(place, image_crs, saved_crs):
    boxes, crowns = _drawn_in(image_crs, place)
    saved = _saved_in(boxes + crowns, image_crs, saved_crs)
    kinds = infer_geom_kind(saved, crs=saved_crs, image_crs=image_crs)
    assert list(kinds) == [BOX] * len(boxes) + [MASK] * len(crowns)


@pytest.mark.parametrize("place, image_crs, saved_crs", PAIRS)
def test_densified_boxes_drawn_on_the_image(place, image_crs, saved_crs):
    # Some tools add points along the edges before reprojecting, so the edges bend.
    boxes, _ = _drawn_in(image_crs, place)
    step = box(*boxes[1].bounds).length / 40
    saved = _saved_in(gpd.GeoSeries(boxes).segmentize(step), image_crs, saved_crs)
    assert list(infer_geom_kind(saved, crs=saved_crs, image_crs=image_crs)) == [BOX] * len(boxes)


@pytest.mark.parametrize("place, image_crs, saved_crs", PAIRS)
def test_boxes_drawn_in_the_file_crs(place, image_crs, saved_crs):
    # Axis-aligned in the file's own CRS, whatever the image's CRS.
    boxes, crowns = _drawn_in(saved_crs, place)
    saved = gpd.GeoSeries(boxes + crowns, crs=saved_crs)
    kinds = infer_geom_kind(saved, crs=saved_crs, image_crs=image_crs)
    assert list(kinds) == [BOX] * len(boxes) + [MASK] * len(crowns)


@pytest.mark.parametrize("place, image_crs, saved_crs", PAIRS)
def test_tilted_rectangles_are_masks_in_every_crs(place, image_crs, saved_crs):
    boxes, _ = _drawn_in(image_crs, place)
    tilted = [rotate(b, 10) for b in boxes]
    saved = _saved_in(tilted, image_crs, saved_crs)
    assert list(infer_geom_kind(saved, crs=saved_crs, image_crs=image_crs)) == [MASK] * len(tilted)


# =============================================================================
# What each argument does
# =============================================================================


def test_without_the_image_crs_a_reprojected_box_can_look_tilted():
    # A UTM box saved in WGS84 is tilted there; only the image's CRS shows it is a box.
    boxes, _ = _drawn_in("EPSG:32618", MONTREAL)
    saved = _saved_in(boxes, "EPSG:32618", "EPSG:4326")
    assert MASK in infer_geom_kind(saved, crs="EPSG:4326")
    kinds = infer_geom_kind(saved, crs="EPSG:4326", image_crs="EPSG:32618")
    assert list(kinds) == [BOX] * len(boxes)


def test_pixel_coordinates():
    shapes = [box(0, 0, 10, 4), rotate(box(0, 0, 10, 4), 10), Point(3, 3).buffer(2)]
    expected = [BOX, MASK, MASK]
    assert list(infer_geom_kind(shapes)) == expected
    # Without a crs, there is nothing to move.
    assert list(infer_geom_kind(shapes, image_crs="EPSG:32618")) == expected


@pytest.mark.parametrize("make_crs", [str, CRS.from_user_input, RasterioCRS.from_string])
def test_every_way_of_giving_a_crs(make_crs):
    boxes, _ = _drawn_in("EPSG:32618", MONTREAL)
    saved = _saved_in(boxes, "EPSG:32618", "EPSG:4326")
    kinds = infer_geom_kind(saved, crs=make_crs("EPSG:4326"), image_crs=make_crs("EPSG:32618"))
    assert list(kinds) == [BOX] * len(boxes)


def test_the_same_crs_written_differently():
    boxes, _ = _drawn_in("EPSG:4326", MONTREAL)
    kinds = infer_geom_kind(boxes, crs="EPSG:4326", image_crs=CRS.from_epsg(4326).to_wkt())
    assert list(kinds) == [BOX] * len(boxes)


# =============================================================================
# Shapes
# =============================================================================

SHAPES = [
    (Point(1, 2), POINT),
    (MultiPoint([(0, 0), (1, 1)]), POINT),
    (box(0, 0, 10, 4), BOX),
    (Polygon([(0, 0), (5, 0), (10, 0), (10, 4), (0, 4)]), BOX),  # an extra point on an edge
    (Polygon([(0, 0), (10, 1e-9), (10, 4), (1e-9, 4)]), BOX),  # rounding noise
    (Polygon([(0, 0), (10, 0), (10.3, 4), (0.3, 4)]), MASK),  # skewed
    (Polygon([(0, 0), (4, -1), (10, 0), (4, 1)]), MASK),  # a kite
    (Polygon([(0, 0), (10, 0), (5, 8)]), MASK),  # a triangle
    (Point(0, 0).buffer(5), MASK),  # a round crown
    (Point(0, 0).buffer(5, quad_segs=2), MASK),  # an octagon
    (box(0, 0, 10, 10).difference(box(8, 8, 10, 10)), MASK),  # a box missing a corner
    (Polygon(box(0, 0, 10, 10).exterior, holes=[box(4, 4, 6, 6).exterior]), MASK),
    (MultiPolygon([box(0, 0, 1, 1)]), MASK),
    (MultiPolygon([box(0, 0, 1, 1), box(3, 3, 4, 4)]), MASK),
    (Polygon([(0, 0), (1, 1), (2, 2), (3, 3)]), MASK),  # no area
    (Polygon(), MASK),
]


@pytest.mark.parametrize("geometry, kind", SHAPES)
def test_shapes(geometry, kind):
    assert list(infer_geom_kind([geometry])) == [kind]


def test_the_order_is_kept():
    series = gpd.GeoSeries(
        [Point(0, 0), box(0, 0, 1, 1), Point(0, 0).buffer(1), box(5, 5, 6, 9)],
        index=[7, 3, 5, 1],
    )
    assert list(infer_geom_kind(series)) == [POINT, BOX, MASK, BOX]
    kinds = infer_geom_kind([])
    assert isinstance(kinds, np.ndarray) and len(kinds) == 0


def test_other_geometries_are_refused():
    with pytest.raises(ValueError, match="LineString, None"):
        infer_geom_kind([box(0, 0, 1, 1), LineString([(0, 0), (1, 1)]), None])
