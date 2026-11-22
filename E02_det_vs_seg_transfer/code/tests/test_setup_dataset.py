import numpy as np

from e02.setup_dataset import gapped_label, label_checks, parse_name, rasterize


def sq(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def test_pixel_centre_rule():
    inst, cnt = rasterize([sq(2, 3, 5, 6)], shape=(10, 10))
    # pixels whose centres (c+0.5, r+0.5) lie in [2,5]x[3,6] -> cols 2..4, rows 3..5
    assert inst.sum() == 9 and inst[3:6, 2:5].all()
    assert cnt == {"n_polygons": 1, "n_instances": 1, "n_lost": 0, "overlap_px": 0}


def test_overlap_goes_to_smaller_and_ids_are_compact():
    # 1: big 6x6 square; 2: tiny polygon (no pixel); 3: 2x2 square on top of 1
    inst, cnt = rasterize([sq(0, 0, 6, 6), sq(8, 8, 8.2, 8.2), sq(1, 1, 3, 3)], shape=(10, 10))
    assert cnt["overlap_px"] == 4 and (inst[1:3, 1:3] == 2).all() and (inst[0, 0:6] == 1).all()
    assert cnt["n_lost"] == 1 and cnt["n_instances"] == 2 and set(np.unique(inst)) == {0, 1, 2}


def test_equal_area_overlap_goes_to_higher_annotation():
    inst, _ = rasterize([sq(0, 0, 4, 4), sq(2, 0, 6, 4)], shape=(10, 10))
    assert (inst[0:4, 2:4] == 2).all()


def test_gap_larger_instance_gives_way():
    inst = np.zeros((7, 9), np.uint16)
    inst[1:5, 1:5] = 1           # 16 px
    inst[1:5, 5:8] = 2           # 12 px, touches 1 along column 4|5
    inst[0, 0] = 3               # 1 px, diagonal to 1's corner
    lab = gapped_label(inst)
    assert (lab[1:5, 4] == 0).all() and lab[1:5, 1:4].sum() == 11 and lab[1:5, 5:8].all()
    assert lab[0, 0] == 1 and lab[1, 1] == 0     # the 1-px cell stays; the big one gives way
    chk = label_checks(inst, lab)
    assert chk["n_cc_multi_instance"] == 0 and chk["n_empty_after_gap"] == 0


def test_gap_equal_area_lower_id_gives_way():
    inst = np.zeros((4, 6), np.uint16)
    inst[1:3, 1:3] = 1
    inst[1:3, 3:5] = 2
    lab = gapped_label(inst)
    assert (lab[1:3, 2] == 0).all() and lab[1:3, 3:5].all()


def test_parse_name():
    m = parse_name("SKOV3_Phase_E4_2_01d08h00m_3.tif")
    assert m == {"cell_line": "SKOV3", "well": "SKOV3_E4", "fov": "SKOV3_E4_2", "time": "01d08h00m", "crop": "3"}
