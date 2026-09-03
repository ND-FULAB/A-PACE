import pytest

import app as webapp


@pytest.mark.parametrize("descending", [False, True])
def test_multi_peak_overlay_retains_both_cp_samples_and_isolates_each_peak(
    monkeypatch, descending
):
    potential = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
    current = [0, 1, 2, 3, 2, 1, 2, 4, 2, 1, 0]
    if descending:
        potential.reverse()
        current.reverse()

    results = {}
    references = []
    expected = []
    intervals = [(6, 9), (1, 4)] if descending else [(1, 4), (6, 9)]
    for number, ((lower, upper), suffix) in enumerate(
        zip(intervals, ("First", "Second")), start=1
    ):
        name = f"two.csv-{suffix}"
        results[name] = {
            "Curve No. 1": {
                "Raw Poetntial ": potential,
                "Raw Current": current,
                "Baseline Mean ": [0.25] * len(potential),
                "Source File Path": "two.csv",
                "Peak Number": number,
                "Change Point Indexes ": [upper, lower],
                "Change Point Values ": sorted([potential[lower], potential[upper]]),
                "review_status": "pass",
            }
        }
        references.append({"file_name": name, "curve_no": "Curve No. 1"})
        expected.append((potential[lower : upper + 1], current[lower : upper + 1]))
    monkeypatch.setattr(webapp, "load_results", lambda: results)

    response = webapp.app.test_client().post(
        "/post_exp/overlay_graphs", json={"graphs": references}
    )

    assert response.status_code == 200
    traces = response.get_json()["figure"]["data"]
    assert len(traces) == 2
    for trace, (expected_x, expected_y), reference in zip(traces, expected, references):
        assert trace["x"] == pytest.approx(expected_x)
        assert trace["y"] == pytest.approx([value - 0.25 for value in expected_y])
        assert trace["name"] == f"{reference['file_name']} Curve No. 1"
