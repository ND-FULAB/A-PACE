import datetime

import pytest

import demo


def _apace_csv_text():
    header = [
        "Date and time:",
        "Notes: café, current in µA",
        "Method Aim",
        "synthetic regression fixture",
        '"Date and time measurement:,2024-04-03 12:24:51"',
        '"V,µA"',
    ]
    data = [f"{index / 1000:.3f},{index + 0.25:.2f}" for index in range(46)]
    return "\n".join(header + data) + "\n"


@pytest.mark.parametrize(
    ("encoding", "expected_encoding"),
    [
        pytest.param("utf-8", "utf-8", id="utf-8-no-bom"),
        pytest.param("utf-8-sig", "utf-8-sig", id="utf-8-bom"),
        pytest.param("utf-16", "utf-16", id="utf-16-bom"),
        pytest.param("utf-16-le", "utf-16-le", id="utf-16le-no-bom"),
        pytest.param("utf-16-be", "utf-16-be", id="utf-16be-no-bom"),
        pytest.param("cp1252", "cp1252", id="cp1252"),
    ],
)
def test_detect_csv_encoding_and_read_apace_csv(tmp_path, encoding, expected_encoding):
    csv_path = tmp_path / f"sample-{encoding}.csv"
    csv_text = _apace_csv_text()
    csv_path.write_bytes(csv_text.encode(encoding))

    detected_encoding = demo.detect_csv_encoding(csv_path)

    assert detected_encoding == expected_encoding
    assert csv_path.read_text(encoding=detected_encoding) == csv_text

    curves, measured_at, curve_count = demo.read_csv_file(csv_path)

    assert curve_count == 1
    assert measured_at == [datetime.datetime(2024, 4, 3, 12, 24, 51)]
    assert curves[0][0] == pytest.approx([value / 1000 for value in range(20, 25)])
    assert curves[1][0] == pytest.approx([value + 0.25 for value in range(20, 25)])


def test_detect_csv_encoding_validates_the_complete_file(tmp_path):
    csv_path = tmp_path / "invalid-after-preview.csv"
    csv_path.write_bytes(
        _apace_csv_text().encode("utf-8")
        + b" " * demo.CSV_ENCODING_SAMPLE_SIZE
        + b"\x81"
    )

    with pytest.raises(UnicodeError, match="Could not determine the encoding"):
        demo.detect_csv_encoding(csv_path)


def test_cp1252_character_after_preview_is_not_misdetected_as_utf8(tmp_path):
    csv_path = tmp_path / "cp1252-after-preview.csv"
    csv_path.write_bytes(
        b"Date and time measurement:,2024-04-03 12:24:51\n"
        + b" " * demo.CSV_ENCODING_SAMPLE_SIZE
        + b"\xb5"
    )

    assert demo.detect_csv_encoding(csv_path) == "cp1252"


def test_invalid_data_after_a_bom_is_not_reinterpreted(tmp_path):
    csv_path = tmp_path / "invalid-utf8-bom.csv"
    csv_path.write_bytes(b"\xef\xbb\xbfvalid prefix\xff")

    with pytest.raises(UnicodeError, match="BOM indicates utf-8-sig"):
        demo.detect_csv_encoding(csv_path)


def test_csv_format_error_is_reported_separately_from_encoding(tmp_path):
    csv_path = tmp_path / "not-apace.csv"
    csv_path.write_text("valid UTF-8, but not an APACE export\n", encoding="utf-8")

    assert demo.detect_csv_encoding(csv_path) == "utf-8"
    with pytest.raises(ValueError, match="expected APACE header"):
        demo.read_csv_file(csv_path)
