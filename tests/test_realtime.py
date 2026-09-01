import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import file_upload
import folder_upload
import real_time_analysis as realtime


class FakeListbox:
    def __init__(self, values=(), selected=()):
        self.values = list(values)
        self.selected = tuple(selected)
        self.after_calls = []

    def get(self, first, last=None):
        if last is not None:
            return tuple(self.values)
        return self.values[first]

    def insert(self, _, value):
        self.values.append(value)

    def delete(self, first, last=None):
        if last is not None:
            self.values.clear()
        else:
            del self.values[first]

    def curselection(self):
        return self.selected

    def after(self, delay, callback, *args):
        self.after_calls.append((delay, callback, args))


class RealTimeTests(unittest.TestCase):
    def _measurement(self, timestamp="2026-08-24 12:00:00"):
        return SimpleNamespace(
            potential_arrays=[np.arange(60, dtype=float)],
            current_arrays=[np.arange(60, dtype=float)],
            timestamp=timestamp,
        )

    def test_metadata_resets_and_missing_fields_are_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "first.pssession")
            second = os.path.join(directory, "second.pssession")
            with open(first, "w", encoding="utf-16le") as output:
                output.write('E_STEP=1.000E-003 FREQ=7.500E+001 E_AMP=1.000E-002 "channel":4,')
            with open(second, "w", encoding="utf-16le") as output:
                output.write('{"measurements": [{}]}')

            processor = realtime.DataProcess(3, "model", "cost", 0.5, 2, [])
            with mock.patch.object(
                realtime.pspyfiles, "load_session_file", return_value=[self._measurement()]
            ):
                _, _, count, freq, amp, step, channel = processor.read_pssession_file(first)
                self.assertEqual((count, freq, amp, step, channel), (1, [75.0], [0.01], [0.001], [4]))
                _, _, count, freq, amp, step, channel = processor.read_pssession_file(second)

            self.assertEqual(count, 1)
            self.assertEqual(freq, [None])
            self.assertEqual(amp, [None])
            self.assertEqual(step, [None])
            self.assertEqual(channel, [None])

    def test_baseline_mwse_uses_squared_signal_range(self):
        raw = np.linspace(7.0, 12.0, 20)

        self.assertTrue(realtime.baseline_fitting_standard([14, 5], raw, raw))
        self.assertFalse(
            realtime.baseline_fitting_standard([14, 5], raw, np.zeros_like(raw))
        )
        self.assertFalse(
            realtime.baseline_fitting_standard([14, 5], np.ones(20), np.ones(20))
        )

    def test_sliding_change_points_drive_mask_baseline_and_peak_region(self):
        state = realtime.RealTimeState(cp_history=[[6.0, 4.0]])
        processor = realtime.DataProcess(2, "model", "cost", 0.5, 2, ["fit"], state=state)
        potential = np.arange(11, dtype=float)
        current = np.arange(11, dtype=float)
        observed = {}

        def fake_fit(_, x, y, order, iterations, weight):
            observed["weight"] = weight.copy()
            return (np.zeros_like(y), None), None

        def fake_standard(boundary, raw, baseline):
            observed["boundary"] = list(boundary)
            return True

        with mock.patch.object(
            realtime.Change_Point_Detection,
            "CPD",
            return_value=([8, 2], [8.0, 2.0], current),
        ), mock.patch.object(realtime, "get_algo_instance", side_effect=fake_fit), mock.patch.object(
            realtime, "baseline_fitting_standard", side_effect=fake_standard
        ):
            result = processor.process_file("sample", [[potential], [current]], 1, 1)

        self.assertEqual(result[1][0], [7, 3])
        self.assertEqual(result[2][0], [7.0, 3.0])
        self.assertEqual(observed["boundary"], [7, 3])
        np.testing.assert_array_equal(
            observed["weight"],
            np.array([True, True, True, False, False, False, False, True, True, True, True]),
        )

    def test_handler_filters_retries_failures_and_deduplicates_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sample.PSSESSION")
            with open(path, "wb") as output:
                output.write(b"first")
            processor = SimpleNamespace(state=realtime.RealTimeState())
            handler = realtime.RealTimeAnalysis(
                mock.Mock(), processor, stability_checker=lambda _: True
            )
            handler.process_new_file = mock.Mock(side_effect=[ValueError("partial"), None, None])
            txt_event = SimpleNamespace(src_path="ignored.txt", is_directory=False)
            event = SimpleNamespace(src_path=path, is_directory=False)

            handler.on_created(txt_event)
            handler.on_created(event)
            handler.on_modified(event)
            handler.on_created(event)
            self.assertEqual(handler.process_new_file.call_count, 2)

            with open(path, "ab") as output:
                output.write(b"-completed-later")
            handler.on_modified(event)

            self.assertEqual(handler.process_new_file.call_count, 3)

    def test_reprocessing_a_path_replaces_state_instead_of_appending(self):
        state = realtime.RealTimeState()
        state.replace_file("sample.pssession", ["t1"], [1], [0.5], [1.5], [[6, 4]])
        state.replace_file("sample.pssession", ["t2"], [2], [1.5], [2.5], [[8, 2]])

        self.assertEqual(state.snapshot_records(), [("t2", 2, 1.5, 2.5)])
        self.assertEqual(state.recent_cp_values(2), [[8.0, 2.0]])

    def test_write_during_processing_is_not_marked_as_the_new_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "growing.pssession")
            with open(path, "wb") as output:
                output.write(b"stable-prefix")
            processor = SimpleNamespace(state=realtime.RealTimeState())
            handler = realtime.RealTimeAnalysis(
                mock.Mock(), processor, stability_checker=lambda _: True
            )
            calls = 0

            def process_and_grow(_):
                nonlocal calls
                calls += 1
                if calls == 1:
                    with open(path, "ab") as output:
                        output.write(b"-late-write")

            handler.process_new_file = mock.Mock(side_effect=process_and_grow)
            event = SimpleNamespace(src_path=path, is_directory=False)

            handler.on_created(event)
            handler.on_modified(event)

            self.assertEqual(handler.process_new_file.call_count, 2)

    def test_stability_check_and_flat_four_column_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "stable.pssession")
            with open(path, "wb") as output:
                output.write(b"complete")
            self.assertTrue(
                realtime.wait_for_file_stable(path, timeout=0.2, interval=0.005, stable_observations=2)
            )

            csv_path = os.path.join(directory, "out.csv")
            plotter = realtime.RealTimePlotter.__new__(realtime.RealTimePlotter)
            plotter.save_data(csv_path, [("t1", 1, 0.5, 1.5), ("t2", 2, 1.5, 2.5)])
            with open(csv_path, newline="", encoding="utf-8") as source:
                rows = list(csv.reader(source))
            self.assertEqual(rows[0], ["Time", "Peak Mean", "Peak Min", "Peak Max"])
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(len(row) == 4 for row in rows))

    def test_window_close_stops_observer_and_performs_final_save(self):
        class FakeObserver:
            instance = None

            def __init__(self):
                self.stopped = False
                self.joined = False
                self.__class__.instance = self

            def schedule(self, *args, **kwargs):
                pass

            def start(self):
                pass

            def stop(self):
                self.stopped = True

            def join(self):
                self.joined = True

        class FakePlotter:
            instance = None

            def __init__(self, plot_queue):
                self.fig = object()
                self.periodic_stopped = False
                self.saved = []
                self.__class__.instance = self

            def start_periodic_save(self, *args):
                pass

            def stop_periodic_save(self):
                self.periodic_stopped = True

            def save_plot(self, path):
                self.saved.append(path)

            def save_data(self, path, data):
                self.saved.append(path)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            realtime, "Observer", FakeObserver
        ), mock.patch.object(realtime, "RealTimePlotter", FakePlotter), mock.patch.object(
            realtime.plt, "show"
        ), mock.patch.object(realtime.plt, "close"):
            realtime.run_real_time_analysis(directory, 2, "model", "cost", 0.5, 2, [])

        self.assertTrue(FakeObserver.instance.stopped)
        self.assertTrue(FakeObserver.instance.joined)
        self.assertTrue(FakePlotter.instance.periodic_stopped)
        self.assertEqual(FakePlotter.instance.saved, ["output_plot.png", "output_data.csv"])


class UploadTests(unittest.TestCase):
    def test_file_upload_defaults_and_selected_delete_are_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = os.path.join(directory, "database", "uploaded_files.json")
            self.assertEqual(file_upload.read_json(storage), {"csv": [], "pssession": []})
            os.makedirs(os.path.dirname(storage), exist_ok=True)
            with open(storage, "w", encoding="utf-8") as output:
                json.dump({"csv": []}, output)
            self.assertEqual(file_upload.read_json(storage), {"csv": [], "pssession": []})
            file_upload.write_json(storage, {"csv": ["one.csv"], "pssession": ["two.pssession"]})
            listbox = FakeListbox(["one.csv", "two.pssession"], selected=(0,))
            with mock.patch.object(file_upload, "json_file_path", storage), mock.patch.object(
                file_upload, "file_list", listbox
            ), mock.patch.object(file_upload.messagebox, "showinfo"):
                file_upload.delete_selected_files()

            self.assertEqual(file_upload.read_json(storage), {"csv": [], "pssession": ["two.pssession"]})
            self.assertEqual(listbox.values, ["two.pssession"])

    def test_folder_handler_uses_tk_queue_and_strict_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = os.path.join(directory, "uploaded.json")
            listbox = FakeListbox()
            handler = folder_upload.FileMonitorHandler(directory, "pssession", listbox, storage)
            handler._record(os.path.join(directory, "ignored.pssession.bak"))
            accepted = os.path.join(directory, "accepted.PSSESSION")
            handler._record(accepted)

            self.assertEqual(listbox.values, [])
            self.assertEqual(len(listbox.after_calls), 1)
            _, callback, args = listbox.after_calls[0]
            callback(*args)
            self.assertEqual(listbox.values, [os.path.normpath(accepted)])
            self.assertEqual(folder_upload.read_json(storage, folder_upload.DEFAULT_FILES)["pssession"], [os.path.normpath(accepted)])

    def test_switching_folder_stops_previous_observer(self):
        class FakeObserver:
            instances = []

            def __init__(self):
                self.stopped = False
                self.joined = False
                self.__class__.instances.append(self)

            def schedule(self, *args, **kwargs):
                pass

            def start(self):
                pass

            def stop(self):
                self.stopped = True

            def join(self):
                self.joined = True

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            folder_upload, "Observer", FakeObserver
        ):
            folder_upload.stop_monitoring()
            listbox = FakeListbox()
            folder_upload.start_monitoring(directory, "pssession", listbox, os.path.join(directory, "one.json"))
            first = FakeObserver.instances[-1]
            folder_upload.start_monitoring(directory, "pssession", listbox, os.path.join(directory, "two.json"))
            self.assertTrue(first.stopped)
            self.assertTrue(first.joined)
            folder_upload.stop_monitoring()


if __name__ == "__main__":
    unittest.main()
