"""Offline V4 regression tests. No camera, Modbus connection, or motion commands.

Run: python -m unittest discover -s tests -p test_project_v4_logic.py
Extract the logic using AST so these tests also run without hardware SDKs/OpenCV.
"""
import ast
import json
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'project_v4.py'


def load_logic():
    source = SOURCE.read_text(encoding='utf-8')
    compile(source, str(SOURCE), 'exec')
    names = {'match_points', 'point_label', 'MultiDotTracker', 'PointSelection',
             'build_machining_path', 'MultiDotVisit', 'SelectionUI', 'pixel_to_robot_pose',
             'RobotMotionSession', 'move_to_pose', 'check_motion_response'}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    fake_cv = SimpleNamespace(EVENT_MOUSEWHEEL=10, EVENT_LBUTTONDOWN=1,
                              FONT_HERSHEY_SIMPLEX=0, LINE_AA=16,
                              INTER_AREA=3, putText=Mock(), rectangle=Mock(),
                              resize=Mock(side_effect=lambda image, size, **kwargs:
                                          np.zeros((size[1], size[0], 3), dtype=image.dtype)))
    scope = dict(np=np, sys=sys, threading=threading, json=json, datetime=datetime,
                 timezone=timezone, textwrap=textwrap, cv2=fake_cv, time=time,
                 INITIAL_SPEED=20, INITIAL_MOVE_TIMEOUT=60,
                 ROBOT_HOST='offline', ROBOT_PORT=502, ModbusTcpClient=Mock(),
                 BinaryPayloadBuilder=Mock(), Endian=SimpleNamespace(BIG='big', LITTLE='little'),
                 request=SimpleNamespace(Get_Pose_Flag=Mock(return_value=1), Get_TCP_Pose=Mock()),
                 DOT_CONTACT_Z=463.084, DOT_TRAVEL_Z=500., DOT_MOVE_SPEED=20,
                 DOT_CONTACT_SPEED=10, DOT_DWELL=0, DOT_STABLE_FRAMES=20,
                 DOT_MAX_JITTER_PX=2., CLICK_RADIUS_PX=20, PANEL_WIDTH=430,
                 INITIAL_POINT=(393.085, 5.232, 730.939, -179.366, 1.545, -105.652))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope


class V4Tests(unittest.TestCase):
    def setUp(self):
        self.ns = load_logic()

    def lock(self, count):
        tracker = self.ns['MultiDotTracker']()
        points = np.array([(100. + i * 30, 150. + (i % 3) * 30) for i in range(count)])
        for i in range(20):
            tracker.update((points[::-1] + (0.2 if i % 2 else -0.2)).tolist(), True)
        self.assertTrue(tracker.valid)
        np.testing.assert_allclose(tracker.pixel_points, points)
        return tracker, points

    def path(self, count):
        targets = [(i + 1, (float(i * 3), float(i * 4), 463.084, 180., 0., 0.))
                   for i in range(count)]
        return self.ns['build_machining_path'](targets)

    def test_any_count_and_fixed_ids(self):
        for count in (1, 2, 5, 30):
            tracker, points = self.lock(count)
            tracker.update(points[::-1].tolist(), True)
            np.testing.assert_allclose(tracker.pixel_points, points)
            self.assertTrue(tracker.valid)
        self.assertEqual([self.ns['point_label'](i) for i in (0, 25, 26, 701)],
                         ['A', 'Z', 'AA', 'ZZ'])

    def test_scene_changes_require_reset(self):
        for change in ('count', 'position', 'pose'):
            tracker, points = self.lock(5)
            altered = points[:-1] if change == 'count' else points + 10 if change == 'position' else points
            tracker.update(altered.tolist(), change != 'pose')
            self.assertTrue(tracker.invalidated)
            self.assertFalse(tracker.valid)
            tracker.update(points.tolist(), True)
            self.assertFalse(tracker.valid)
            tracker.reset()
            self.assertIsNone(tracker.pixel_points)
            self.assertFalse(tracker.invalidated)

    def test_count_change_while_collecting_and_ambiguous_match(self):
        tracker = self.ns['MultiDotTracker']()
        for _ in range(19):
            tracker.update([(10, 10), (30, 30)], True)
        tracker.update([(10, 10)], True)
        self.assertEqual(len(tracker.samples), 1)
        self.assertIsNone(tracker.pixel_points)
        self.assertIsNone(self.ns['match_points'](np.array([[0, 0], [1, 0]]), [(0, 0), (1, 0)]))

    def test_selection_order_subset_duplicate_undo_clear(self):
        _, points = self.lock(5)
        selection = self.ns['PointSelection']()
        for index in (2, 0, 4, 2):
            selection.click(*points[index], points)
        self.assertEqual(selection.order, [2, 0, 4])
        selection.click(-100, -100, points)
        self.assertEqual(selection.order, [2, 0, 4])
        np.testing.assert_allclose(selection.selected_points(points), points[[2, 0, 4]])
        selection.undo()
        self.assertEqual(selection.order, [2, 0])
        selection.clear()
        selection.undo()
        self.assertEqual(selection.selected_points(points).shape, (0, 2))

    def test_variable_length_path_and_bad_targets(self):
        for count in (2, 3, 8, 30):
            path = self.path(count)
            self.assertEqual(len(path['moves']), count + 2)
            self.assertEqual(path['total_cut_length_mm'], 5 * (count - 1))
            self.assertEqual([m['type'] for m in path['moves']],
                             ['travel', 'plunge'] + ['cut'] * (count - 1) + ['retract'])
            self.assertTrue(all(m['command'] == 302 for m in path['moves']))
            self.assertFalse(path['closed'])
            self.assertEqual(path['moves'][-1]['tcp_pose'][2], 500.)
        for targets in ([], [(1, (0, 0, 463.084, 0, 0, 0))],
                        [(1, (0, 0, 463.084, 0, 0, 0))] * 2,
                        [(1, (0, 0, 0, 0, 0, 0)), (2, (1, 1, 463.084, 0, 0, 0))]):
            with self.assertRaises(ValueError):
                self.ns['build_machining_path'](targets)

    def test_execution_cancel_and_failure(self):
        path = self.path(5)
        for cancel_after in (None, 0, 1, 2, 3, 5, 6):
            visit = self.ns['MultiDotVisit']()
            calls = []

            def move(target, label, **kwargs):
                calls.append(tuple(target))
                if len(calls) == cancel_after:
                    visit.cancel.set()

            self.ns['move_to_pose'] = move
            if cancel_after == 0:
                visit.cancel.set()
            visit._run(path)
            if cancel_after is None:
                self.assertEqual(calls, [tuple(m['tcp_pose']) for m in path['moves']])
                self.assertTrue(visit.status.startswith('DONE'))
            else:
                self.assertTrue(visit.status.startswith('Cancelled'))
                if cancel_after >= 2:
                    self.assertEqual(calls[-1][2], 500.)
                    self.assertEqual(list(calls[-1][:2]), path['moves'][cancel_after - 1]['tcp_pose'][:2])
        visit = self.ns['MultiDotVisit']()
        move = Mock(side_effect=RuntimeError('simulated failure'))
        self.ns['move_to_pose'] = move
        visit._run(path)
        self.assertEqual(move.call_count, 1)
        self.assertTrue(visit.status.startswith('FAILED'))

    def test_path_reuses_connection_and_writes_speed_only_on_change(self):
        path = self.path(3)
        client = self.ns['ModbusTcpClient'].return_value
        ok = SimpleNamespace(isError=lambda: False)
        client.write_register.return_value = ok
        client.write_registers.return_value = ok
        self.ns['request'].Get_TCP_Pose.side_effect = [m['tcp_pose'] for m in path['moves']]
        visit = self.ns['MultiDotVisit']()
        visit._run(path)
        self.assertTrue(visit.status.startswith('DONE'), visit.status)
        client.connect.assert_called_once()
        client.close.assert_called_once()
        writes = [call.args for call in client.write_register.call_args_list]
        self.assertEqual([args for args in writes if args[0] == 0x0324],
                         [(0x0324, 20, 2), (0x0324, 10, 2)])
        self.assertEqual([args for args in writes if args[0] == 0x0300],
                         [(0x0300, 302, 2)] * len(path['moves']))
        self.assertEqual(client.write_registers.call_count, len(path['moves']))

    def test_speed_failure_stops_before_coordinates_or_motion(self):
        client = self.ns['ModbusTcpClient'].return_value
        client.write_register.return_value = SimpleNamespace(isError=lambda: True, exception_code=6)
        with self.ns['RobotMotionSession']() as motion:
            with self.assertRaisesRegex(RuntimeError, '0x0324.*exception_code=6'):
                motion.move((1, 2, 463.084, 180, 0, 0), 'offline test')
            self.assertIsNone(motion.speed)
        client.write_registers.assert_not_called()
        self.assertEqual(client.write_register.call_count, 1)
        client.close.assert_called_once()

    def test_motion_command_failure_is_not_retried(self):
        client = self.ns['ModbusTcpClient'].return_value
        ok = SimpleNamespace(isError=lambda: False)
        client.write_registers.return_value = ok
        client.write_register.side_effect = [ok, None]
        visit = self.ns['MultiDotVisit']()
        visit._run(self.path(3))
        self.assertTrue(visit.status.startswith('FAILED'))
        self.assertEqual(client.write_register.call_count, 2)
        self.ns['request'].Get_TCP_Pose.assert_not_called()
        client.close.assert_called_once()

    def test_start_exports_selected_labels_and_exact_executed_moves(self):
        points = np.array([[10., 10.], [30., 30.], [50., 50.]])
        selected = points[[2, 0]]
        self.ns['build_dot_targets'] = lambda calibration, values: tuple(
            (i, (float(p[0]), float(p[1]), 463.084, 180., 0., 0.))
            for i, p in enumerate(values, 1))
        calls = []
        self.ns['move_to_pose'] = lambda target, label, **kw: calls.append(list(target))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'path.json'
            self.ns['DOT_TARGET_PATH'] = output
            calibration = SimpleNamespace(homography=np.eye(3))
            visit = self.ns['MultiDotVisit']()
            visit.start(calibration, selected, ['C', 'A'], points)
            visit.thread.join()
            data = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(data['detected_point_count'], 3)
            self.assertEqual(data['selected_point_count'], 2)
            self.assertEqual(data['machining_path']['marker_order'], ['C', 'A'])
            self.assertEqual([t['marker'] for t in data['targets']], ['C', 'A'])
            self.assertEqual(calls, [m['tcp_pose'] for m in data['machining_path']['moves']])
            visit.start(calibration, selected, ['C', 'A'], points)
            self.assertEqual(len(calls), 4)  # One job per application run.

    def test_sidebar_and_mouse_actions_preserve_image_coordinates(self):
        tracker, points = self.lock(3)
        ui = self.ns['SelectionUI']()
        selection = self.ns['PointSelection']()
        visit = self.ns['MultiDotVisit']()
        canvas = ui.render(np.zeros((480, 640, 3), dtype=np.uint8), tracker,
                           selection, visit, SimpleNamespace())
        self.assertEqual(canvas.shape, (720, 1070, 3))
        ui.on_mouse(1, 100, 150, 0, None)
        action, pixel = ui.actions.pop()
        self.assertEqual((action, pixel), ('click', (100, 150)))
        selection.click(*pixel, points)
        self.assertEqual(selection.order, [0])
        start_rect = dict(ui.buttons)['start']
        ui.on_mouse(1, start_rect[0] + 5, start_rect[1] + 5, 0, None)
        self.assertEqual(ui.actions, [('start', None)])
        self.assertFalse(visit.started)  # Mouse callback itself never commands motion.

    def test_high_resolution_preview_maps_clicks_to_camera_coordinates(self):
        for width, height in ((1920, 1080), (1280, 720), (641, 1081)):
            tracker, _ = self.lock(3)
            ui = self.ns['SelectionUI']()
            selection = self.ns['PointSelection']()
            visit = self.ns['MultiDotVisit']()
            canvas = ui.render(np.zeros((height, width, 3), dtype=np.uint8),
                               tracker, selection, visit, SimpleNamespace())
            self.assertLessEqual(canvas.shape[1], 1280)
            self.assertEqual(canvas.shape[0], 720)
            pw, ph = ui.preview_size
            # Use the rounded display location of a real camera point.
            x, y = tracker.pixel_points[0]
            ui.on_mouse(1, round(x * pw / width), round(y * ph / height), 0, None)
            action, pixel = ui.actions.pop()
            self.assertEqual(action, 'click')
            self.assertTrue(selection.click(*pixel, tracker.pixel_points))
            self.assertEqual(selection.order, [0])
            self.assertFalse(selection.click(*pixel, tracker.pixel_points))
            for name, (x1, y1, x2, y2) in ui.buttons:
                self.assertLessEqual(x2, canvas.shape[1])
                ui.on_mouse(1, x1 + 5, y1 + 5, 0, None)
                self.assertEqual(ui.actions.pop(), (name, None))
            # Blank space under the camera image and panel titles are not points.
            ui.on_mouse(1, 5, ph, 0, None)
            ui.on_mouse(1, pw + 5, 10, 0, None)
            self.assertEqual(ui.actions, [])
            self.assertFalse(visit.started)


if __name__ == '__main__':
    unittest.main()
