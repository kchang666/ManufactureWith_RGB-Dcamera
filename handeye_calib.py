"""
手眼校正 (AX = XB)：求相機與夾爪 (或手臂基座) 的轉換矩陣

  Eye-in-Hand (相機裝在夾爪上) : ArUco 固定在桌上，求 T_gripper<-cam
  Eye-to-Hand (相機固定在架子)  : ArUco 貼在夾爪上，求 T_base<-cam

用法：
  python handeye_calib.py --mode eye_in_hand            # 收集資料 + 求解
  python handeye_calib.py --mode eye_in_hand --solve    # 用已存的 samples 重新求解

收集時：用教導器 / keyboard_jog_example 把手臂移到不同姿態 (位置不同、
       rx/ry 傾斜 ±10~20°、rz 轉 ±30° 以上)，ArUco 要在畫面中，
       停穩後在影像視窗按 s 存一筆。至少 10 筆，建議 15~20 筆。
       按 c 計算，按 q 離開。
"""
import argparse
import json
import os

import cv2
import numpy as np

import pen_vision as pv

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES_FILE = os.path.join(HERE, "handeye_samples.json")
RESULT_FILE = os.path.join(HERE, "handeye.json")
ROBOT_IP = "192.168.1.1"

METHODS = {} if not hasattr(cv2, "CALIB_HAND_EYE_TSAI") else {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _log_R(R):
    return cv2.Rodrigues(R)[0].ravel()


def handeye_numpy(R_g2b, t_g2b, R_t2c, t_t2c):
    """AX = XB 的 numpy 解法 (Park & Martin)，輸入輸出格式同 cv2.calibrateHandEye
    G_i·X·C_i = 常數  →  (G_j⁻¹G_i)·X = X·(C_j·C_i⁻¹)"""
    G = [np.vstack([np.hstack([R, np.reshape(t, (3, 1))]), [0, 0, 0, 1]]) for R, t in zip(R_g2b, t_g2b)]
    C = [np.vstack([np.hstack([R, np.reshape(t, (3, 1))]), [0, 0, 0, 1]]) for R, t in zip(R_t2c, t_t2c)]
    As, Bs = [], []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            As.append(np.linalg.inv(G[j]) @ G[i])
            Bs.append(C[j] @ np.linalg.inv(C[i]))
    # 旋轉：alpha = R_X·beta，用 SVD 求最佳旋轉
    H = sum(np.outer(_log_R(B[:3, :3]), _log_R(A[:3, :3])) for A, B in zip(As, Bs))
    U, _, Vt = np.linalg.svd(H)
    Rx = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    # 平移：(R_A - I)·t_X = R_X·t_B - t_A，最小平方
    M = np.vstack([A[:3, :3] - np.eye(3) for A in As])
    b = np.hstack([Rx @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
    tx = np.linalg.lstsq(M, b, rcond=None)[0]
    return Rx, tx.reshape(3, 1)


def _solvers():
    """可用的解法：numpy 版一定有；若 OpenCV 有 calibrateHandEye 也一起比較"""
    out = {"PARK_NUMPY": handeye_numpy}
    if hasattr(cv2, "calibrateHandEye"):
        for name, m in METHODS.items():
            out[name] = (lambda m: lambda *a: cv2.calibrateHandEye(*a, method=m))(m)
    return out


def residual(mode, X, T_bg_list, T_cm_list):
    """校正品質：理論上固定不動的 ArUco，換算到固定座標系後應該在同一點"""
    pts, rots = [], []
    for T_bg, T_cm in zip(T_bg_list, T_cm_list):
        if mode == "eye_in_hand":
            T = T_bg @ X @ T_cm                     # T_base<-marker，應固定
        else:
            T = np.linalg.inv(T_bg) @ X @ T_cm      # T_gripper<-marker，應固定
        pts.append(T[:3, 3])
        rots.append(T[:3, :3])
    pts = np.array(pts)
    mean = pts.mean(axis=0)
    err = np.linalg.norm(pts - mean, axis=1)
    R0 = rots[0]
    ang = [np.degrees(np.arccos(np.clip((np.trace(R0.T @ R) - 1) / 2, -1, 1))) for R in rots]
    return float(err.mean()), float(err.max()), float(np.max(ang)), mean


def solve(samples, mode):
    if len(samples) < 5:
        raise SystemExit("樣本太少 (至少 5 筆，建議 15 筆以上)")
    T_cm = [pv.rt_to_T(s["rvec"], s["tvec"]) for s in samples]
    best = None
    for conv in pv.EULER_CANDIDATES:
        T_bg = [pv.pose_to_T(s["tcp"], conv) for s in samples]
        A = T_bg if mode == "eye_in_hand" else [np.linalg.inv(T) for T in T_bg]
        for mname, fn in _solvers().items():
            try:
                R, t = fn([T[:3, :3] for T in A], [T[:3, 3] for T in A],
                          [T[:3, :3] for T in T_cm], [T[:3, 3] for T in T_cm])
            except (cv2.error, np.linalg.LinAlgError):
                continue
            X = np.eye(4); X[:3, :3] = R; X[:3, 3] = t.ravel()
            e_mean, e_max, a_max, marker = residual(mode, X, T_bg, T_cm)
            print(f"  {conv:7s} {mname:10s}  平均誤差 {e_mean:6.2f} mm  最大 {e_max:6.2f} mm  角度 {a_max:5.2f}°")
            if best is None or e_mean < best["err_mean"]:
                best = dict(mode=mode, euler=conv, method=mname, T=X.tolist(),
                            err_mean=e_mean, err_max=e_max, marker=marker.tolist())
    print("\n=== 最佳結果 ===")
    print(f"尤拉角順序: {best['euler']}   方法: {best['method']}")
    print(f"平均誤差 {best['err_mean']:.2f} mm, 最大 {best['err_max']:.2f} mm "
          f"({'< 3mm 很好' if best['err_mean'] < 3 else '偏大，建議重收姿態更分散的資料'})")
    name = "T_gripper<-cam" if mode == "eye_in_hand" else "T_base<-cam"
    print(f"{name} =\n{np.round(np.array(best['T']), 3)}")
    if mode == "eye_in_hand":
        m = best["marker"]
        print(f"\n驗證：ArUco 中心在手臂座標約為 x={m[0]:.1f}, y={m[1]:.1f}, z={m[2]:.1f}")
        print("      把夾爪尖端移到 ArUco 中心，TCP 讀值應該很接近這組數字。")
    with open(RESULT_FILE, "w") as f:
        json.dump(best, f, indent=2)
    print("已儲存", RESULT_FILE)
    return best


def collect(mode, marker_id, length):
    from pymodbus.client import ModbusTcpClient
    from drv_modbus import request
    c = ModbusTcpClient(host=ROBOT_IP, port=502, unit_id=2)
    if not c.connect():
        raise SystemExit("無法連線到手臂")
    cam = pv.Camera()
    samples = []
    if os.path.exists(SAMPLES_FILE) and input("發現舊的樣本，要接續使用嗎? (y/N) ").lower() == "y":
        samples = json.load(open(SAMPLES_FILE))["samples"]
    print(__doc__)
    while True:
        img, _ = cam.read()
        res = pv.marker_pose(img, cam.K, cam.D, marker_id, length)
        show = img.copy()
        if res:
            rvec, tvec, corners = res
            cv2.drawFrameAxes(show, cam.K, cam.D, rvec, tvec, length)
            cv2.putText(show, f"dist {np.linalg.norm(tvec):.0f} mm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(show, f"samples: {len(samples)}  [s]save [c]compute [q]quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("hand-eye calib", show)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            if not res:
                print("看不到 ArUco，沒存")
                continue
            # 多張平均降低 ArUco 雜訊
            rs_, ts_ = [], []
            for _ in range(10):
                r = pv.marker_pose(cam.read()[0], cam.K, cam.D, marker_id, length)
                if r:
                    rs_.append(r[0]); ts_.append(r[1])
            tcp = list(request.Get_TCP_Pose(c))
            samples.append(dict(tcp=tcp, rvec=np.median(rs_, 0).tolist(),
                                tvec=np.median(ts_, 0).tolist()))
            json.dump(dict(mode=mode, K=cam.K.tolist(), samples=samples),
                      open(SAMPLES_FILE, "w"), indent=2)
            print(f"第 {len(samples)} 筆  TCP={np.round(tcp, 2)}")
        elif k == ord("c"):
            solve(samples, mode)
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()
    cam.stop()
    c.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eye_in_hand", "eye_to_hand"], default="eye_in_hand")
    ap.add_argument("--marker-id", type=int, default=1)
    ap.add_argument("--marker-len", type=float, default=50.0, help="ArUco 黑框邊長 (mm)，已實測 5 cm")
    ap.add_argument("--solve", action="store_true", help="只用已存樣本重新求解")
    a = ap.parse_args()
    if a.solve:
        solve(json.load(open(SAMPLES_FILE))["samples"], a.mode)
    else:
        collect(a.mode, a.marker_id, a.marker_len)
