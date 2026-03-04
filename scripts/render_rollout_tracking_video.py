#!/usr/bin/env python3
"""
Render a rollout NPZ to MP4 using either:
  --mode states  (replay x_traj=[qpos|qvel])
  --mode actions (roll out u_traj from (qpos0,qvel0) with mj_step)

Usage:
  python3 scripts/render_rollout_tracking_video.py runs/eval_rollouts_npz/ep_000.npz --mode states
  python3 scripts/render_rollout_tracking_video.py runs/eval_rollouts_npz/ep_000.npz --mode actions
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import mujoco as mj
from mujoco.glfw import glfw


def _ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg not found on PATH. Please install ffmpeg.")


def _open_ffmpeg(out_path: str, width: int, height: int, fps: int):
    _ensure_ffmpeg()
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
        "-vf", "vflip,scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _init_hidden_glfw_context():
    if not glfw.init():
        raise RuntimeError("GLFW init failed (needed for OpenGL context).")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    win = glfw.create_window(64, 64, "", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("Failed to create hidden GLFW window/context.")
    glfw.make_context_current(win)
    return win


def _close_hidden_glfw_context(win):
    try:
        glfw.destroy_window(win)
    except Exception:
        pass
    try:
        glfw.terminate()
    except Exception:
        pass


def _prepare_renderer(model: mj.MjModel, width: int, height: int):
    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)

    opt = mj.MjvOption()
    mj.mjv_defaultOption(opt)

    scene = mj.MjvScene(model, maxgeom=20000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, context)
    if context.offWidth != width or context.offHeight != height:
        mj.mjr_resizeOffscreen(width, height, context)

    viewport = mj.MjrRect(0, 0, width, height)
    rgb_buf = np.empty((height, width, 3), dtype=np.uint8)
    return cam, opt, scene, context, viewport, rgb_buf


def _free_renderer(scene, context):
    try:
        mj.mjr_freeContext(context)
    except Exception:
        pass
    try:
        mj.mjv_freeScene(scene)
    except Exception:
        pass


def _set_state_from_x(model: mj.MjModel, data: mj.MjData, x: np.ndarray):
    nq, nv = model.nq, model.nv
    data.qpos[:] = x[:nq]
    data.qvel[:] = x[nq:nq + nv]


def _set_state_from_qpos_qvel(data: mj.MjData, qpos: np.ndarray, qvel: np.ndarray):
    data.qpos[:] = qpos
    data.qvel[:] = qvel


def _apply_tracking_camera(cam: mj.MjvCamera, model: mj.MjModel, body_name: str,
                           *, distance=1.2, azimuth=90.0, elevation=-25.0):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body '{body_name}' not found in model.")
    cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = bid
    cam.distance = float(distance)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)


def _rotmat_from_xdir(xdir: np.ndarray) -> np.ndarray:
    x = np.asarray(xdir, dtype=float)
    n = np.linalg.norm(x)
    if n < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x / n

    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    yn = np.linalg.norm(y)
    if yn < 1e-9:
        z = np.array([0.0, 1.0, 0.0])
        y = np.cross(z, x)
        y = y / (np.linalg.norm(y) + 1e-12)
        z = np.cross(x, y)
        z = z / (np.linalg.norm(z) + 1e-12)
    else:
        y = y / yn
        z = np.cross(x, y)
        z = z / (np.linalg.norm(z) + 1e-12)

    return np.column_stack([x, y, z])


def _add_goal_geoms(
    scene: mj.MjvScene,
    goal: np.ndarray,
    *,
    z=0.02,
    radius=0.03,
    arrow_len=0.20,
    xy_thresh: float | None = None,
    yaw_thresh: float | None = None,
):
    if goal is None or len(goal) < 3:
        return

    x, y, yaw = float(goal[0]), float(goal[1]), float(goal[2])

    # We'll add up to 5 geoms (sphere + arrow + optional ring + optional yaw wedge (2 arms)).
    if scene.ngeom + 5 >= scene.maxgeom:
        return

    # sphere
    g = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mj.mjv_initGeom(
        g, mj.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0, 0], dtype=np.float32),
        pos=np.array([x, y, z], dtype=np.float32),
        mat=np.eye(3, dtype=np.float32).flatten(),
        rgba=np.array([1.0, 0.1, 0.1, 0.9], dtype=np.float32),
    )
    g.category = mj.mjtCatBit.mjCAT_DECOR

    # arrow
    xdir = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    R = _rotmat_from_xdir(xdir).astype(np.float32)
    g = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mj.mjv_initGeom(
        g, mj.mjtGeom.mjGEOM_ARROW,
        size=np.array([0.01, arrow_len, 0.01], dtype=np.float32),
        pos=np.array([x, y, z], dtype=np.float32),
        mat=R.flatten(),
        rgba=np.array([1.0, 0.3, 0.3, 0.9], dtype=np.float32),
    )
    g.category = mj.mjtCatBit.mjCAT_DECOR

    # optional XY threshold ring (thin cylinder) to visualize "arrived" region
    if xy_thresh is not None and xy_thresh > 0:
        g = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        # cylinder size: [radius, half-height]
        mj.mjv_initGeom(
            g, mj.mjtGeom.mjGEOM_CYLINDER,
            size=np.array([float(xy_thresh), 0.003, 0], dtype=np.float32),
            pos=np.array([x, y, z], dtype=np.float32),
            mat=np.eye(3, dtype=np.float32).flatten(),
            rgba=np.array([1.0, 0.2, 0.2, 0.25], dtype=np.float32),
        )
        g.category = mj.mjtCatBit.mjCAT_DECOR

    # optional yaw threshold wedge arms (two thin capsules) to visualize +/- yaw_thresh
    if yaw_thresh is not None and yaw_thresh > 0:
        for sgn in (-1.0, +1.0):
            ang = yaw + sgn * float(yaw_thresh)
            xdir2 = np.array([np.cos(ang), np.sin(ang), 0.0])
            R2 = _rotmat_from_xdir(xdir2).astype(np.float32)
            g = scene.geoms[scene.ngeom]
            scene.ngeom += 1
            mj.mjv_initGeom(
                g, mj.mjtGeom.mjGEOM_CAPSULE,
                size=np.array([0.004, arrow_len, 0], dtype=np.float32),
                pos=np.array([x, y, z], dtype=np.float32),
                mat=R2.flatten(),
                rgba=np.array([1.0, 0.35, 0.35, 0.6], dtype=np.float32),
            )
            g.category = mj.mjtCatBit.mjCAT_DECOR


def _render_frame(model, data, opt, cam, scene, context, viewport, rgb_buf, goal=None, *, xy_thresh=None, yaw_thresh=None) -> np.ndarray:
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    _add_goal_geoms(scene, goal, xy_thresh=xy_thresh, yaw_thresh=yaw_thresh)
    mj.mjr_render(viewport, scene, context)
    mj.mjr_readPixels(rgb_buf, None, viewport, context)
    return rgb_buf


def _apply_fixed_camera(cam: mj.MjvCamera, lookat_xyz, *, distance=4.0, azimuth=90.0, elevation=-45.0):
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(lookat_xyz, dtype=float)
    cam.distance = float(distance)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=str)
    ap.add_argument("--mode", choices=["states", "actions"], default="states")
    ap.add_argument("--xml", type=str, default="models/pololu.xml")
    ap.add_argument("--body", type=str, default="chassis")
    ap.add_argument("--camera_mode", choices=["tracking", "fixed"], default="tracking")
    ap.add_argument("--cam_distance", type=float, default=None)
    ap.add_argument("--cam_azimuth", type=float, default=None)
    ap.add_argument("--cam_elevation", type=float, default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=None, help="Override FPS (otherwise uses 1/log_dt if present).")
    ap.add_argument("--dt", type=float, default=None,
                    help="(actions mode) override model.opt.timestep for rollout playback. "
                         "If not set, uses XML timestep. If NPZ has sim_dt, we use that by default.")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    assert npz_path.exists(), f"Missing {npz_path}"
    xml_path = Path(args.xml)
    assert xml_path.exists(), f"Missing {xml_path}"

    d = np.load(npz_path, allow_pickle=True)
    goal = np.asarray(d["goal"]) if "goal" in d else None
    xy_thresh = float(d["goal_xy_threshold"]) if "goal_xy_threshold" in d else None
    yaw_thresh = float(d["goal_yaw_threshold"]) if "goal_yaw_threshold" in d else None

    # FPS: prefer NPZ log_dt
    if args.fps is not None:
        fps = int(args.fps)
    elif "log_dt" in d:
        fps = int(round(1.0 / float(d["log_dt"])))
    else:
        fps = 100
    fps = max(1, fps)

    out_dir = npz_path.parent / f"{npz_path.stem}_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"track_{args.mode}.mp4"

    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)

    # In actions mode, pick dt:
    # - if --dt given: use it
    # - else if NPZ has sim_dt: use that
    # - else keep XML dt
    if args.mode == "actions":
        if args.dt is not None:
            model.opt.timestep = float(args.dt)
        elif "sim_dt" in d:
            model.opt.timestep = float(d["sim_dt"])

    win = _init_hidden_glfw_context()
    cam, opt, scene, context, viewport, rgb_buf = _prepare_renderer(model, args.width, args.height)

    # Camera setup
    if args.camera_mode == "tracking":
        _apply_tracking_camera(
            cam,
            model,
            args.body,
            distance=1.4 if args.cam_distance is None else args.cam_distance,
            azimuth=90.0 if args.cam_azimuth is None else args.cam_azimuth,
            elevation=-25.0 if args.cam_elevation is None else args.cam_elevation,
        )
    else:
        # Fixed "full view": center camera between start and goal if available.
        if "x_traj" in d:
            x_traj = np.asarray(d["x_traj"], dtype=np.float32)
            xs = x_traj[:, 0]
            ys = x_traj[:, 1]
            cx = float(0.5 * (xs.min() + xs.max()))
            cy = float(0.5 * (ys.min() + ys.max()))
            span = float(max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0))
        else:
            cx, cy, span = 0.0, 0.0, 2.0
        if goal is not None:
            cx = 0.5 * (cx + float(goal[0]))
            cy = 0.5 * (cy + float(goal[1]))
        dist = (2.0 + 1.5 * span) if args.cam_distance is None else args.cam_distance
        _apply_fixed_camera(
            cam,
            [cx, cy, 0.0],
            distance=dist,
            azimuth=90.0 if args.cam_azimuth is None else args.cam_azimuth,
            elevation=-60.0 if args.cam_elevation is None else args.cam_elevation,
        )

    proc = _open_ffmpeg(str(out_mp4), args.width, args.height, fps)

    try:
        if args.mode == "states":
            x_traj = np.asarray(d["x_traj"])
            for x in x_traj:
                _set_state_from_x(model, data, x)
                mj.mj_forward(model, data)
                frame = _render_frame(model, data, opt, cam, scene, context, viewport, rgb_buf, goal=goal, xy_thresh=xy_thresh, yaw_thresh=yaw_thresh)
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())

        else:
            # actions mode requires qpos0/qvel0 and u_traj (which is ctrl)
            assert "u_traj" in d, "actions mode needs u_traj in NPZ"
            assert "qpos0" in d and "qvel0" in d, "actions mode needs qpos0 and qvel0 in NPZ"

            u_traj = np.asarray(d["u_traj"], dtype=np.float32)   # <-- this is ctrl
            qpos0  = np.asarray(d["qpos0"], dtype=np.float32)
            qvel0  = np.asarray(d["qvel0"], dtype=np.float32)

            action_repeat = int(d["action_repeat"]) if "action_repeat" in d else 1

            _set_state_from_qpos_qvel(data, qpos0, qvel0)
            mj.mj_forward(model, data)

            # render initial
            frame = _render_frame(model, data, opt, cam, scene, context, viewport, rgb_buf, goal=goal, xy_thresh=xy_thresh, yaw_thresh=yaw_thresh)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())

            # roll out controls with mj_step
            for ctrl in u_traj[1:]:
                data.ctrl[:] = ctrl
                for _ in range(action_repeat):
                    mj.mj_step(model, data)

                frame = _render_frame(model, data, opt, cam, scene, context, viewport, rgb_buf, goal=goal, xy_thresh=xy_thresh, yaw_thresh=yaw_thresh)
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())


    finally:
        try:
            proc.stdin.close()
            proc.wait()
        except Exception:
            pass
        _free_renderer(scene, context)
        _close_hidden_glfw_context(win)

    print(f"Rendered {out_mp4} @ {fps} FPS (mode={args.mode})")


if __name__ == "__main__":
    main()
