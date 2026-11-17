"""
fastloop.py
-----------
The per-step part of `sim.Episode.run`, compiled with Numba. The loop calls
`mj_step` of the libmujoco that the `mujoco` wheel ships (through its address,
so the compiled code stays cacheable) and does the controller targets, the
phase update, the energy, the heading, the fall check and the per-tick
behaviour sampling without going back to Python between steps.

It does the same operations in the same order as the Python loop of
`Episode.run` (`DARWINBOT_PY_LOOP=1` selects that loop, which is the reference
of `tests/test_fastloop.py`). Anything that cannot be set up (library or symbol
not found, a MuJoCo whose `mjData` layout the probe does not recognise) makes
`prepare` return None, with one warning, and the Python loop runs instead.

The compiled loop reads a few `mjData` fields the Python bindings only expose
as copies (`time`, `ncon`, the warning counters, the `contact` pointer). Their
byte offsets are found once per process by writing sentinel values into a small
model's data and scanning for them, so no offset is hard coded; the contact
pointer and `ncon` are checked again against the bindings at the start of every
episode.
"""

import ctypes
import glob
import math
import os
import warnings

import mujoco
import numpy as np
from numba import carray, njit, types
from numba.extending import intrinsic

BLOWUP_WARNINGS = (mujoco.mjtWarning.mjWARN_BADQACC,
                   mujoco.mjtWarning.mjWARN_BADQVEL,
                   mujoco.mjtWarning.mjWARN_BADQPOS)

_TINY = """<mujoco><option timestep="0.004"/><worldbody>
<geom type="plane" size="2 2 .1"/>
<body pos="0 0 .06"><freejoint/><geom type="sphere" size=".05"/></body>
<body pos=".4 0 .5"><freejoint/><geom type="sphere" size=".05"/></body>
<body pos="-.4 0 .9"><freejoint/><geom type="sphere" size=".05"/></body>
<body pos="0 .4 1.3"><freejoint/><geom type="sphere" size=".05"/></body>
</worldbody></mujoco>"""


# ---- compiled pieces ----------------------------------------------------

@intrinsic
def _mj_step(typingctx, fn, model, data):
    """Call the C function at address `fn` as mj_step(model, data)."""
    from llvmlite import ir
    sig = types.void(types.intp, types.intp, types.intp)

    def codegen(context, builder, sig, args):
        i8p = ir.IntType(8).as_pointer()
        fty = ir.FunctionType(ir.VoidType(), [i8p, i8p])
        f = builder.inttoptr(args[0], fty.as_pointer())
        builder.call(f, [builder.inttoptr(args[1], i8p),
                         builder.inttoptr(args[2], i8p)])
        return context.get_dummy_value()
    return sig, codegen


@intrinsic
def _int_ptr(typingctx, addr):
    """An int32 pointer from an integer address."""
    from llvmlite import ir
    sig = types.CPointer(types.int32)(types.intp)

    def codegen(context, builder, sig, args):
        return builder.inttoptr(args[0], ir.IntType(32).as_pointer())
    return sig, codegen


@njit(cache=True)
def _bad_state(qacc, qvel):
    for a in qacc:
        if not math.isfinite(a):
            return True
    for v in qvel:
        if abs(v) > 1e3:
            return True
    return False


@njit(cache=True)
def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@njit(cache=True)
def _psum(a, n):
    """Sum of a[:n] in the order of NumPy's pairwise float sum (n < 128)."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[i]
        return res
    r0, r1, r2, r3 = a[0], a[1], a[2], a[3]
    r4, r5, r6, r7 = a[4], a[5], a[6], a[7]
    i = 8
    while i < n - (n % 8):
        r0 += a[i]
        r1 += a[i + 1]
        r2 += a[i + 2]
        r3 += a[i + 3]
        r4 += a[i + 4]
        r5 += a[i + 5]
        r6 += a[i + 6]
        r7 += a[i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[i]
        i += 1
    return res


@njit(cache=True)
def _advance(theta, omega, sigma, force, dt, torque):
    # advance_phases of controller.py, repeated here so the cached loop does
    # not depend on another module's compiled function
    for i in range(theta.shape[0]):
        n = force[i] / torque
        if n > 1.0:
            n = 1.0
        elif n < -1.0:
            n = -1.0
        theta[i] += dt * (omega - sigma[i] * n * math.cos(theta[i]))


@njit(cache=True)
def run_loop(step_fn, model_addr, data_addr, n_steps, hold, dt, torque,
             ramp_time, upright_min,
             theta, omega, sigma, offset, amp1, ph1, amp2, ph2,
             force, velocity, ctrl, qacc, qvel, xmat, xpos, geom_xpos,
             time_v, ncon_v, warn_v, widx, cptr_v, stride, goff,
             robot, tgeoms, sgeoms, feet, lat_buf, mid,
             hit, touch, prev_pos, prev_touch, scratch, out):
    """Steps the episode. `out` receives the scalars:
    0 energy, 1 yaw, 2 fell, 3 unstable, 4 x_mid (nan if none), 5 zsum,
    6 upsum, 7 ticks, 8 tiltsum, 9 slipsum, 10 sgsum, 11 steps taken."""
    nu = theta.shape[0]
    nt = tgeoms.shape[0]
    L = lat_buf.shape[0]
    head = 0
    tick_dt = hold * dt
    energy = 0.0
    yaw = 0.0
    prev = math.atan2(xmat[3], xmat[0])
    fell = False
    unstable = False
    x_mid = np.nan
    t_prev = 0.0
    zsum = 0.0
    upsum = 0.0
    tiltsum = 0.0
    slipsum = 0.0
    sgsum = 0.0
    ticks = 0
    done = 0
    for n in range(1, n_steps + 1):
        if (n - 1) % hold == 0:
            t = time_v[0]
            r = min(1.0, t / ramp_time)
            for i in range(nu):
                v = r * (offset[i] + amp1[i] * math.sin(theta[i] + ph1[i])
                         + amp2[i] * math.sin(2.0 * theta[i] + ph2[i]))
                if L > 0:
                    ctrl[i] = lat_buf[head, i]
                    lat_buf[head, i] = v
                else:
                    ctrl[i] = v
            if L > 0:
                head = (head + 1) % L
        _mj_step(step_fn, model_addr, data_addr)
        done = n
        _advance(theta, omega, sigma, force, dt, torque)
        for i in range(nu):
            scratch[i] = abs(force[i] * velocity[i])
        energy += _psum(scratch, nu) * dt
        now = math.atan2(xmat[3], xmat[0])
        yaw += _wrap(now - prev)
        prev = now
        if (warn_v[widx[0]] != 0 or warn_v[widx[1]] != 0 or warn_v[widx[2]] != 0
                or time_v[0] < t_prev or _bad_state(qacc, qvel)):
            unstable = True
            break
        t_prev = time_v[0]
        if xmat[8] < upright_min:
            fell = True
            break
        if n == mid:
            x_mid = xpos[0]
        if n % hold == 0:
            hit[:] = False
            ncon = ncon_v[0]
            if ncon > 0:
                g = carray(_int_ptr(cptr_v[0]), ncon * stride)
                for k in range(ncon):
                    a = g[k * stride + goff]
                    b = g[k * stride + goff + 1]
                    if not robot[b]:
                        hit[a] = True
                    if not robot[a]:
                        hit[b] = True
            nsl = 0.0
            m = 0
            for j in range(nt):
                tj = hit[tgeoms[j]]
                if tj:
                    touch[j] += 1.0
                if tj and prev_touch[j] and feet[j]:
                    dx = geom_xpos[tgeoms[j], 0] - prev_pos[j, 0]
                    dy = geom_xpos[tgeoms[j], 1] - prev_pos[j, 1]
                    scratch[m] = math.sqrt(dx * dx + dy * dy)
                    m += 1
            if m > 0:
                nsl = _psum(scratch, m)
                slipsum += nsl / tick_dt
            for j in range(nt):
                prev_touch[j] = hit[tgeoms[j]]
                prev_pos[j, 0] = geom_xpos[tgeoms[j], 0]
                prev_pos[j, 1] = geom_xpos[tgeoms[j], 1]
            for j in range(sgeoms.shape[0]):
                if hit[sgeoms[j]]:
                    sgsum += 1.0
                    break
            zsum += xpos[2]
            upsum += xmat[8]
            tiltsum += math.acos(min(1.0, max(-1.0, xmat[8])))
            ticks += 1
    out[0] = energy
    out[1] = yaw
    out[2] = 1.0 if fell else 0.0
    out[3] = 1.0 if unstable else 0.0
    out[4] = x_mid
    out[5] = zsum
    out[6] = upsum
    out[7] = ticks
    out[8] = tiltsum
    out[9] = slipsum
    out[10] = sgsum
    out[11] = done


# ---- set-up -------------------------------------------------------------

_state = {"done": False, "info": None}


def _library_paths():
    """Candidate files of the libmujoco the bindings use: the one mapped into
    this process first (Linux), then the ones in the wheel's directory."""
    paths = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                p = line.split()[-1] if line.split() else ""
                if "libmujoco" in os.path.basename(p) and p not in paths:
                    paths.append(p)
    except OSError:
        pass
    here = os.path.dirname(mujoco.__file__)
    for pat in ("libmujoco*.so*", "libmujoco*.dylib", "mujoco.dll"):
        for p in sorted(glob.glob(os.path.join(here, pat))):
            if p not in paths:
                paths.append(p)
    return paths


def _step_address():
    for p in _library_paths():
        try:
            lib = ctypes.CDLL(p)
            fn = lib.mj_step
        except (OSError, AttributeError):
            continue
        return ctypes.cast(fn, ctypes.c_void_p).value, lib
    raise RuntimeError("libmujoco with mj_step not found")


def _raw(addr, nbytes):
    return ctypes.string_at(addr, nbytes)


def _discover():
    """Byte offsets in mjData of time, ncon, the warning counters and the
    contact pointer, plus the geom offset and stride inside mjContact."""
    m = mujoco.MjModel.from_xml_string(_TINY)
    d = mujoco.MjData(m)
    t_sent = 1234.5678901
    w_sent = [1000003 + 7919 * i for i in range(len(BLOWUP_WARNINGS))]
    d.time = t_sent
    for w, s in zip(BLOWUP_WARNINGS, w_sent):
        d.warning[w].number = s
    mujoco.mj_forward(m, d)
    base = d._address
    cp = d.contact.dist.ctypes.data
    found_w = [None] * len(w_sent)
    off_t = off_cp = None
    chunk = 1024            # bytes; read upward until the contact pointer shows
    pos = 0
    buf = b""
    while off_cp is None and pos < (1 << 21):
        buf += _raw(base + pos, chunk)
        pos += chunk
        i32 = np.frombuffer(buf, dtype=np.int32)
        f64 = np.frombuffer(buf[:len(buf) // 8 * 8], dtype=np.float64)
        i64 = np.frombuffer(buf[:len(buf) // 8 * 8], dtype=np.int64)
        for i, s in enumerate(w_sent):
            hit = np.nonzero(i32 == s)[0]
            if len(hit):
                found_w[i] = int(hit[0]) * 4
        hit = np.nonzero(f64 == t_sent)[0]
        if len(hit):
            off_t = int(hit[0]) * 8
        hit = np.nonzero(i64 == cp)[0]
        if len(hit):
            off_cp = int(hit[0]) * 8
    if off_t is None or off_cp is None or None in found_w:
        raise RuntimeError("mjData layout not recognised")
    # ncon: an int32 between the warnings and the contact pointer that follows
    # the contact count of the data as it falls and lands
    d.time = 0.0
    for w in BLOWUP_WARNINGS:
        d.warning[w].number = 0
    lo = (max(found_w) + 4)
    cands = None
    seen = set()
    mujoco.mj_resetData(m, d)
    for _ in range(600):
        mujoco.mj_step(m, d)
        raw = _raw(base + lo, off_cp - lo)
        arr = np.frombuffer(raw, dtype=np.int32)
        match = set(np.nonzero(arr == d.ncon)[0].tolist())
        cands = match if cands is None else cands & match
        seen.add(int(d.ncon))
        if int(d.ncon) > 0:     # the pointer must follow the bindings' view
            if np.frombuffer(_raw(base + off_cp, 8), dtype=np.int64)[0] != d.contact.dist.ctypes.data:
                raise RuntimeError("contact pointer offset is not stable")
    if len(seen) < 3 or not cands:
        raise RuntimeError("ncon offset not found")
    off_ncon = lo + 4 * min(cands)
    geom = d.contact.geom
    g0 = geom.ctypes.data - d.contact.dist.ctypes.data
    if g0 % 4 or geom.strides[0] % 4:
        raise RuntimeError("unexpected mjContact layout")
    return dict(time=off_t, ncon=off_ncon, warn=found_w, cptr=off_cp,
                goff=g0 // 4, stride=geom.strides[0] // 4)


def _info():
    if not _state["done"]:
        _state["done"] = True
        try:
            fn, lib = _step_address()
            off = _discover()
            _state["info"] = dict(off, fn=fn, lib=lib)
        except Exception as e:      # fail soft: the Python loop does the work
            warnings.warn(f"darwinbot: compiled episode loop unavailable ({e}); "
                          "using the Python loop", RuntimeWarning, stacklevel=3)
    return _state["info"]


def _view(addr, dtype, n):
    nbytes = np.dtype(dtype).itemsize * int(n)
    buf = (ctypes.c_char * nbytes).from_address(addr)
    return np.frombuffer(buf, dtype=dtype)


class Prepared:
    """Per-episode views and constants of the compiled loop."""

    def __init__(self, ep, info):
        d = ep.data
        a = d._address
        self.info = info
        self.time = _view(a + info["time"], np.float64, 1)
        self.ncon = _view(a + info["ncon"], np.int32, 1)
        self.cptr = _view(a + info["cptr"], np.int64, 1)
        offs = info["warn"]
        wmin = min(offs)
        self.warn = _view(a + wmin, np.int32, (max(offs) - wmin) // 4 + 1)
        self.widx = np.array([(o - wmin) // 4 for o in offs], dtype=np.int64)
        self.torso = ep.torso
        self.robot = ep._robot.copy()
        self.tgeoms = np.asarray(ep._tgeoms, dtype=np.int64)
        self.sgeoms = np.asarray(ep._sgeoms, dtype=np.int64)
        self.feet = ~np.asarray(ep._twheel, dtype=bool)
        nt = len(self.tgeoms)
        self.hit = np.zeros(ep.model.ngeom, dtype=np.bool_)
        self.touch = np.zeros(nt)
        self.prev_pos = np.zeros((nt, 2))
        self.prev_touch = np.zeros(nt, dtype=np.bool_)
        self.scratch = np.zeros(max(ep.model.nu, nt, 1))
        self.out = np.zeros(12)
        self.xmat = d.xmat[ep.torso]
        self.xpos = d.xpos[ep.torso]

    def consistent(self, d) -> bool:
        """The raw reads agree with the bindings right now (after a reset)."""
        if int(self.ncon[0]) != d.ncon or self.time[0] != d.time:
            return False
        if d.ncon > 0 and int(self.cptr[0]) != d.contact.dist.ctypes.data:
            return False
        return True


def prepare(ep):
    """The compiled-loop state of an episode, or None when unavailable."""
    if os.environ.get("DARWINBOT_PY_LOOP"):
        return None
    info = _info()
    if info is None:
        return None
    prep = getattr(ep, "_fast", None)
    if prep is None or prep.info is not info:
        try:
            prep = Prepared(ep, info)
        except Exception as e:
            warnings.warn(f"darwinbot: compiled episode loop unavailable ({e}); "
                          "using the Python loop", RuntimeWarning, stacklevel=3)
            _state["info"] = None
            return None
        ep._fast = prep
    return prep


def run(ep, prep, controller, sim_model, variant_latency, torque, ramp_time, upright_min):
    """Run the loop on an episode just reset. Returns the 12 scalars of
    `run_loop`'s `out` plus the touch counts."""
    d = ep.data
    p = controller.params
    mid = ep.n_steps // 2
    prep.touch[:] = 0.0
    prep.prev_pos[:] = 0.0
    prep.prev_touch[:] = False
    lat = np.zeros((variant_latency, d.ctrl.shape[0]))
    run_loop(prep.info["fn"], sim_model._address, d._address, ep.n_steps, ep.hold,
             ep.model.opt.timestep, torque, ramp_time, upright_min,
             controller.theta, controller.omega, p.sigma, p.offset, p.amp1,
             p.phase1, p.amp2, p.phase2,
             d.actuator_force, d.actuator_velocity, d.ctrl, d.qacc, d.qvel,
             prep.xmat, prep.xpos, d.geom_xpos,
             prep.time, prep.ncon, prep.warn, prep.widx, prep.cptr,
             prep.info["stride"], prep.info["goff"],
             prep.robot, prep.tgeoms, prep.sgeoms, prep.feet, lat, mid,
             prep.hit, prep.touch, prep.prev_pos, prep.prev_touch,
             prep.scratch, prep.out)
    return prep.out, prep.touch


def warmup() -> bool:
    """Compile (or load from the cache) the loop in this process, so a pool
    forked afterwards starts with it ready. True when the compiled loop is in
    use. A 0.1 s episode of a small body; costs nothing when already compiled."""
    if os.environ.get("DARWINBOT_PY_LOOP"):
        return False
    from darwinbot import controller, morphology
    from darwinbot.sim import Episode
    from darwinbot.terrain import flat
    ep = Episode(morphology.decode(morphology.encode(morphology.template(2))),
                 flat(), 0.1)
    ep.run(controller.random_controller(np.random.default_rng(0)))
    return getattr(ep, "_fast", None) is not None
