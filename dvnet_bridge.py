"""
dvnet_bridge.py  —  ParkWatch DVNetSDK bridge for AMAAN ANPR
=============================================================
MUST run under 32-bit Python (same bitness as DVNetSDK.dll).

Quick check — run this to see your Python bitness:
    python -c "import struct; print(struct.calcsize('P')*8)"

If it prints 64, install 32-bit Python 3.10 from python.org:
    https://www.python.org/downloads/release/python-31011/
    Download "Windows installer (32-bit)"
    Install to e.g. C:\\Python310-32\\

Then run this bridge with the 32-bit Python:
    C:\\Python310-32\\python.exe dvnet_bridge.py

Install flask in that 32-bit environment first:
    C:\\Python310-32\\python.exe -m pip install flask

DVNetSDK.dll must be in the SAME folder as this script.

LED Display Layout per camera:
    Line 0: "WELCOME" (entry) or "GOODBYE" (exit)  — permanent
    Line 1: plate number e.g. "KAA123B"            — shown 5 seconds, then cleared

app.py calls:  POST http://localhost:8889/led/display
               {"cam_id": 0, "plate": "KAA123B", "device_type": "Entry1"}
"""

import ctypes
import os
import struct
import threading
import time
import traceback

# ── Flask import (only dependency needed in 32-bit env) ──────────────────────
try:
    from flask import Flask, request, jsonify
except ImportError:
    raise ImportError(
        "\n\nflask not installed in this Python environment.\n"
        "Run:  python -m pip install flask\n"
    )

# =============================================================================
# BITNESS GUARD — fail immediately with a clear message if 64-bit
# =============================================================================
_BITS = struct.calcsize("P") * 8
if _BITS != 32:
    raise RuntimeError(
        f"\n\n{'='*60}\n"
        f"  ERROR: dvnet_bridge.py requires 32-bit Python!\n"
        f"  You are running {_BITS}-bit Python.\n"
        f"\n"
        f"  DVNetSDK.dll is a 32-bit DLL and CANNOT be loaded by\n"
        f"  a 64-bit process — Windows will refuse it immediately.\n"
        f"\n"
        f"  Fix:\n"
        f"  1. Download Python 3.10 (32-bit) from python.org\n"
        f"  2. Install to C:\\Python310-32\\\n"
        f"  3. Run:  C:\\Python310-32\\python.exe -m pip install flask\n"
        f"  4. Run:  C:\\Python310-32\\python.exe dvnet_bridge.py\n"
        f"{'='*60}\n"
    )

# =============================================================================
# ── CONFIGURATION — edit these to match your site ────────────────────────────
# =============================================================================

BRIDGE_PORT = 8889          # port this bridge listens on
BRIDGE_HOST = "0.0.0.0"    # 0.0.0.0 so app.py on same machine can reach it

# Camera / LED board definitions
# dev_type: 1=ZhiShi, 2=HuaXia, 3=QiYun, 6=DaHua, 7=TongWei
CAMERAS = [
    {"id": 0, "name": "Entry1", "ip": "10.10.10.151", "user": "admin", "password": "admin", "dev_type": 1},
    {"id": 1, "name": "Entry2", "ip": "10.10.10.152", "user": "admin", "password": "admin", "dev_type": 1},
    {"id": 2, "name": "Exit1",  "ip": "10.10.10.153", "user": "admin", "password": "admin", "dev_type": 1},
    {"id": 3, "name": "Exit2",  "ip": "10.10.10.154", "user": "admin", "password": "admin", "dev_type": 1},
]

# What Line 0 shows permanently per camera name
# Keys must match the "name" field above (case-insensitive match done at runtime)
LINE0_TEXT = {
    "entry": "WELCOME",    # shown on Entry1, Entry2
    "exit":  "THANK YOU",  # shown on Exit1, Exit2
}

PLATE_DISPLAY_SECONDS = 5   # how long the plate stays on Line 1 before clearing
LED_ADDR = 0x00             # 0x00 = broadcast; change if your board has a fixed address

# Reconnect interval in seconds if a camera drops
RECONNECT_INTERVAL = 15

# =============================================================================
# DVNetSDK constants
# =============================================================================
DVERR_OK            = 0
DV_IP_LEN           = 128
DV_IP_LEN_SHORT     = 64
DV_NAME_LEN         = 128
DV_NAME_LEN_SHORT   = 64
DV_RESERVED_LEN     = 64

DVCMD_OPEN_RELAY       = 100
DVCMD_CLOSE_RELAY      = 101
DVCMD_OPEN_CLOSE_RELAY = 102

# =============================================================================
# ctypes structures  (must exactly mirror the C structs)
# =============================================================================

class DV_DeviceCnnInfo(ctypes.Structure):
    _fields_ = [
        ("szIP",            ctypes.c_char * DV_IP_LEN),
        ("nPort",           ctypes.c_int),
        ("szUserName",      ctypes.c_char * DV_NAME_LEN),
        ("szPassword",      ctypes.c_char * DV_NAME_LEN),
        ("nLoginTimeout",   ctypes.c_int),
        ("nDeviceType",     ctypes.c_int),
        ("szCloudGateIP",   ctypes.c_char * DV_IP_LEN_SHORT),
        ("nCloudGatePort",  ctypes.c_int),
        ("szChannelId",     ctypes.c_char * DV_NAME_LEN_SHORT),
        ("szChannelName",   ctypes.c_char * DV_NAME_LEN_SHORT),
        ("Reserved",        ctypes.c_byte  * 60),
    ]


class DV_Cmd(ctypes.Structure):
    _fields_ = [
        ("nCmd",     ctypes.c_int),
        ("nParam",   ctypes.c_int),
        ("nParam2",  ctypes.c_int),
        ("pszData",  ctypes.c_void_p),
        ("Reserved", ctypes.c_int * (DV_RESERVED_LEN - 1)),
    ]


# =============================================================================
# DVNetSDK wrapper
# =============================================================================

class DVNetSDK:
    """Thin ctypes wrapper. Call init() once at startup, release() at shutdown."""

    def __init__(self):
        dll_dir  = os.path.dirname(os.path.abspath(__file__))
        dll_path = os.path.join(dll_dir, "DVNetSDK.dll")
        if not os.path.exists(dll_path):
            raise FileNotFoundError(
                f"DVNetSDK.dll not found at:\n  {dll_path}\n"
                f"Put DVNetSDK.dll in the same folder as dvnet_bridge.py"
            )
        self._dll  = ctypes.WinDLL(dll_path)
        self._lock = threading.Lock()
        self._setup_prototypes()

    def _setup_prototypes(self):
        d = self._dll

        # Lifecycle
        d.DV_InitSDK.restype    = ctypes.c_int;  d.DV_InitSDK.argtypes    = []
        d.DV_ReleaseSDK.restype = ctypes.c_int;  d.DV_ReleaseSDK.argtypes = []

        # Error message
        d.DV_GetErrorMessage.restype  = ctypes.c_int
        d.DV_GetErrorMessage.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

        # Device
        d.DV_OpenDevice.restype  = ctypes.c_int
        d.DV_OpenDevice.argtypes = [
            ctypes.POINTER(DV_DeviceCnnInfo),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        d.DV_CloseDevice.restype  = ctypes.c_int
        d.DV_CloseDevice.argtypes = [ctypes.c_void_p]

        # RS-485
        d.DV_OpenRS485.restype  = ctypes.c_int
        d.DV_OpenRS485.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)
        ]
        d.DV_CloseRS485.restype  = ctypes.c_int
        d.DV_CloseRS485.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        d.DV_WriteRS485.restype  = ctypes.c_int
        d.DV_WriteRS485.argtypes = [
            ctypes.c_void_p,            # hDev
            ctypes.c_void_p,            # hRS485
            ctypes.c_char_p,            # pData
            ctypes.c_int,               # nDataLength
            ctypes.POINTER(ctypes.c_int), # pDataWriteLength
        ]

        # Exec (relay)
        d.DV_Exec.restype  = ctypes.c_int
        d.DV_Exec.argtypes = [ctypes.c_void_p, ctypes.POINTER(DV_Cmd)]

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def init(self):
        return self._dll.DV_InitSDK()

    def release(self):
        return self._dll.DV_ReleaseSDK()

    def get_error(self, code):
        if code == DVERR_OK:
            return "OK"
        buf = ctypes.create_string_buffer(512)
        self._dll.DV_GetErrorMessage(code, buf, 512)
        msg = buf.value.decode("utf-8", errors="replace").strip()
        return msg if msg else f"Error {code}"

    # ── Device ────────────────────────────────────────────────────────────────
    def open_device(self, ip, user, password, dev_type, timeout_ms=8000):
        info = DV_DeviceCnnInfo()
        info.szIP           = ip.encode()
        info.nPort          = 0
        info.szUserName     = user.encode()
        info.szPassword     = password.encode()
        info.nLoginTimeout  = timeout_ms
        info.nDeviceType    = dev_type
        info.szCloudGateIP  = b""
        info.nCloudGatePort = 0
        info.szChannelId    = b""
        info.szChannelName  = b""
        hdev = ctypes.c_void_p(0)
        with self._lock:
            ret = self._dll.DV_OpenDevice(ctypes.byref(info), ctypes.byref(hdev))
        return ret, hdev.value

    def close_device(self, hdev):
        if hdev:
            with self._lock:
                self._dll.DV_CloseDevice(ctypes.c_void_p(hdev))

    # ── RS-485 ────────────────────────────────────────────────────────────────
    def open_rs485(self, hdev, com_id=0):
        hrs = ctypes.c_void_p(0)
        with self._lock:
            ret = self._dll.DV_OpenRS485(
                ctypes.c_void_p(hdev), com_id, ctypes.byref(hrs)
            )
        return ret, hrs.value

    def close_rs485(self, hdev, hrs):
        if hdev and hrs:
            with self._lock:
                self._dll.DV_CloseRS485(
                    ctypes.c_void_p(hdev), ctypes.c_void_p(hrs)
                )

    def write_rs485(self, hdev, hrs, data: bytes):
        written = ctypes.c_int(0)
        with self._lock:
            ret = self._dll.DV_WriteRS485(
                ctypes.c_void_p(hdev),
                ctypes.c_void_p(hrs),
                data,
                len(data),
                ctypes.byref(written),
            )
        return ret, written.value

    # ── Relay ─────────────────────────────────────────────────────────────────
    def exec_relay(self, hdev, cmd, relay_idx, hold_ms=0):
        c = DV_Cmd()
        c.nCmd    = cmd
        c.nParam  = relay_idx
        c.nParam2 = hold_ms
        c.pszData = None
        with self._lock:
            return self._dll.DV_Exec(ctypes.c_void_p(hdev), ctypes.byref(c))


# =============================================================================
# LED Protocol  — exact Python port of the C# BuildPacket / CRC16
# =============================================================================

_CRC_HI = bytes([
    0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,
    0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,
    0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,
    0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,
    0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,
    0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,
    0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,
    0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,
    0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,
    0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,
    0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,
    0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,
    0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,
    0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,
    0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,
    0x00,0xC1,0x81,0x40,0x01,0xC0,0x80,0x41,0x01,0xC0,0x80,0x41,0x00,0xC1,0x81,0x40,
])

_CRC_LO = bytes([
    0x00,0xC0,0xC1,0x01,0xC3,0x03,0x02,0xC2,0xC6,0x06,0x07,0xC7,0x05,0xC5,0xC4,0x04,
    0xCC,0x0C,0x0D,0xCD,0x0F,0xCF,0xCE,0x0E,0x0A,0xCA,0xCB,0x0B,0xC9,0x09,0x08,0xC8,
    0xD8,0x18,0x19,0xD9,0x1B,0xDB,0xDA,0x1A,0x1E,0xDE,0xDF,0x1F,0xDD,0x1D,0x1C,0xDC,
    0x14,0xD4,0xD5,0x15,0xD7,0x17,0x16,0xD6,0xD2,0x12,0x13,0xD3,0x11,0xD1,0xD0,0x10,
    0xF0,0x30,0x31,0xF1,0x33,0xF3,0xF2,0x32,0x36,0xF6,0xF7,0x37,0xF5,0x35,0x34,0xF4,
    0x3C,0xFC,0xFD,0x3D,0xFF,0x3F,0x3E,0xFE,0xFA,0x3A,0x3B,0xFB,0x39,0xF9,0xF8,0x38,
    0x28,0xE8,0xE9,0x29,0xEB,0x2B,0x2A,0xEA,0xEE,0x2E,0x2F,0xEF,0x2D,0xED,0xEC,0x2C,
    0xE4,0x24,0x25,0xE5,0x27,0xE7,0xE6,0x26,0x22,0xE2,0xE3,0x23,0xE1,0x21,0x20,0xE0,
    0xA0,0x60,0x61,0xA1,0x63,0xA3,0xA2,0x62,0x66,0xA6,0xA7,0x67,0xA5,0x65,0x64,0xA4,
    0x6C,0xAC,0xAD,0x6D,0xAF,0x6F,0x6E,0xAE,0xAA,0x6A,0x6B,0xAB,0x69,0xA9,0xA8,0x68,
    0x78,0xB8,0xB9,0x79,0xBB,0x7B,0x7A,0xBA,0xBE,0x7E,0x7F,0xBF,0x7D,0xBD,0xBC,0x7C,
    0xB4,0x74,0x75,0xB5,0x77,0xB7,0xB6,0x76,0x72,0xB2,0xB3,0x73,0xB1,0x71,0x70,0xB0,
    0x50,0x90,0x91,0x51,0x93,0x53,0x52,0x92,0x96,0x56,0x57,0x97,0x55,0x95,0x94,0x54,
    0x9C,0x5C,0x5D,0x9D,0x5F,0x9F,0x9E,0x5E,0x5A,0x9A,0x9B,0x5B,0x99,0x59,0x58,0x98,
    0x88,0x48,0x49,0x89,0x4B,0x8B,0x8A,0x4A,0x4E,0x8E,0x8F,0x4F,0x8D,0x4D,0x4C,0x8C,
    0x44,0x84,0x85,0x45,0x87,0x47,0x46,0x86,0x82,0x42,0x43,0x83,0x41,0x81,0x80,0x40,
])


def _crc16(data: bytes) -> int:
    """Exact port of MB_CRC16 from the LED protocol document."""
    hi = 0xFF
    lo = 0xFF
    for b in data:
        idx = lo ^ b
        lo  = hi ^ _CRC_HI[idx]
        hi  = _CRC_LO[idx]
    return (hi << 8) | lo


def _build_packet(cmd: int, msg: bytes, led_addr: int = 0x00) -> bytes:
    """
    Build a complete LED protocol packet.
    Layout: DA(1) VR(1=0x64) PN(2=0xFFFF) CMD(1) ML(1) MSG(n) CRC16(2)
    CRC covers DA..MSG inclusive, little-endian.
    """
    header = bytes([
        led_addr,       # DA — device address
        0x64,           # VR — protocol version (fixed)
        0xFF, 0xFF,     # PN — packet number (fixed, unused)
        cmd,            # CMD
        len(msg),       # ML — message length
    ])
    for_crc = header + msg
    crc     = _crc16(for_crc)
    return for_crc + bytes([crc & 0xFF, crc >> 8])   # little-endian CRC


def _build_display_packet(
    line_id:    int,
    text:       str,
    effect:     int   = 0x01,   # 0x01 = Right→Left scroll
    speed:      int   = 8,
    duration:   int   = 5,      # seconds to stay on board
    repeats:    int   = 1,      # 1 = show once then hold; 0 = infinite
    text_color: tuple = (255, 255, 0),   # R G B  (yellow)
    bg_color:   tuple = (0,   0,   0),   # R G B  (black)
    led_addr:   int   = 0x00,
) -> bytes:
    """
    Build CMD 0x62 (Display Text) packet.
    MSG layout: TWID ETM ETS DM DT EXM EXS FINDEX DRS TC[4] BC[4] TL[2] TEXT[...]
    """
    text_bytes = text.encode("utf-8")
    msg = bytes([
        line_id,                            # TWID  — window / line index
        effect  & 0xFF,                     # ETM   — entry transition mode
        speed   & 0xFF,                     # ETS   — entry transition speed
        0x00,                               # DM    — reserved
        duration & 0xFF,                    # DT    — display stay time (seconds)
        0x00,                               # EXM   — exit transition mode (reserved)
        0x00,                               # EXS   — exit transition speed (reserved)
        0x03,                               # FINDEX — font index (fixed)
        repeats & 0xFF,                     # DRS   — repeat count (0=infinite)
        text_color[0], text_color[1], text_color[2], 0x00,  # TC: R G B A
        bg_color[0],   bg_color[1],   bg_color[2],   0x00,  # BC: R G B A
        len(text_bytes) & 0xFF,             # TL low byte
        len(text_bytes) >> 8,               # TL high byte
    ]) + text_bytes
    return _build_packet(0x62, msg, led_addr)


def _build_clear_packet(line_id: int, led_addr: int = 0x00) -> bytes:
    """
    Send an empty string on a line to effectively clear it.
    Uses CMD 0x62 with 0-length text and duration=0.
    """
    return _build_display_packet(
        line_id    = line_id,
        text       = " ",       # single space — boards don't like truly empty
        effect     = 0x00,      # immediate (no scroll)
        speed      = 1,
        duration   = 0,
        repeats    = 1,
        text_color = (0, 0, 0), # black text on black bg = invisible
        bg_color   = (0, 0, 0),
        led_addr   = led_addr,
    )


# =============================================================================
# Camera connection state
# =============================================================================

class CameraState:
    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.hdev     = None   # device handle (int or None)
        self.hrs      = None   # RS-485 handle (int or None)
        self.connected= False
        self.lock     = threading.Lock()
        # Timer for clearing Line 1 after PLATE_DISPLAY_SECONDS
        self._clear_timer: threading.Timer | None = None

    @property
    def cam_id(self)     -> int:  return self.cfg["id"]
    @property
    def name(self)       -> str:  return self.cfg["name"]
    @property
    def is_entry(self)   -> bool: return "entry" in self.cfg["name"].lower()
    @property
    def line0_text(self) -> str:
        return LINE0_TEXT["entry"] if self.is_entry else LINE0_TEXT["exit"]


# =============================================================================
# Bridge manager — handles all cameras, reconnects, LED sends
# =============================================================================

class BridgeManager:

    def __init__(self, sdk: DVNetSDK):
        self._sdk    = sdk
        self._states = {c["id"]: CameraState(c) for c in CAMERAS}

    # ── Connect one camera ────────────────────────────────────────────────────
    def _connect_one(self, state: CameraState):
        cfg = state.cfg
        _log(f"[{state.name}] Connecting to {cfg['ip']} ...")
        ret, hdev = self._sdk.open_device(
            cfg["ip"], cfg["user"], cfg["password"], cfg["dev_type"]
        )
        if ret != DVERR_OK or not hdev:
            _log(f"[{state.name}] DV_OpenDevice FAILED ret={ret}: {self._sdk.get_error(ret)}")
            return False

        _log(f"[{state.name}] Camera connected OK (handle={hdev})")

        # Open RS-485 for LED
        ret2, hrs = self._sdk.open_rs485(hdev, 0)
        if ret2 != DVERR_OK or not hrs:
            _log(f"[{state.name}] DV_OpenRS485 FAILED ret={ret2}: {self._sdk.get_error(ret2)}"
                 f" — LED will NOT work for this camera")
            # Still mark connected so relay etc. can work; LED sends will be skipped
            with state.lock:
                state.hdev      = hdev
                state.hrs       = None
                state.connected = True
            return True

        _log(f"[{state.name}] RS-485 opened OK (handle={hrs})")

        with state.lock:
            state.hdev      = hdev
            state.hrs       = hrs
            state.connected = True

        # Show the permanent Line 0 welcome/goodbye text immediately
        self._send_line0(state)
        return True

    def _disconnect_one(self, state: CameraState):
        with state.lock:
            if state.hrs:
                self._sdk.close_rs485(state.hdev, state.hrs)
                state.hrs = None
            if state.hdev:
                self._sdk.close_device(state.hdev)
                state.hdev = None
            state.connected = False

    # ── Background reconnect loop ─────────────────────────────────────────────
    def _reconnect_loop(self, state: CameraState):
        """Runs in its own daemon thread. Keeps trying to (re)connect."""
        while True:
            if not state.connected:
                try:
                    self._connect_one(state)
                except Exception:
                    _log(f"[{state.name}] Exception during connect:\n{traceback.format_exc()}")
            time.sleep(RECONNECT_INTERVAL)

    def start(self):
        self._sdk.init()
        _log("DVNetSDK initialised.")
        for state in self._states.values():
            t = threading.Thread(
                target=self._reconnect_loop,
                args=(state,),
                name=f"reconn-{state.name}",
                daemon=True,
            )
            t.start()
        _log(f"Reconnect threads started for {len(self._states)} cameras.")

    def stop(self):
        for state in self._states.values():
            self._disconnect_one(state)
        self._sdk.release()

    # ── Low-level RS-485 send ─────────────────────────────────────────────────
    def _send_rs485(self, state: CameraState, packet: bytes) -> bool:
        with state.lock:
            if not state.connected or not state.hrs:
                _log(f"[{state.name}] RS-485 not open — skipping send")
                return False
            hdev = state.hdev
            hrs  = state.hrs

        ret, written = self._sdk.write_rs485(hdev, hrs, packet)
        if ret == DVERR_OK:
            _log(f"[{state.name}] RS-485 wrote {written}/{len(packet)} bytes "
                 f"TX: {packet.hex(' ').upper()}")
            return True
        else:
            _log(f"[{state.name}] DV_WriteRS485 FAILED ret={ret}: {self._sdk.get_error(ret)}")
            # Mark as disconnected so the reconnect loop picks it up
            with state.lock:
                state.connected = False
            return False

    # ── Send Line 0 (WELCOME / GOODBYE) ──────────────────────────────────────
    def _send_line0(self, state: CameraState):
        """
        Line 0: permanent scrolling text.
        White text, black background, continuous left scroll, infinite repeat.
        """
        pkt = _build_display_packet(
            line_id    = 0,
            text       = state.line0_text,
            effect     = 0x01,            # Right→Left scroll
            speed      = 6,
            duration   = 0,               # 0 = stay until replaced
            repeats    = 0,               # 0 = infinite loop
            text_color = (255, 255, 255), # white
            bg_color   = (0,   0,   0),   # black
            led_addr   = LED_ADDR,
        )
        self._send_rs485(state, pkt)

    # ── Send plate on Line 1 then auto-clear ─────────────────────────────────
    def send_plate(self, cam_id: int, plate: str) -> dict:
        state = self._states.get(cam_id)
        if state is None:
            return {"ok": False, "error": f"Unknown cam_id {cam_id}"}

        if not state.connected:
            return {"ok": False, "error": f"Camera {state.name} not connected"}

        plate = plate.strip().upper()
        if not plate:
            return {"ok": False, "error": "Empty plate text"}

        # Cancel any pending clear timer from the previous plate
        with state.lock:
            if state._clear_timer:
                state._clear_timer.cancel()
                state._clear_timer = None

        # Send plate on Line 1 — yellow text, shown for PLATE_DISPLAY_SECONDS
        pkt = _build_display_packet(
            line_id    = 1,
            text       = plate,
            effect     = 0x00,            # 0x00 = immediate (no scroll — plate reads instantly)
            speed      = 1,
            duration   = PLATE_DISPLAY_SECONDS,
            repeats    = 1,               # show once
            text_color = (255, 255, 0),   # yellow
            bg_color   = (0,   0,   0),   # black
            led_addr   = LED_ADDR,
        )
        ok = self._send_rs485(state, pkt)
        if not ok:
            return {"ok": False, "error": "RS-485 write failed"}

        _log(f"[{state.name}] Plate displayed: '{plate}' for {PLATE_DISPLAY_SECONDS}s")

        # Schedule clearing Line 1 after PLATE_DISPLAY_SECONDS seconds
        def _clear_line1():
            _log(f"[{state.name}] Clearing Line 1 (plate timer expired)")
            clear_pkt = _build_clear_packet(line_id=1, led_addr=LED_ADDR)
            self._send_rs485(state, clear_pkt)
            with state.lock:
                state._clear_timer = None

        timer = threading.Timer(PLATE_DISPLAY_SECONDS, _clear_line1)
        with state.lock:
            state._clear_timer = timer
        timer.daemon = True
        timer.start()

        return {"ok": True, "plate": plate, "cam_id": cam_id, "camera": state.name}

    # ── Relay control ─────────────────────────────────────────────────────────
    def relay_cmd(self, cam_id: int, relay_idx: int, cmd: str, hold_ms: int = 500) -> dict:
        state = self._states.get(cam_id)
        if state is None:
            return {"ok": False, "error": f"Unknown cam_id {cam_id}"}
        if not state.connected or not state.hdev:
            return {"ok": False, "error": f"Camera {state.name} not connected"}

        cmd_map = {
            "open":  DVCMD_OPEN_RELAY,
            "close": DVCMD_CLOSE_RELAY,
            "pulse": DVCMD_OPEN_CLOSE_RELAY,
        }
        dvcmd = cmd_map.get(cmd.lower())
        if dvcmd is None:
            return {"ok": False, "error": f"Unknown relay cmd '{cmd}' — use open/close/pulse"}

        ret = self._sdk.exec_relay(state.hdev, dvcmd, relay_idx, hold_ms if dvcmd == DVCMD_OPEN_CLOSE_RELAY else 0)
        if ret == DVERR_OK:
            _log(f"[{state.name}] Relay {relay_idx} {cmd} OK")
            return {"ok": True}
        else:
            err = self._sdk.get_error(ret)
            _log(f"[{state.name}] Relay {relay_idx} {cmd} FAILED: {err}")
            return {"ok": False, "error": err}

    # ── Status ────────────────────────────────────────────────────────────────
    def status(self) -> list:
        out = []
        for state in self._states.values():
            out.append({
                "cam_id":    state.cam_id,
                "name":      state.name,
                "ip":        state.cfg["ip"],
                "connected": state.connected,
                "rs485_open": state.hrs is not None,
            })
        return out


# =============================================================================
# Logging helper
# =============================================================================

def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =============================================================================
# Flask API
# =============================================================================

app     = Flask(__name__)
_sdk    = DVNetSDK()
_bridge = BridgeManager(_sdk)


@app.route("/led/display", methods=["POST"])
def api_led_display():
    """
    Called by app.py after a plate is saved to DB.
    Body: {"cam_id": 0, "plate": "KAA123B", "device_type": "Entry1"}
    """
    try:
        data    = request.get_json(force=True) or {}
        cam_id  = int(data.get("cam_id", -1))
        plate   = str(data.get("plate", "")).strip().upper()
        if not plate or plate in ("K.", "K", ""):
            return jsonify({"ok": False, "error": "No valid plate text"}), 400
        result = _bridge.send_plate(cam_id, plate)
        return jsonify(result), (200 if result["ok"] else 503)
    except Exception as e:
        _log(f"[API /led/display] ERROR: {traceback.format_exc()}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/led/relay", methods=["POST"])
def api_relay():
    """
    Manual relay control.
    Body: {"cam_id": 0, "relay": 0, "cmd": "pulse", "ms": 500}
    cmd: "open" | "close" | "pulse"
    """
    try:
        data      = request.get_json(force=True) or {}
        cam_id    = int(data.get("cam_id", 0))
        relay_idx = int(data.get("relay",  0))
        cmd       = str(data.get("cmd",   "pulse"))
        hold_ms   = int(data.get("ms",    500))
        result    = _bridge.relay_cmd(cam_id, relay_idx, cmd, hold_ms)
        return jsonify(result), (200 if result["ok"] else 503)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/status", methods=["GET"])
def api_status():
    """Returns connection status for all 4 cameras."""
    return jsonify({"ok": True, "cameras": _bridge.status()})


@app.route("/led/resend_line0/<int:cam_id>", methods=["POST"])
def api_resend_line0(cam_id):
    """Re-send the permanent Line 0 text (useful if board rebooted)."""
    state = _bridge._states.get(cam_id)
    if state is None:
        return jsonify({"ok": False, "error": "Unknown cam_id"}), 404
    if not state.connected:
        return jsonify({"ok": False, "error": "Not connected"}), 503
    _bridge._send_line0(state)
    return jsonify({"ok": True, "text": state.line0_text})


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  AMAAN ANPR — DVNetSDK Bridge")
    print(f"  Python {_BITS}-bit — OK")
    print(f"  Listening on port {BRIDGE_PORT}")
    print("=" * 60)
    print()

    _bridge.start()

    try:
        # use_reloader=False is CRITICAL — reloader forks, second process
        # tries to load DVNetSDK.dll again and the SDK init double-fires
        app.run(host=BRIDGE_HOST, port=BRIDGE_PORT,
                debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        _log("Shutting down bridge...")
        _bridge.stop()
        _log("Done.")