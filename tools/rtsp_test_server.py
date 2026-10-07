#!/usr/bin/env python3
"""
Local RTSP test server - lets you test main.py without a real camera.

Streams a looping "slideshow" of images as live MJPEG video (RTP/JPEG, RFC 2435)
over RTSP with TCP interleaving - the same transport main.py forces for cameras.
Pure Python + OpenCV, nothing else to install.

    python tools/rtsp_test_server.py --scene fire.jpg:20 --scene street.jpg:10 ^
           --user admin --password "Test@123"

    -> rtsp://127.0.0.1:8554/live   (Digest login, like most IP cameras)

Then point main.py at it (username/password come from .env as usual):

    python main.py --source rtsp://127.0.0.1:8554/live

Test options: --stall-after N freezes the video once after N seconds while
keeping the connection open (tests the read timeout / auto-reconnect).
Killing and restarting the server tests reconnecting after a dropped stream.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import socket
import struct
import threading
import time
from base64 import b64decode
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REALM = "FireSmokeTest"


def log(message: str) -> None:
    print(f"[SERVER {datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {message}", flush=True)


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Video source: looping scenes with a clock overlay (so frames keep changing)
# -----------------------------------------------------------------------------
class SceneSource:
    def __init__(self, scenes: list[tuple[str, float]], width: int, height: int):
        self.scenes = []
        for path, seconds in scenes:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit(f"Cannot read image: {path}")
            self.scenes.append((Path(path).stem, self._cover(image, width, height), seconds))
        self.total = sum(s for _, _, s in self.scenes)
        self.t0 = time.monotonic()
        self._shown: int | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Scale to fill width x height, cropping the overflow (keeps aspect ratio)."""
        h, w = image.shape[:2]
        scale = max(width / w, height / h)
        resized = cv2.resize(image, (max(width, round(w * scale)), max(height, round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        y, x = (resized.shape[0] - height) // 2, (resized.shape[1] - width) // 2
        return np.ascontiguousarray(resized[y:y + height, x:x + width])

    def render(self) -> tuple[np.ndarray, str]:
        t = (time.monotonic() - self.t0) % self.total
        for index, (name, image, seconds) in enumerate(self.scenes):
            if t < seconds:
                break
            t -= seconds
        with self._lock:  # log the moment a new scene is first sent (for latency tests)
            if index != self._shown:
                self._shown = index
                log(f"scene -> {name}")
        frame = image.copy()
        h, w = frame.shape[:2]
        text = f"TEST CAMERA  {datetime.now():%Y-%m-%d %H:%M:%S}  scene: {name}"
        cv2.rectangle(frame, (0, h - 34), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        x = int((time.monotonic() * 120) % (w - 20))  # moving marker = visible motion
        cv2.rectangle(frame, (x, h - 30), (x + 20, h - 26), (0, 255, 255), -1)
        return frame, name


# -----------------------------------------------------------------------------
# RTP/JPEG packetiser (RFC 2435)
# -----------------------------------------------------------------------------
def split_jpeg(data: bytes) -> tuple[int, int, int, bytes, bytes]:
    """Baseline JPEG -> (rtp type, width, height, 2 quantisation tables, scan data)."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    pos, tables, jtype, width, height = 2, {}, None, 0, 0
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            raise ValueError(f"JPEG marker expected at byte {pos}")
        marker = data[pos + 1]
        pos += 2
        if marker == 0xFF or 0xD0 <= marker <= 0xD8:
            pos -= 1 if marker == 0xFF else 0
            continue
        seg_len = struct.unpack(">H", data[pos:pos + 2])[0]
        seg = data[pos + 2:pos + seg_len]
        if marker == 0xDB:                                   # DQT
            i = 0
            while i < len(seg):
                if seg[i] >> 4:
                    raise ValueError("16-bit quantisation tables are not supported")
                tables[seg[i] & 0x0F] = seg[i + 1:i + 65]
                i += 65
        elif marker == 0xC0:                                 # SOF0 (baseline)
            height, width = struct.unpack(">HH", seg[1:5])
            sampling = [seg[7 + 3 * k] for k in range(seg[5])]
            if len(sampling) != 3 or sampling[1:] != [0x11, 0x11] or sampling[0] not in (0x21, 0x22):
                raise ValueError(f"unsupported chroma sampling {sampling}")
            jtype = 0 if sampling[0] == 0x21 else 1          # 4:2:2 -> 0, 4:2:0 -> 1
        elif 0xC1 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            raise ValueError("only baseline JPEG is supported")
        elif marker == 0xDD and struct.unpack(">H", seg[:2])[0]:
            raise ValueError("JPEG restart markers are not supported")
        elif marker == 0xDA:                                 # SOS: entropy-coded data follows
            scan = data[pos + seg_len:]
            if scan.endswith(b"\xff\xd9"):
                scan = scan[:-2]
            return jtype, width, height, tables[0] + tables[1], scan
        pos += seg_len
    raise ValueError("JPEG has no image data")


class RtpJpegPacketizer:
    def __init__(self, max_payload: int = 1400):
        self.seq = random.randint(0, 0xFFFF)
        self.ssrc = random.getrandbits(32)
        self.max_payload = max_payload

    def packets(self, jpeg: bytes, timestamp: int) -> list[bytes]:
        jtype, width, height, qtables, scan = split_jpeg(jpeg)
        if width % 8 or height % 8 or width > 2040 or height > 2040:
            raise ValueError("RTP/JPEG needs width/height multiples of 8, max 2040")
        out, offset = [], 0
        while True:
            header = struct.pack(">I", offset) + struct.pack(">BBBB", jtype, 255, width // 8, height // 8)
            if offset == 0:  # Q=255: quantisation tables travel with every frame
                header += struct.pack(">BBH", 0, 0, len(qtables)) + qtables
            chunk = scan[offset:offset + self.max_payload - len(header)]
            offset += len(chunk)
            last = offset >= len(scan)
            rtp = struct.pack(">BBHII", 0x80, (0x80 if last else 0) | 26, self.seq, timestamp & 0xFFFFFFFF, self.ssrc)
            self.seq = (self.seq + 1) & 0xFFFF
            out.append(rtp + header + chunk)
            if last:
                return out


# -----------------------------------------------------------------------------
# RTSP session (one per TCP connection)
# -----------------------------------------------------------------------------
class RtspSession(threading.Thread):
    def __init__(self, conn: socket.socket, addr, server: "Server"):
        super().__init__(daemon=True)
        self.conn, self.addr, self.server = conn, addr, server
        self.nonce = md5(os.urandom(16).hex())
        self.session_id = str(random.randint(10_000_000, 99_999_999))
        self.channel = 0
        self.send_lock = threading.Lock()
        self.alive = True
        self.streamer: threading.Thread | None = None

    def send(self, data: bytes) -> None:
        with self.send_lock:
            self.conn.sendall(data)

    def reply(self, cseq: str, status: str = "200 OK", headers: list[str] | None = None, body: bytes = b"") -> None:
        lines = [f"RTSP/1.0 {status}", f"CSeq: {cseq}", "Server: FireSmokeTestServer"] + (headers or [])
        if body:
            lines.append(f"Content-Length: {len(body)}")
        self.send(("\r\n".join(lines) + "\r\n\r\n").encode() + body)

    def run(self) -> None:
        log(f"client connected {self.addr[0]}:{self.addr[1]}")
        buf = b""
        try:
            while self.alive:
                data = self.conn.recv(65536)
                if not data:
                    break
                buf += data
                while buf:
                    if buf[:1] == b"$":  # interleaved RTCP from the client - ignore
                        if len(buf) < 4 or len(buf) < 4 + struct.unpack(">H", buf[2:4])[0]:
                            break
                        buf = buf[4 + struct.unpack(">H", buf[2:4])[0]:]
                        continue
                    end = buf.find(b"\r\n\r\n")
                    if end < 0:
                        break
                    lines = buf[:end].decode("latin-1").split("\r\n")
                    headers = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:])}
                    body_len = int(headers.get("content-length", "0") or 0)
                    if len(buf) < end + 4 + body_len:
                        break
                    buf = buf[end + 4 + body_len:]
                    method, uri = (lines[0].split(" ") + ["", ""])[:2]
                    self.handle(method, uri, headers)
        except OSError:
            pass
        finally:
            self.alive = False
            try:
                self.conn.close()
            except OSError:
                pass
            log(f"client disconnected {self.addr[0]}:{self.addr[1]}")

    def authorized(self, method: str, headers: dict) -> bool:
        srv = self.server
        if not srv.user:
            return True
        auth = headers.get("authorization", "")
        if srv.auth == "basic":
            if not auth.lower().startswith("basic "):
                return False
            try:
                user, _, password = b64decode(auth[6:]).decode("utf-8").partition(":")
            except ValueError:
                return False
            return user == srv.user and password == srv.password
        if not auth.lower().startswith("digest "):
            return False
        p = {k.lower(): a or b for k, a, b in re.findall(r'(\w+)=(?:"([^"]*)"|([^\s,]+))', auth[7:])}
        if p.get("username") != srv.user or p.get("nonce") != self.nonce:
            return False
        ha1 = md5(f"{srv.user}:{REALM}:{srv.password}")
        ha2 = md5(f"{method}:{p.get('uri', '')}")
        if p.get("qop"):
            expected = md5(f"{ha1}:{self.nonce}:{p.get('nc')}:{p.get('cnonce')}:{p.get('qop')}:{ha2}")
        else:
            expected = md5(f"{ha1}:{self.nonce}:{ha2}")
        return p.get("response") == expected

    def handle(self, method: str, uri: str, headers: dict) -> None:
        cseq = headers.get("cseq", "0")
        srv = self.server
        if method == "OPTIONS":
            return self.reply(cseq, headers=["Public: OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN, GET_PARAMETER"])
        path = re.sub(r"^rtsp://[^/]+", "", uri).rstrip("/")
        if not (path == srv.path or path.startswith(srv.path + "/")):
            log(f"{method} {path or '/'} -> 404 (the stream path is {srv.path})")
            return self.reply(cseq, "404 Not Found")
        if not self.authorized(method, headers):
            if headers.get("authorization"):
                log(f"{method} -> 401 wrong username/password")
            challenge = (f'Basic realm="{REALM}"' if srv.auth == "basic"
                         else f'Digest realm="{REALM}", nonce="{self.nonce}"')
            return self.reply(cseq, "401 Unauthorized", [f"WWW-Authenticate: {challenge}"])
        if method == "DESCRIBE":
            sdp = ("v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=Fire test stream\r\nc=IN IP4 0.0.0.0\r\nt=0 0\r\n"
                   "a=control:*\r\nm=video 0 RTP/AVP 26\r\na=rtpmap:26 JPEG/90000\r\n"
                   f"a=framerate:{srv.fps}\r\na=control:trackID=0\r\n")
            base = uri.rstrip("/") + "/"
            return self.reply(cseq, headers=["Content-Type: application/sdp", f"Content-Base: {base}"],
                              body=sdp.encode())
        if method == "SETUP":
            transport = headers.get("transport", "")
            match = re.search(r"interleaved=(\d+)-(\d+)", transport)
            if "TCP" not in transport.upper() or not match:
                log(f"SETUP with non-TCP transport '{transport}' -> 461 (this server is TCP only)")
                return self.reply(cseq, "461 Unsupported Transport")
            self.channel = int(match.group(1))
            return self.reply(cseq, headers=[f"Transport: RTP/AVP/TCP;unicast;interleaved={match.group(1)}-{match.group(2)}",
                                             f"Session: {self.session_id};timeout=60"])
        if method == "PLAY":
            self.reply(cseq, headers=[f"Session: {self.session_id}", "Range: npt=0.000-"])
            if self.streamer is None:
                log(f"PLAY - streaming to {self.addr[0]}:{self.addr[1]} ({srv.width}x{srv.height} @ {srv.fps} fps, "
                    f"TCP interleaved)")
                self.streamer = threading.Thread(target=self.stream, daemon=True)
                self.streamer.start()
            return None
        if method in ("GET_PARAMETER", "SET_PARAMETER"):
            return self.reply(cseq, headers=[f"Session: {self.session_id}"])
        if method == "TEARDOWN":
            self.reply(cseq, headers=[f"Session: {self.session_id}"])
            self.alive = False
            return None
        return self.reply(cseq, "405 Method Not Allowed")

    def stream(self) -> None:
        srv = self.server
        packetizer = RtpJpegPacketizer()
        timestamp, step = random.getrandbits(32), round(90000 / srv.fps)
        started = next_t = time.monotonic()
        while self.alive:
            if srv.should_stall(time.monotonic() - started):
                time.sleep(0.1)
                next_t = time.monotonic()
                continue
            frame, _ = srv.source.render()
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, srv.quality])
            data = b"".join(b"$" + bytes([self.channel]) + struct.pack(">H", len(p)) + p
                            for p in packetizer.packets(jpeg.tobytes(), timestamp))
            try:
                self.send(data)
            except OSError:
                break
            timestamp = (timestamp + step) & 0xFFFFFFFF
            next_t += 1.0 / srv.fps
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()


class Server:
    def __init__(self, args: argparse.Namespace, source: SceneSource):
        self.source = source
        self.path = "/" + args.path.strip("/")
        self.user, self.password, self.auth = args.user, args.password, args.auth
        self.fps, self.quality = args.fps, args.quality
        self.width, self.height = args.width, args.height
        self.stall_after, self.stall_for = args.stall_after, args.stall_for
        self._stall_state = "pending" if args.stall_after > 0 else "done"
        self._stall_lock = threading.Lock()

    def should_stall(self, elapsed: float) -> bool:
        """Freeze the video once (first client to reach stall_after) for stall_for seconds."""
        with self._stall_lock:
            if self._stall_state == "pending" and elapsed >= self.stall_after:
                self._stall_state, self._stall_until = "active", time.monotonic() + self.stall_for
                log(f"STALL: video frozen for {self.stall_for:g} s (connection kept open)")
            if self._stall_state == "active":
                if time.monotonic() < self._stall_until:
                    return True
                self._stall_state = "done"
                log("STALL finished")
            return False


def parse_scene(text: str) -> tuple[str, float]:
    path, sep, seconds = text.rpartition(":")
    if not sep or not seconds.replace(".", "", 1).isdigit():
        return text, 10.0
    return path, float(seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RTSP test server (MJPEG over RTSP/TCP)")
    parser.add_argument("--scene", action="append", required=True, metavar="IMAGE[:SECONDS]",
                        help="image to show and for how long (repeat for a looping sequence)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument("--path", default="live")
    parser.add_argument("--user", default="", help="require this username (no login if empty)")
    parser.add_argument("--password", default="")
    parser.add_argument("--auth", choices=("digest", "basic"), default="digest")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality")
    parser.add_argument("--stall-after", type=float, default=0, help="freeze the video once after N s")
    parser.add_argument("--stall-for", type=float, default=20, help="length of the freeze in seconds")
    args = parser.parse_args()
    if args.width % 16 or args.height % 16:
        parser.error("--width and --height must be multiples of 16")

    source = SceneSource([parse_scene(s) for s in args.scene], args.width, args.height)
    server = Server(args, source)
    listener = socket.create_server((args.host, args.port))
    listener.settimeout(0.5)
    scenes = ", ".join(f"{name} {secs:g}s" for name, _, secs in source.scenes)
    login = f"{args.auth} login as '{args.user}'" if args.user else "no login"
    log(f"RTSP test server on rtsp://{args.host}:{args.port}{server.path} ({login}; scenes: {scenes})")
    try:
        while True:
            try:
                conn, addr = listener.accept()
            except socket.timeout:
                continue
            conn.settimeout(None)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            RtspSession(conn, addr, server).start()
    except KeyboardInterrupt:
        log("stopped")
    finally:
        listener.close()


if __name__ == "__main__":
    main()
