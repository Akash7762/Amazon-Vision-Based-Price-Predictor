"""
Checks a running copy of the API: it answers, it's the released model, the
three test photos get their reference prices, and a non-image is refused.

    python deploy/smoke_test.py http://localhost:8000            # the API itself
    python deploy/smoke_test.py https://<api>.onrender.com        # on Render
    python deploy/smoke_test.py https://<app>.vercel.app/api      # through the web app

Python's standard library only, so it runs anywhere without installing
anything. The expected answers are e2e/fixtures/expected.json (Phase 6).
--wait gives a sleeping free server time to wake up first.

Load checks (the free server once fell over under them):

    --burst 12 --burst-photo photo-1600.jpg   12 uploads of that photo at once
    --also big.jpg                            more photos that just need an answer
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "e2e" / "fixtures"
CENT = 0.015  # a cent either way, for float differences between CPUs


def call(url, body=None, content_type=None, timeout=120):
    """(status, parsed JSON or text, seconds)"""
    req = urllib.request.Request(url, data=body, headers={"Content-Type": content_type} if content_type else {})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    seconds = time.perf_counter() - started
    try:
        return status, json.loads(raw), seconds
    except ValueError:
        return status, raw.decode(errors="replace")[:200], seconds


def upload(base, name, data, content_type):
    boundary = uuid.uuid4().hex
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n").encode()
    body = head + data + f"\r\n--{boundary}--\r\n".encode()
    return call(f"{base}/predict", body, f"multipart/form-data; boundary={boundary}")


def wait_until_up(base, seconds):
    deadline = time.monotonic() + seconds
    started = time.monotonic()
    while True:
        try:
            status, body, _ = call(f"{base}/health", timeout=max(5, deadline - time.monotonic()))
            if status == 200:
                return body, time.monotonic() - started
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        if time.monotonic() > deadline:
            sys.exit(f"FAIL: {base}/health didn't answer within {seconds} s")
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser(description="Check a running copy of the price API.")
    ap.add_argument("base", help="where the API answers, e.g. http://localhost:8000 or https://<app>.vercel.app/api")
    ap.add_argument("--wait", type=int, default=90, help="seconds to wait for it to come up (default 90)")
    ap.add_argument("--burst", type=int, default=0, help="also send this many uploads at once")
    ap.add_argument("--burst-photo", type=Path, default=FIXTURES / "photos" / "olive-oil.jpg",
                    help="the photo the burst sends (default: a test photo)")
    ap.add_argument("--also", type=Path, nargs="*", default=[], help="more photos that must get an answer")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    expected = json.loads((FIXTURES / "expected.json").read_text())
    failures = []

    def check(ok, what):
        print(f"  {'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            failures.append(what)

    health, waited = wait_until_up(base, args.wait)
    print(f"{base} is up (answered after {waited:.0f} s)")
    check(health.get("model_version") == expected["model_version"], f"model version {health.get('model_version')}")
    check(health.get("model_sha256") == expected["model_sha256"], "model file is the released one (sha256)")

    for photo in expected["photos"]:
        data = (FIXTURES / "photos" / photo["file"]).read_bytes()
        status, body, seconds = upload(base, photo["file"], data, "image/jpeg")
        if status != 200:
            check(False, f"{photo['file']}: HTTP {status} {body}")
            continue
        got = (body["price"], body["range"]["low"], body["range"]["high"])
        want = (photo["price"], photo["low"], photo["high"])
        same = all(abs(g - w) <= CENT for g, w in zip(got, want))
        check(same, f"{photo['file']}: ${got[0]:.2f} (${got[1]:.2f}-${got[2]:.2f}), "
                    f"expected ${want[0]:.2f}, {seconds:.2f} s")

    status, body, _ = upload(base, "notes.txt", b"not a photo", "text/plain")
    check(status == 415, f"a text file is refused with 415 (got {status})")

    for path in args.also:
        status, body, seconds = upload(base, path.name, path.read_bytes(), "image/jpeg")
        check(status == 200, f"{path.name}: HTTP {status}, {seconds:.1f} s"
                             + (f", ${body['price']:.2f}" if status == 200 else f" {body}"))

    if args.burst:
        data = args.burst_photo.read_bytes()
        with ThreadPoolExecutor(args.burst) as pool:
            answers = list(pool.map(lambda i: upload(base, f"burst-{i}.jpg", data, "image/jpeg"), range(args.burst)))
        ok = [a for a in answers if a[0] == 200]
        prices = {a[1]["price"] for a in ok}
        check(len(ok) == args.burst and len(prices) == 1,
              f"{args.burst} uploads of {args.burst_photo.name} at once: {len(ok)} answered "
              f"(statuses {sorted(a[0] for a in answers)}), slowest {max(a[2] for a in answers):.1f} s")
        status, _, _ = call(f"{base}/health")
        check(status == 200, f"still up afterwards (health: HTTP {status})")

    if failures:
        sys.exit(f"\n{len(failures)} check(s) failed")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
