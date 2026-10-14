"""What a node may do with the platform's cache, tried from a real node (GPU-200).

The cache of compiled wheels and kernels lives in a bucket of the platform's,
and a node holds no key to it: it asks the backend, with its agent's token, to
sign each operation, and the backend decides by the workspace of the token
what it may read (``global/`` and its own ``ws/<workspace>/``) and write (its
own alone). This script checks that on a node, where it matters, and prints
one ``CHECK`` line per thing it tried and a ``VERDICT`` at the end:

* no bucket key in the environment of the training - only the names of the
  variables are printed, never a value;
* a put and a get in the workspace's cache go through, the bytes come back the
  same, and a listing finds them;
* a put in ``global/`` is refused, and so is a path that climbs out with ``..``;
* a signed put takes no more bytes than it was signed for (GPU-218);
* the run's own keys (``RAVEX_S3_*``, R2's temporary ones for the workspace's
  bucket, GPU-217) list that bucket and are refused by ``gpuzero-cache``;
* ``ravex cache push`` and ``pull`` reach the cache through ``sign+https://``,
  and a push to ``global`` fails.

It uses ``RAVEX_METRICS_TOKEN``, which is the agent's token and so the same
one smith gives ``ravex cache``. At the end it compiles a small model, so the
push smith makes after the job has Inductor and Triton kernels to send.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid

FAILED: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print("CHECK %-4s %s%s" % ("ok" if ok else "FAIL", name, (" - " + detail) if detail else ""), flush=True)
    if not ok:
        FAILED.append(name)


def ask(scope: str, op: str, path: str, size: int | None = None) -> tuple[int, dict]:
    """One operation on the signing route: the status and the answer."""
    body = {"op": op, "path": path}
    if size is not None:
        body["size"] = size
    request = urllib.request.Request(
        os.environ["RAVEX_METRICS_ENDPOINT"].rstrip("/") + "/api/cache/" + scope,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "ravex",
                 "Authorization": "Bearer " + os.environ["RAVEX_METRICS_TOKEN"]},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, {"detail": exc.read().decode("utf-8", "replace")[:200]}


def transfer(url: str, data: bytes | None = None) -> tuple[int, bytes]:
    """A GET, or a PUT of ``data``, on a signed URL: the status and the body."""
    request = urllib.request.Request(url, data=data, method="PUT" if data is not None else "GET",
                                     headers={"User-Agent": "ravex", "Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()[:200]


def environment() -> None:
    # Names only: a value printed here would be in the job's log for good.
    suspicious = re.compile(r"(KEY|SECRET|SESSION|AWS_|R2_|CACHE)", re.I)
    names = sorted(name for name in os.environ if suspicious.search(name))
    print("ENV names that look like credentials: %s" % (", ".join(names) or "none"), flush=True)
    check("no RAVEX_CACHE_* in the training's environment", not any(n.startswith("RAVEX_CACHE_") for n in names))
    check("no AWS_* or R2_* keys in the environment",
          not any(n.startswith(("AWS_ACCESS", "AWS_SECRET", "R2_")) for n in names))


def route() -> None:
    probe = "probe/%s.bin" % uuid.uuid4().hex
    data = os.urandom(4096)

    status, answer = ask("workspace", "put", probe, len(data))
    check("workspace put is signed", status == 200 and "url" in answer, "%d %s" % (status, answer.get("detail", "")))
    if status == 200:
        status, _ = transfer(answer["url"], data)
        check("signed PUT reaches the bucket", status == 200, str(status))

    status, answer = ask("workspace", "list", "probe/")
    check("workspace listing finds it", status == 200 and probe in answer.get("paths", []),
          "%d, %d paths" % (status, len(answer.get("paths", []))))

    status, answer = ask("workspace", "get", probe)
    if status == 200:
        status, back = transfer(answer["url"])
        check("signed GET brings the same bytes", status == 200 and back == data, str(status))
    else:
        check("workspace get is signed", False, str(status))

    status, answer = ask("workspace", "put", "probe/longer.bin", 16)
    if status == 200:
        status, _ = transfer(answer["url"], os.urandom(64))
        check("a PUT longer than it was signed for is refused by the bucket", status >= 400, str(status))
    else:
        check("workspace put is signed (for the length check)", False, str(status))

    status, answer = ask("global", "put", probe, len(data))
    check("global put is refused", status == 403, "%d %s" % (status, answer.get("detail", "")))

    # A prefix, not "": the backend refuses an empty path in every scope.
    status, answer = ask("global", "list", "probe/")
    check("global listing is allowed", status == 200,
          "%d %s" % (status, answer.get("detail", "%d paths" % len(answer.get("paths", [])))))

    status, answer = ask("workspace", "put", "../1/escape.bin", 4)
    check("a path with .. is refused", status == 403, "%d %s" % (status, answer.get("detail", "")))

    status, answer = ask("ws", "list", "")
    check("a scope other than global and workspace is refused", status >= 400, str(status))


def listing(bucket: str, prefix: str) -> int:
    """The status of a ListObjectsV2 on ``bucket``, signed with the run's keys.

    A SigV4 of a few lines and not a library: the training's environment has
    no boto3, and this only has to tell 200 from 403."""
    import datetime
    import hashlib
    import hmac
    import urllib.parse

    endpoint = os.environ["RAVEX_STORAGE_ENDPOINT"].rstrip("/")
    region = os.environ.get("RAVEX_STORAGE_REGION") or "auto"
    host = urllib.parse.urlparse(endpoint).netloc
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    query = urllib.parse.urlencode(sorted({"list-type": "2", "max-keys": "1", "prefix": prefix}.items()),
                                   quote_via=urllib.parse.quote, safe="")
    headers = {"host": host, "x-amz-content-sha256": hashlib.sha256(b"").hexdigest(), "x-amz-date": stamp}
    if os.environ.get("RAVEX_S3_SESSION_TOKEN"):
        headers["x-amz-security-token"] = os.environ["RAVEX_S3_SESSION_TOKEN"]
    names = ";".join(sorted(headers))
    canonical = "\n".join(["GET", "/" + bucket, query, "".join("%s:%s\n" % (k, headers[k]) for k in sorted(headers)),
                           names, headers["x-amz-content-sha256"]])
    scope = "%s/%s/s3/aws4_request" % (day, region)
    to_sign = "\n".join(["AWS4-HMAC-SHA256", stamp, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    key = ("AWS4" + os.environ["RAVEX_S3_SECRET_KEY"]).encode()
    for part in (day, region, "s3", "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    headers["Authorization"] = "AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s" % (
        os.environ["RAVEX_S3_ACCESS_KEY"], scope, names, signature)
    del headers["host"]
    status, _ = transfer_with(endpoint + "/" + bucket + "?" + query, headers)
    return status


def transfer_with(url: str, headers: dict) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers=dict(headers, **{"User-Agent": "ravex"}))
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()[:200]


def run_keys() -> None:
    # The run's keys are R2's temporary ones for the workspace's own bucket
    # (GPU-217): they must open that bucket and not the cache's.
    if not (os.environ.get("RAVEX_S3_ACCESS_KEY") and os.environ.get("RAVEX_STORAGE_ENDPOINT")):
        print("no run keys in the environment: nothing to try", flush=True)
        return
    own = os.environ.get("RAVEX_STORAGE_BUCKET", "")
    status = listing(own, os.environ.get("RAVEX_STORAGE_PREFIX", ""))
    check("the run's keys list the workspace's own bucket", status == 200, "%s: %d" % (own, status))
    status = listing("gpuzero-cache", "")
    check("the run's keys do not open gpuzero-cache", status in (401, 403), str(status))


def ravex_cache() -> None:
    signed = "sign+" + os.environ["RAVEX_METRICS_ENDPOINT"].rstrip("/") + "/api/cache/"
    env = dict(os.environ, RAVEX_CACHE_TOKEN=os.environ["RAVEX_METRICS_TOKEN"])
    name = "probe-" + uuid.uuid4().hex[:8]
    with tempfile.TemporaryDirectory() as there, tempfile.TemporaryDirectory() as back:
        with open(os.path.join(there, "kernel.bin"), "wb") as handle:
            handle.write(os.urandom(2048))

        def cli(action: str, directory: str, scope: str) -> subprocess.CompletedProcess:
            done = subprocess.run([sys.executable, "-m", "ravex._cli", "cache", action, "--name", name,
                                   "--dir", directory, "--store", signed + scope],
                                  env=env, capture_output=True, text=True)
            print("ravex cache %s %s -> %d\n%s%s" % (action, scope, done.returncode, done.stdout, done.stderr), flush=True)
            return done

        check("ravex cache push to workspace", cli("push", there, "workspace").returncode == 0)
        cli("pull", back, "workspace")
        check("ravex cache pull from workspace brings the file back",
              os.path.isfile(os.path.join(back, "kernel.bin")))
        check("ravex cache push to global fails", cli("push", there, "global").returncode != 0)


def compile_something() -> None:
    import torch

    if not torch.cuda.is_available():
        print("no GPU: nothing compiled for smith to push", flush=True)
        return
    model = torch.nn.Sequential(torch.nn.Linear(512, 2048), torch.nn.GELU(), torch.nn.Linear(2048, 512)).cuda()
    compiled = torch.compile(model)
    compiled(torch.randn(64, 512, device="cuda")).sum().backward()
    torch.cuda.synchronize()
    print("compiled a small MLP: Inductor and Triton kernels are in the job's caches", flush=True)


def main() -> None:
    for name in ("RAVEX_METRICS_ENDPOINT", "RAVEX_METRICS_TOKEN"):
        if not os.environ.get(name):
            print("VERDICT FAIL - %s is not set, so this is not a node of the platform" % name, flush=True)
            sys.exit(1)
    environment()
    route()
    run_keys()
    ravex_cache()
    compile_something()
    print("VERDICT %s%s" % ("FAIL - " if FAILED else "ok", ", ".join(FAILED)), flush=True)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
