# Design decisions and tradeoffs

### Why idempotency, and why Redis `SET NX EX`
A merchant's client can always retry a request that timed out without knowing whether
it actually went through - without idempotency, that retry would charge the card twice.
`SETNX key "processing" EX 60` atomically claims the key so a second, concurrent
request with the same `Idempotency-Key` sees it's already in flight (409) rather than
racing the first one; once the first request finishes, it overwrites the key with the
final result (`EX 24h`), so later replays return the cached outcome instead of
re-running authorization. **Simplification:** unlike real Stripe, this sim doesn't hash
the request body, so reusing a key with different parameters silently returns the
original result instead of erroring.

### Why retries with backoff, and why they're centralized
Every inter-service call in this system funnels through one function,
`common.http.post_with_retry` - this was a deliberate refactor in Day 4 (it started as
copy-pasted `httpx.post` + `try/except` in each service). Centralizing it meant the
Day 5 integration tests could get the entire real chain running in-process just by
monkeypatching that one function per service, instead of mocking each service
individually.

### Reversal keyed by a client-generated `request_id`
The failure mode is specifically "the issuer approved, but the response never arrived."
At the moment the gateway needs to clean that up, it has, by definition, never
received anything from the issuer - no server-generated authorization id, no
confirmation of anything. The only thing it's guaranteed to still have is the
`request_id` *it itself generated* before making the call. So the issuer persists its
approval keyed by that same id (sent by the caller, not invented by the issuer), which
lets a reversal reference "undo whatever you did for this request" using information
the caller already had - this mirrors how real card networks use the original
transaction's reference for a network reversal/advice message, rather than an id from
a response that may never arrive.

One bug this surfaced: a retried authorize request carries the *same* `request_id` on
every attempt. The issuer's first attempt persists the record and then sleeps (to
simulate the lost response); the caller's retry is a brand new HTTP request that also
reaches the issuer and also tries to record an approval for the same `request_id` -
which collided with the primary key and raised an unhandled exception. Fixed by making
`record_authorization` a no-op if that `request_id` is already recorded
(`services/issuer/accounts.py`).

### Why settlement polls the gateway over HTTP instead of a second Redis pattern
Redis already demonstrates one pattern (the idempotency lock). Settlement could have
used a Redis sorted set as a delay queue instead, but that would mean two different
inter-service communication styles in one project. Staying consistent with the rule
used everywhere else - services never share a database file, they talk over HTTP - felt
like the better teaching choice, even though a real delay queue is a reasonable
alternative design.

### Why services never share a database file
Each service (`gateway`, `issuer`, `settlement`, ...) has its own SQLite file and no
other service ever opens it directly. This is what makes "one sqlite file per service"
in the setup checklist meaningful rather than cosmetic: it's the thing that forces
every cross-service interaction to go through an HTTP contract, which is the whole
point of simulating separate companies in the first place.

### Why integration tests don't need Docker
All five services are plain FastAPI apps. Since every inter-service call already goes
through the one `post_with_retry` chokepoint (see above), the test suite monkeypatches
it to dispatch to each service's own `TestClient` based on URL prefix instead of making
a real network call. A thread pool + `future.result(timeout=...)` makes this faithful
even for the reversal scenario: a handler that sleeps past its caller's timeout
genuinely times out in the test the same way a real `httpx` call would, abandoned
thread and all. The one exception is idempotency, which needs a real Redis - CI runs
one as a service container; locally, `docker compose up -d redis` is enough.

### Simplified settlement fee
`settlement_fee_bps` (290 = 2.9%) is percentage-only. Real processors also charge a
flat per-transaction fee (Stripe's is $0.30) - omitted here to keep the ledger math in
one dimension.

### The Day-2 bind-mount lesson
The first version of `docker-compose.yml` bind-mounted the whole repo (`.:/app`) into
every container for live-reload convenience. On Windows this silently shadowed each
image's own `.venv` with the host's Windows-style venv (`Scripts/` not `bin/`), so
`uvicorn` wasn't actually importable at runtime - and it went unnoticed because a
stray local `uvicorn` process from manual testing was still squatting on port 8000,
answering health checks while the container itself had crashed. The fix was to drop
the bind mounts entirely: the image's own copy of the source is authoritative, and a
rebuild (`docker compose up --build`) is required to pick up code changes. Slower
inner loop, but no more silent host/container path mismatches.

### Fraud features: one definition, two paths, and a parity test between them
The fraud model is trained on features computed offline from a CSV
(`ml/offline_features.py`) but served from features computed live in Redis
(`services/fraud/features.py`). If the two drift apart - a window off by one, a
boundary that's inclusive on one side and exclusive on the other - the model silently
scores inputs it never saw in training (training/serving skew). Three things guard
against that:

- **One definition.** Feature names and window sizes live only in
  `services/fraud/features.py`; the offline code imports them.
- **A parity test.** `tests/unit/test_feature_parity.py` replays generated traffic
  through Redis (fakeredis) and requires the same numbers as the offline path on every
  row; `python -m ml.check_parity` does the same at larger scale against real Redis.
- **Boundary unit tests.** Random millisecond timestamps almost never land exactly on a
  window edge, so the parity test alone missed a deliberately planted inclusive/exclusive
  bug. `test_short_window_boundary_is_exclusive` catches it.

**The current payment's outcome is never a feature.** A real-time scorer runs before
the issuer has decided, so it can't know whether this payment will be declined. Using it
offline would leak the answer into training. So each payment touches Redis twice:
`observe()` at score time records the attempt and returns features, then
`record_outcome()` after authorization records a decline. A side effect: a payment the
fraud check blocks still counts as an attempt, so an attacker can't reset their own
counters by getting blocked.

**Sorted sets, one round trip.** Each window is a sorted set scored by timestamp; trim
with `ZREMRANGEBYSCORE` (inclusive, which drops an event exactly `window` seconds old),
count with `ZCOUNT` using an exclusive `(` lower bound. Distinct cards per IP is a sorted
set keyed by card with its last-seen time as the score. All trims, adds, counts and
TTLs go in one `MULTI` pipeline - one network round trip per score, no Lua needed.

### Why the fraud service serves gradient boosting, and no sequence model (yet)
The original plan had an optional GRU over each IP's recent events. It was skipped on
purpose: gradient boosting on the six window features plus amount already reached
F1 0.95 on the seed-42 test slice (vs 0.54 for the rules), and PyTorch would have added
~200 MB to the ML dependencies and slowed every CI run. In hindsight the "little room
left" argument was weaker than it looked: across five dataset seeds gradient boosting
averages F1 0.87 (0.74-0.96), and most of the gap is false positives on the flash sale
(`ml/results/seeds.md`, `docs/fraud.md`). That's a better case for a sequence model than
the one I skipped it on - a burst of *different customers* buying at one merchant looks
different over time from one actor cycling cards, even when the window counts match.

### Why the fraud model is trained at image build time
`ml/artifacts/` is gitignored, so the fraud image can't just copy a model in. Instead its
Docker build runs `ml.generate_data` + `ml.train` (fixed seed, ~30 s), gated by a
`TRAIN_FRAUD_MODEL` build arg so the other services' images are unchanged. The
alternative - committing `gbm.joblib` - would tie the repo to one scikit-learn version's
pickle format and put an opaque binary in review diffs. Building it from code keeps the
model reproducible from what's in git. Adding `.dockerignore` came with this: without it,
`COPY . .` also copied the host's `.venv` and any locally trained artifacts into the image.

The service reads the model's feature order and block threshold from the artifact's JSON
metadata rather than redefining them, so a retrained model can't silently be fed columns
in the wrong order.

### The fraud check fails open, with a 150 ms budget
The gateway asks the fraud service about every payment (when `FRAUD_CHECK_ENABLED`)
before it reaches the acquirer, and gives it 150 ms, one attempt, no retry. If the call
times out, errors or returns something unparseable, the payment **continues without a
fraud decision** (fail open).

- **Why open, not closed.** Failing closed turns a fraud-service outage into a total
  payment outage - every legitimate customer declined because a scoring dependency is
  down. Failing open lets fraud through only for as long as the outage lasts, and it's
  visible: every scored payment gets a `FraudCheck` row, so unscored ones are exactly the
  intents without one. A real system would alert on that gap.
- **Why no retry.** The budget is the point. A retry inside 150 ms either doesn't fit or
  doubles the load on a service that is probably already struggling.
- **`review` still goes through.** Review means "a human or a slower check should look at
  this later", not "stop" - only `block` skips the acquirer.
- **Blocked attempts are reported as declines.** The first version reported nothing for
  them, since they never reached the issuer. The final code review caught what that does
  to the features: blocked attempts still count as attempts, but not as declines, so an
  IP's decline rate *sinks toward 0* the longer it is blocked. In training, attack traffic
  is mostly declined, so a model scored mid-attack on a falling decline rate drifts toward
  "allow". To the requester a block is a decline, so it's recorded as one. Declines are
  also stamped with the attempt's own time rather than the time the outcome arrives, so
  they leave the window together with their attempt, as in the offline features.
- **Missing shopper details get a key of their own.** If the merchant doesn't send the
  shopper's IP (or email), the payment gets a unique placeholder. Falling back to the
  request's source address would have used the *merchant's server* IP, lumping all its
  shoppers into one apparent burst. The cost: a merchant that never sends these fields
  gets no IP/email velocity checks at all. A "field missing" feature would be the real fix.
- **Fingerprint, not token.** The vault mints a fresh random token per tokenize call, so
  the gateway sends an HMAC of the card number instead - otherwise retrying the same card
  would look like many different cards. The merchant's API key is hashed for the same
  reason: the fraud service never sees secrets or card numbers.

Measured on the Docker stack (`python -m scripts.fraud_latency`, `ml/results/latency.md`):
`/v1/score` is ~5 ms p50 / ~6 ms p95 on its own. On Day 5, turning the check on added
~17 ms p50 to a gateway payment intent - much more than the score call itself, because
`post_with_retry` opened a new TCP connection for every call. Day 6 switched it to one
pooled `httpx.Client` per process: the fraud check now adds ~8 ms p50 / ~8 ms p95, and
since every inter-service hop uses the same helper, the whole payment got faster too
(p50 with the check off went from 43 ms to 20 ms).

**Two things the demo surfaced:**
- **BIN spillover.** After a burst on BIN 424242, a clean customer on that BIN scored 0.82
  (`review`) for the next minute, vs 0.07 on a different BIN - the per-BIN window can't
  tell the attacker's traffic from everyone else's. Not blocked, but it's real collateral
  cost and worth watching if the block threshold is ever lowered.
- **`localhost` on Windows.** POSTs to `localhost` from the Windows host carried ~45 ms of
  extra latency that `127.0.0.1` and in-container calls didn't; the first latency run
  measured that instead of the service. The latency script uses `127.0.0.1`.

### Other excluded scope
Encryption at rest, authorization/capture as separate steps, chargebacks, and
multi-currency are all left out - see the README's "What's excluded" section. None of
these change the architecture; they'd each add a dimension (key management, a second
API call per payment, a dispute lifecycle, currency conversion) orthogonal to what this
project is teaching.
