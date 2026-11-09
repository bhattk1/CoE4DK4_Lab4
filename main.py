
import csv
import heapq
import random
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# COMPENG 4DK4 - LAB 4
# Simulation of ARQ Protocols
# ============================================================

OUT = Path("lab4_output")
OUT.mkdir(exist_ok=True)

SEEDS = [400442107, 2026, 41729]

PACKET_MS = 1.0
ONE_WAY_MS = 5.0
DEFAULT_TIMEOUT = 15.0

LONG_RUN = 1200

WINDOWS = [1, 2, 4, 6, 8, 10, 11, 12, 16, 24, 32]
RECEIVE_WINDOWS = [2, 4, 8]

ERROR_PROBS = [0, .01, .025, .05, .10, .15, .20, .30]
ACK_PROBS = [0, .01, .025, .05, .10, .15, .20, .30]

TIMEOUTS = [5, 8, 10, 11, 12, 15, 20, 30, 50, 80]


# ============================================================
# DISCRETE-EVENT ARQ SIMULATOR
# ============================================================

class ARQ:

    def __init__(
        self, protocol, ws, wr, pd, pa,
        timeout, count, seed, trace=False
    ):

        assert protocol in ("SW", "GBN", "SR")
        assert ws >= 1 and wr >= 1 and count >= 1
        assert timeout > 0
        assert 0 <= pd <= 1 and 0 <= pa <= 1

        self.protocol = protocol
        self.ws = ws
        self.wr = wr
        self.pd = pd
        self.pa = pa
        self.timeout = timeout
        self.count = count

        self.rng = random.Random(seed)

        self.q = []
        self.serial = 0

        self.time = 0.0

        self.base = 0
        self.next_seq = 0
        self.recv_base = 0

        self.buffer = set()
        self.acked = set()

        self.last_finish = {}
        self.timer_id = {}
        self.gbn_timer_id = 0

        self.pending = deque()
        self.pending_set = set()

        self.busy = False

        self.attempts = 0
        self.data_errors = 0
        self.ack_errors = 0
        self.duplicates = 0

        self.trace = [] if trace else None

        self.max_events = max(200000, count * 5000)

    def log(self, message):

        if self.trace is not None:
            if len(self.trace) < 350:
                self.trace.append(
                    f"{self.time:10.3f} ms  {message}"
                )

    def event(self, when, kind, *args):

        self.serial += 1

        heapq.heappush(
            self.q,
            (when, self.serial, kind, args)
        )

    def enqueue(self, seq):

        if (
            seq not in self.pending_set
            and seq < self.next_seq
            and seq >= self.base
        ):
            self.pending.append(seq)
            self.pending_set.add(seq)

    # --------------------------------------------------------
    # Sender
    # --------------------------------------------------------

    def start_send(self):

        if self.busy:
            return

        seq = None

        while self.pending:

            candidate = self.pending.popleft()
            self.pending_set.remove(candidate)

            if candidate >= self.base:

                if (
                    self.protocol != "SR"
                    or candidate not in self.acked
                ):
                    seq = candidate
                    break

        if seq is None:

            if (
                self.next_seq < self.count
                and self.next_seq < self.base + self.ws
            ):
                seq = self.next_seq
                self.next_seq += 1

        if seq is None:
            return

        self.busy = True
        self.attempts += 1

        self.log(
            f"TX START packet={seq} "
            f"attempt={self.attempts}"
        )

        self.event(
            self.time + PACKET_MS,
            "TX_END",
            seq
        )

    # --------------------------------------------------------
    # Timers
    # --------------------------------------------------------

    def arm_sr_timer(self, seq):

        self.timer_id[seq] = (
            self.timer_id.get(seq, 0) + 1
        )

        self.event(
            self.time + self.timeout,
            "SR_TIMEOUT",
            seq,
            self.timer_id[seq]
        )

    def arm_gbn_timer(self, reference=None):

        self.gbn_timer_id += 1

        if self.base < self.next_seq:

            if reference is None:
                reference = self.last_finish.get(
                    self.base, self.time
                )

            deadline = max(
                self.time,
                reference + self.timeout
            )

            self.event(
                deadline,
                "GBN_TIMEOUT",
                self.gbn_timer_id,
                self.base
            )

    # --------------------------------------------------------
    # Data packet transmission
    # --------------------------------------------------------

    def tx_end(self, seq):

        self.busy = False

        self.last_finish[seq] = self.time

        if self.protocol == "SR":

            if seq not in self.acked:
                self.arm_sr_timer(seq)

        elif seq == self.base:

            self.arm_gbn_timer(self.time)

        if self.rng.random() < self.pd:

            self.data_errors += 1

            self.log(
                f"DATA LOST packet={seq}"
            )

        else:

            self.event(
                self.time + ONE_WAY_MS,
                "RECEIVE",
                seq
            )

        self.start_send()

    # --------------------------------------------------------
    # Receiver
    # --------------------------------------------------------

    def receive(self, seq):

        if self.protocol == "SR":

            if seq < self.recv_base:

                self.duplicates += 1

                self.log(
                    f"DUPLICATE packet={seq}; "
                    "ACK again"
                )

                self.send_ack(seq)

            elif seq >= self.recv_base + self.wr:

                self.log(
                    f"RECEIVER WINDOW DROP packet={seq}; "
                    f"base={self.recv_base}"
                )

            else:

                if seq in self.buffer:
                    self.duplicates += 1

                self.buffer.add(seq)

                self.log(
                    f"RECEIVE packet={seq}; "
                    "individual ACK"
                )

                self.send_ack(seq)

                while self.recv_base in self.buffer:

                    self.buffer.remove(self.recv_base)

                    self.recv_base += 1

        else:

            if seq == self.recv_base:

                self.recv_base += 1

                self.log(
                    f"RECEIVE in order packet={seq}; "
                    f"cumulative ACK={seq}"
                )

            else:

                self.duplicates += 1

                self.log(
                    f"OUT OF ORDER/DUPLICATE packet={seq}; "
                    f"ACK={self.recv_base - 1}"
                )

            self.send_ack(self.recv_base - 1)

    # --------------------------------------------------------
    # ACK transmission
    # --------------------------------------------------------

    def send_ack(self, ack_seq):

        if self.rng.random() < self.pa:

            self.ack_errors += 1

            self.log(
                f"ACK LOST ack={ack_seq}"
            )

        else:

            self.event(
                self.time + ONE_WAY_MS,
                "ACK",
                ack_seq
            )

    # --------------------------------------------------------
    # ACK arrival at sender
    # --------------------------------------------------------

    def got_ack(self, seq):

        if self.protocol == "SR":

            if (
                seq < self.base
                or seq >= self.next_seq
            ):
                return

            if seq not in self.acked:

                self.acked.add(seq)

                self.timer_id[seq] = (
                    self.timer_id.get(seq, 0) + 1
                )

                self.log(
                    f"ACK RECEIVED packet={seq}"
                )

            while self.base in self.acked:

                self.acked.remove(self.base)

                self.base += 1

        else:

            if (
                seq < self.base
                or seq >= self.next_seq
            ):
                return

            old = self.base

            self.base = seq + 1

            self.log(
                f"CUMULATIVE ACK={seq}; "
                f"base {old}->{self.base}"
            )

            self.arm_gbn_timer()

        self.start_send()

    # --------------------------------------------------------
    # Selective Repeat timeout
    # --------------------------------------------------------

    def sr_timeout(self, seq, token):

        if (
            token != self.timer_id.get(seq, 0)
            or seq < self.base
            or seq in self.acked
        ):
            return

        self.log(
            f"SR TIMEOUT packet={seq}; "
            "selective retransmit"
        )

        self.enqueue(seq)

        self.start_send()

    # --------------------------------------------------------
    # Go-Back-N timeout
    # --------------------------------------------------------

    def gbn_timeout(self, token, base_at_schedule):

        if (
            token != self.gbn_timer_id
            or base_at_schedule != self.base
        ):
            return

        if self.base >= self.next_seq:
            return

        self.log(
            f"GBN TIMEOUT base={self.base}; "
            "retransmit outstanding window"
        )

        for seq in range(self.base, self.next_seq):

            self.enqueue(seq)

        self.gbn_timer_id += 1

        self.start_send()

    # --------------------------------------------------------
    # Main event loop
    # --------------------------------------------------------

    def run(self):

        self.start_send()

        events = 0

        while self.base < self.count:

            if not self.q:

                raise RuntimeError(
                    "Event queue empty before "
                    "all packets were acknowledged"
                )

            self.time, _, kind, args = heapq.heappop(
                self.q
            )

            events += 1

            if events > self.max_events:

                raise RuntimeError(
                    "Too many events: check "
                    "protocol progress"
                )

            if kind == "TX_END":

                self.tx_end(*args)

            elif kind == "RECEIVE":

                self.receive(*args)

            elif kind == "ACK":

                self.got_ack(*args)

            elif kind == "SR_TIMEOUT":

                self.sr_timeout(*args)

            elif kind == "GBN_TIMEOUT":

                self.gbn_timeout(*args)

        self.log("ALL PACKETS ACKNOWLEDGED")

        return {
            "protocol": self.protocol,
            "ws": self.ws,
            "wr": self.wr,
            "pd": self.pd,
            "pa": self.pa,
            "timeout_ms": self.timeout,
            "n": self.count,
            "completion_ms": self.time,
            "efficiency": (
                self.count * PACKET_MS / self.time
            ),
            "attempts_per_packet": (
                self.attempts / self.count
            ),
            "attempts": self.attempts,
            "data_errors": self.data_errors,
            "ack_errors": self.ack_errors,
            "duplicates": self.duplicates
        }


# ============================================================
# EXPERIMENT HELPERS
# ============================================================

def simulate(
    protocol,
    ws,
    wr,
    pd=0,
    pa=0,
    timeout=DEFAULT_TIMEOUT,
    count=LONG_RUN,
    seed=SEEDS[0],
    trace=False
):

    sim = ARQ(
        protocol, ws, wr, pd, pa,
        timeout, count, seed, trace
    )

    return sim.run(), sim.trace


def protocols(ws=16):

    return [
        ("SW", 1, 1),
        ("GBN", ws, 1)
    ] + [
        ("SR", ws, wr)
        for wr in RECEIVE_WINDOWS
    ]


def label(protocol, wr):

    if protocol == "SW":
        return "Stop-and-Wait"

    if protocol == "GBN":
        return "Go-Back-N"

    return f"Selective Repeat (Wr={wr})"


def write_csv(filename, rows):

    if not rows:
        return

    with (
        OUT / filename
    ).open("w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)


def summarize(records, tag):

    groups = {}

    for row in records:

        key = (
            row["protocol"],
            row["wr"],
            row["ws"],
            row["pd"],
            row["pa"],
            row["timeout_ms"],
            row["n"]
        )

        groups.setdefault(key, []).append(row)

    summary = []

    for key, values in groups.items():

        first = {
            k: values[0][k]
            for k in (
                "protocol", "ws", "wr",
                "pd", "pa", "timeout_ms", "n"
            )
        }

        first["seeds"] = len(values)

        for metric in (
            "efficiency",
            "attempts_per_packet",
            "completion_ms"
        ):

            first[metric] = sum(
                v[metric] for v in values
            ) / len(values)

        summary.append(first)

    write_csv(tag + "_raw.csv", records)

    write_csv(tag + "_mean.csv", summary)

    return summary


def run_batch(tag, params):

    records = []

    for protocol, ws, wr, pd, pa, timeout, n in params:

        for seed in SEEDS:

            result, _ = simulate(
                protocol,
                ws,
                wr,
                pd,
                pa,
                timeout,
                n,
                seed
            )

            result["seed"] = seed

            records.append(result)

    return summarize(records, tag)


# ============================================================
# GRAPH GENERATION
# ============================================================

def plot(
    rows,
    xfield,
    metric,
    filename,
    title,
    xlabel,
    ylabel,
    reference=False
):

    series = {}

    for row in rows:

        key = (
            row["protocol"],
            row["wr"]
        )

        series.setdefault(key, []).append(row)

    plt.figure(figsize=(9, 5.5))

    for (proto, wr), group in sorted(series.items()):

        group.sort(
            key=lambda r: r[xfield]
        )

        xs = [
            r[xfield] for r in group
        ]

        ys = [
            r[metric] for r in group
        ]

        name = label(proto, wr)

        if reference and proto == "SW":

            plt.axhline(
                sum(ys) / len(ys),
                linestyle="--",
                label=name
            )

        else:

            plt.scatter(
                xs,
                ys,
                s=35,
                label=name
            )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=180
    )

    plt.close()


# ============================================================
# PART 1 - SIMULATION VERIFICATION
# ============================================================

def part1():

    filename = OUT / "part1_verification_traces.txt"

    with filename.open("w", encoding="utf-8") as f:

        for pd, pa in (
            (0, 0),
            (0.5, 0.5)
        ):

            for proto, ws, wr in protocols(4):

                result, trace = simulate(
                    proto,
                    ws,
                    wr,
                    pd,
                    pa,
                    count=12,
                    seed=SEEDS[0],
                    trace=True
                )

                f.write(
                    f"\n===== {label(proto, wr)} "
                    f"Ws={ws} Wr={wr} "
                    f"pd={pd} pa={pa} =====\n"
                )

                f.write(
                    "\n".join(trace) + "\n"
                )

                f.write(
                    f"Completed in "
                    f"{result['completion_ms']:.3f} ms; "
                    f"attempts={result['attempts']}\n"
                )


# ============================================================
# PART 2 - ERROR-FREE EFFICIENCY
# ============================================================

def part2():

    jobs = []

    for ws in WINDOWS:

        jobs.append(
            ("GBN", ws, 1, 0, 0, 15, LONG_RUN)
        )

        for wr in RECEIVE_WINDOWS:

            jobs.append(
                ("SR", ws, wr, 0, 0, 15, LONG_RUN)
            )

    jobs.append(
        ("SW", 1, 1, 0, 0, 15, LONG_RUN)
    )

    rows = run_batch("part2", jobs)

    plot(
        rows,
        "ws",
        "efficiency",
        "part2_efficiency_vs_ws.png",
        "Part 2: Error-free link efficiency",
        "Send window Ws",
        "Link efficiency",
        reference=True
    )


# ============================================================
# PART 3 - DATA PACKET ERRORS
# ============================================================

def part3():

    # Part 3(a)
    # Efficiency vs packet error probability

    jobs_a = []

    for pd in ERROR_PROBS:

        for proto, ws, wr in protocols(16):

            jobs_a.append(
                (proto, ws, wr, pd, 0, 15, LONG_RUN)
            )

    rows_a = run_batch("part3a", jobs_a)

    plot(
        rows_a,
        "pd",
        "efficiency",
        "part3a_efficiency_vs_pd.png",
        "Part 3(a): Packet errors, Ws=16",
        "Data-packet error probability pd",
        "Link efficiency"
    )

    # Part 3(b)
    # Efficiency and attempts vs send window

    jobs_b = []

    for ws in WINDOWS:

        jobs_b.append(
            ("GBN", ws, 1, .05, 0, 15, LONG_RUN)
        )

        for wr in RECEIVE_WINDOWS:

            jobs_b.append(
                ("SR", ws, wr, .05, 0, 15, LONG_RUN)
            )

    jobs_b.append(
        ("SW", 1, 1, .05, 0, 15, LONG_RUN)
    )

    rows_b = run_batch("part3b", jobs_b)

    plot(
        rows_b,
        "ws",
        "efficiency",
        "part3b_efficiency_vs_ws.png",
        "Part 3(b): Efficiency, pd=0.05",
        "Send window Ws",
        "Link efficiency",
        reference=True
    )

    plot(
        rows_b,
        "ws",
        "attempts_per_packet",
        "part3b_attempts_vs_ws.png",
        "Part 3(b): Transmission attempts",
        "Send window Ws",
        "Attempts per delivered packet",
        reference=True
    )


# ============================================================
# PART 4 - DATA AND ACK ERRORS
# ============================================================

def part4():

    jobs = []

    for pa in ACK_PROBS:

        for proto, ws, wr in protocols(16):

            jobs.append(
                (
                    proto, ws, wr,
                    .05, pa, 15, LONG_RUN
                )
            )

    rows = run_batch("part4", jobs)

    plot(
        rows,
        "pa",
        "efficiency",
        "part4_efficiency_vs_pa.png",
        "Part 4: Efficiency, pd=0.05",
        "ACK error probability pa",
        "Link efficiency"
    )

    plot(
        rows,
        "pa",
        "attempts_per_packet",
        "part4_attempts_vs_pa.png",
        "Part 4: Transmission attempts",
        "ACK error probability pa",
        "Attempts per delivered packet"
    )


# ============================================================
# PART 5 - RETRANSMISSION TIMEOUT
# ============================================================

def part5():

    jobs = []

    for timeout in TIMEOUTS:

        for proto, ws, wr in protocols(16):

            jobs.append(
                (
                    proto,
                    ws,
                    wr,
                    .05,
                    .05,
                    timeout,
                    LONG_RUN
                )
            )

    rows = run_batch("part5", jobs)

    plot(
        rows,
        "timeout_ms",
        "efficiency",
        "part5_efficiency_vs_timeout.png",
        "Part 5: Efficiency vs timeout",
        "Retransmission timeout (ms)",
        "Link efficiency"
    )

    plot(
        rows,
        "timeout_ms",
        "attempts_per_packet",
        "part5_attempts_vs_timeout.png",
        "Part 5: Attempts vs timeout",
        "Retransmission timeout (ms)",
        "Attempts per delivered packet"
    )


# ============================================================
# FINITE TRANSFER HELPER
# ============================================================

def finite_case(tag, n, pd, pa):

    jobs = []

    for ws in WINDOWS:

        jobs.append(
            ("GBN", ws, 1, pd, pa, 15, n)
        )

        for wr in RECEIVE_WINDOWS:

            jobs.append(
                ("SR", ws, wr, pd, pa, 15, n)
            )

    jobs.append(
        ("SW", 1, 1, pd, pa, 15, n)
    )

    rows = run_batch(tag, jobs)

    plot(
        rows,
        "ws",
        "completion_ms",
        tag + "_completion_vs_ws.png",
        f"{tag}: N={n}, pd={pd}, pa={pa}",
        "Send window Ws",
        "Mean transfer completion time (ms)",
        reference=True
    )


# ============================================================
# PART 6 - FINITE PACKET TRANSFER
# ============================================================

def part6():

    finite_case(
        "part6_no_errors",
        100,
        0,
        0
    )

    finite_case(
        "part6_data_errors",
        100,
        .05,
        0
    )

    finite_case(
        "part6_data_ack_errors",
        100,
        .05,
        .05
    )


# ============================================================
# PART 7 - EFFECT OF TRANSFER SIZE
# ============================================================

def part7():

    # Smaller transfer size

    finite_case(
        "part7_n20",
        20,
        .05,
        .05
    )

    # Larger transfer size

    finite_case(
        "part7_n500",
        500,
        .05,
        .05
    )


# ============================================================
# SIMULATION SANITY CHECKS
# ============================================================

def checks():

    # Error-free S&W expected efficiency:
    # 1 ms / (1 ms + 10 ms) = 1/11

    sw, _ = simulate(
        "SW",
        1,
        1,
        count=1200,
        pd=0,
        pa=0
    )

    assert abs(
        sw["efficiency"] - 1 / 11
    ) < .001

    for proto, ws, wr in protocols(16):

        result, _ = simulate(
            proto,
            ws,
            wr,
            count=1200,
            pd=0,
            pa=0
        )

        assert result["attempts"] == 1200

        assert result["efficiency"] <= 1 + 1e-9

        assert result["completion_ms"] > 0

    print(
        "Sanity checks passed: "
        "S&W, GBN and SR under zero errors."
    )


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

def main():

    checks()

    print("Part 1...")
    part1()

    print("Part 2...")
    part2()

    print("Part 3...")
    part3()

    print("Part 4...")
    part4()

    print("Part 5...")
    part5()

    print("Part 6...")
    part6()

    print("Part 7...")
    part7()

    print()
    print("All experiments completed.")
    print(
        f"Results saved to: {OUT.resolve()}"
    )


if __name__ == "__main__":
    main()