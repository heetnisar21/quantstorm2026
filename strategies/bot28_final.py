# Name: Heet Nisar
# College: Dwarkadas J. Sanghvi College of Engineering
# Roll Number: 60005240044

"""
heet_bot_final.py — Divided Oracle entry (v7)
================================================

v1-v5: see the earlier docstring history (value-estimation double-
counting bug in v2-v4, fixed in v5; exact SUBSTITUTE put-expectation;
margin-free final-turn EV comparison). Empirically confirmed here: v5's
round-by-round contract PnL vs adaptive_bidder is +1.04/contract in
round 5 and positive in every round 3-5 (only rounds 1-2 dip slightly
negative, -0.14/-0.08) — the round-5 collapse described in earlier
versions' notes is gone, not just theoretically fixed.

v6 — BUG FOUND BY RE-READING _best_anchor, NOT BY TESTING FIRST:
`_best_anchor` filtered `r < obs.round`, meant to stop `quote()` (called
before this round's anchor exists) from ever seeing it. But that same
filter also ran inside `respond()`, where it excludes THIS round's own
anchor once turn 2 has already written it. Consequence: the maker's
clean opening-quote read, captured at turn 2, was used for exactly one
turn and then silently dropped for turns 4 and 6 of the SAME round --
falling back to whatever was known before the round started, discarding
the round's best available signal right when it mattered most.

Fix: `_best_anchor` now takes `r <= obs.round`. Still safe for `quote()`
-- as Maker we quote before any anchor for this round can exist, so
there is nothing to leak backwards in time.

Measured (engine.play_match, 6 seeds x 150 mirrored deals per matchup,
95% CI on the mean):

    matchup              v5              v6            delta
    vs Rational        7.89 +/- 0.44   8.91 +/- 0.34   +1.02
    vs AdaptiveBidder   3.90 +/- 0.44   4.48 +/- 0.44   +0.59
    vs NaiveEV          8.21 +/- 0.45   9.23 +/- 0.50   +1.02

Consistent, positive, and outside the CI band against all three
baselines -- not noise. Self-play mirrored match: 0.00 exactly, both
in-process and under `--isolate` (sanity check that the fix didn't
introduce an asymmetry).

v7 — TRICK_ROOM / STEALTH_ROCK RE-CALIBRATION, MEASURED NOT GUESSED:
v6's own docstring flagged this and left it alone: the reused
POWER_VALUES for these two were measured on a bot that gated forcing
behind a fixed margin, and v5/v6 removed that margin (pure EV
comparison, see earlier notes). A bot that forces more often should
value forced-fill shift powers more, and the old numbers predate that
change.

Rather than re-guess a constant, swept a multiplier on both powers'
values, 0.8x-2.5x, then narrowed to 1.1x-1.5x at 10 seeds x 200 deals
per point:

    multiplier   vs Rational   vs AdaptiveBidder   vs NaiveEV   avg
    1.0x            8.78            4.41             8.92       7.37
    1.1x            8.70            5.26             8.84       7.60
    1.2x            8.65            5.22             8.78       7.55
    1.3x            8.61            5.16             8.75       7.51
    1.4x            8.59            5.13             8.72       7.48
    1.5x            8.54            5.05             8.70       7.43

Against the two non-bidding baselines the multiplier barely matters and
trends slightly negative -- a single uncontested TE bid wins the power
regardless of size, so overpaying is pure waste. Against the one
baseline that actually bids, going from 1.0x to 1.1x is worth +0.85
ticks/deal, essentially flat after that. The peak position inside
1.1x-1.5x moves with the seed set (noise, not signal) but the shape --
a real, non-noise step up from 1.0x, then a plateau -- does not.
Shipped at 1.15x: inside the plateau, not chasing the specific peak of
any one run.

STILL OPEN, NOT ADDRESSED HERE: the anchor read (this bot's and any
opponent's) assumes an honestly-centred Maker quote with no other
information folded in; a Maker holding its own FORESIGHT or reading its
own earlier-round anchor will bias the read in a way this model doesn't
correct for. TRANSFORM's fire/bid rule is still the flat "swap iff
k_mine == 0" inherited from the baselines (note: k_mine is always a
multiple of REVEAL_PER_ROUND=4, hence always even, so this threshold
and a threshold of 1 are exactly the same rule) -- a full value-of-
swap calculation conditioned on remaining deal length was not derived
here for lack of time before the deadline; see the reasoning in
starter_bot.py and adaptive_bidder.py's DENIAL_WEIGHT note for the
open question if extending this further.

Validated before submission:
  python backtester.py --validate strategies/heet_bot_final.py   -> ACCEPTED
  python backtester.py --bot1 ... --bot2 ... --isolate            -> matches
      direct-mode PnL, no warnings, no forfeits, avg 0.014ms/call
      (design target 2ms, hard limit 50ms).
"""

import math
import random

# Calibrated tick value of each power, per round. FORESIGHT, SUBSTITUTE
# and TRANSFORM are exactly strategies/adaptive_bidder.py's measured
# values (a provided baseline; its constants are explicitly reusable,
# RULEBOOK.md sec 12). TRICK_ROOM and STEALTH_ROCK are that baseline's
# values scaled 1.15x -- see the v7 note above for why and the measured
# sweep that supports it.
POWER_VALUES = {
    "FORESIGHT":    {1: 0.76, 2: 1.16, 3: 1.48, 4: 1.97, 5: 2.02},
    "TRICK_ROOM":   {1: 1.311, 2: 0.0, 3: 0.0, 4: 0.69, 5: 0.598},
    "SUBSTITUTE":   {1: 1.46, 2: 1.15, 3: 0.95, 4: 0.57, 5: 0.29},
    "STEALTH_ROCK": {1: 1.736, 2: 0.862, 3: 0.862, 4: 0.862, 5: 0.0},
    "TRANSFORM":    {1: 1.58, 2: 1.24, 3: 1.31, 4: 0.00, 5: 0.00},
}

SHADE = 0.60                  # first-price shade, see adaptive_bidder.py
FLAT_THRESHOLD = 1            # |k_mine| this small -> hand is worth swapping away
SUBSTITUTE_CAP = -2.0         # RULEBOOK.md sec 5: SUBSTITUTE's exact loss-cap magnitude
# Shift magnitudes per RULEBOOK.md sec 5 / sec 15. Not importable from
# engine (engine.shift_sources is off the allowlist), reimplemented here.
SHIFT_POWERS = {"TRICK_ROOM": 3, "STEALTH_ROCK": 2}


class Bot:
    name = "DividedOracle"

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def reset(self, seat, config, seed) -> None:
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)
        self._opp_anchor = {}     # round -> opponent whole-hand-sum estimate (their opening quote)
        self._foresight_n = 0     # size of the best FORESIGHT sample seen this deal
        self._foresight_sum = 0   # its sum

    # ------------------------------------------------------------------
    # value estimation
    # ------------------------------------------------------------------

    def _update_foresight(self, obs):
        if obs.foresight:
            n = len(obs.foresight)
            if n >= self._foresight_n:
                self._foresight_n = n
                self._foresight_sum = sum(obs.foresight)

    def _best_anchor(self, obs):
        """Most recent opponent-whole-hand-sum read from an opening
        quote, THIS round included once turn 2 has set it. Safe to
        include the current round here: quote() calls this before any
        anchor for the round can exist (Maker always quotes first), so
        there is nothing to look ahead of."""
        cand = [r for r in self._opp_anchor if r <= obs.round]
        if not cand:
            return None, None
        r = max(cand)
        return self._opp_anchor[r], r

    def _estimate_S(self, obs, live_quote=None):
        """Inverse-variance combination of every REAL signal about the
        opponent's whole 20-coin hand. The flat (mean 0, var N_PRIVATE)
        prior is used ONLY when no real signal exists yet -- blending it
        in alongside real signals would double-count the mean-zero
        assumption each signal's own variance formula already makes.

        Returns (S_hat, S_var). S_var includes both remaining
        uncertainty about the opponent's hand AND the structural,
        signal-proof uncertainty about our own still-unrevealed coins
        (RULEBOOK.md sec 2)."""
        n_private = self.config.N_PRIVATE
        per_round = self.config.REVEAL_PER_ROUND

        self._update_foresight(obs)

        components = []  # (mean, variance) of k_theirs, real signals only

        if self._foresight_n > 0:
            var = max(1.0, n_private - self._foresight_n)
            components.append((self._foresight_sum, var))

        anchor, anchor_r = self._best_anchor(obs)
        if anchor is not None:
            var = max(1.0, n_private - per_round * anchor_r)
            components.append((anchor, var))

        # Only ever trust the live quote as a value signal on turn 2 --
        # the one turn guaranteed to be the Maker's untouched opening
        # (RULEBOOK.md sec 9: later ranges are contaminated by both sides).
        if live_quote is not None and not obs.is_maker:
            live_mid = (live_quote[0] + live_quote[1]) / 2.0
            var = max(1.0, n_private - per_round * obs.round)
            components.append((live_mid, var))

        if components:
            w_sum = sum(1.0 / v for _, v in components)
            k_theirs_hat = sum(m / v for m, v in components) / w_sum
            opp_var = 1.0 / w_sum
        else:
            k_theirs_hat = 0.0
            opp_var = float(n_private)

        my_unseen_var = max(0, n_private - len(obs.my_revealed))
        S_var = opp_var + my_unseen_var

        return obs.k_mine + k_theirs_hat, S_var

    @staticmethod
    def _put_ev(strike, mean, sd):
        """E[max(0, strike - X)] for X ~ Normal(mean, sd^2). Closed-form
        expected shortfall below `strike`, used to price SUBSTITUTE's
        exact loss-cap-and-refund mechanic instead of guessing a
        constant."""
        if sd <= 1e-9:
            return max(0.0, strike - mean)
        z = (strike - mean) / sd
        Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return (strike - mean) * Phi + sd * phi

    def _true_ev(self, raw_edge, se, substitute):
        """Raw edge, adjusted for SUBSTITUTE's exact loss-cap-and-refund
        mechanic when held (RULEBOOK.md sec 5, magnitude 2). No-op
        otherwise."""
        if not substitute:
            return raw_edge
        return raw_edge + self._put_ev(SUBSTITUTE_CAP, raw_edge, se)

    # ------------------------------------------------------------------
    # auction
    # ------------------------------------------------------------------

    def _transform_value(self, obs):
        swap = POWER_VALUES["TRANSFORM"].get(obs.round, 0.0)
        if abs(obs.k_mine) <= FLAT_THRESHOLD:
            return swap
        # Decisive hand: only denial would be worth anything, and the
        # measured value of that (on the old spec) was at or below zero
        # against every opponent tested -- see adaptive_bidder.py's
        # DENIAL_WEIGHT note. Left at zero rather than re-guessed here.
        return 0.0

    def bid(self, obs, offered: list) -> dict:
        if not offered or obs.te_mine <= 0:
            return {}

        values = {}
        for name in offered:
            v = self._transform_value(obs) if name == "TRANSFORM" else \
                POWER_VALUES.get(name, {}).get(obs.round, 0.5)
            if v > 0:
                values[name] = v

        if not values:
            return {}

        fair_te = {n: v / self.config.TE_SALVAGE for n, v in values.items()}
        raw_bid = {n: int(f * SHADE) for n, f in fair_te.items()}

        total = sum(raw_bid.values())
        if total <= obs.te_mine:
            return {n: b for n, b in raw_bid.items() if b > 0}

        # Budget too tight for everything offered: fund the highest-value
        # power(s) first instead of losing the whole vector to a clamp.
        # (SLOTS_PER_ROUND == 1 in the current rules, so `offered` is
        # normally a single power -- this branch is defensive, not dead
        # weight, in case that ever changes.)
        budget = obs.te_mine
        out = {}
        for n in sorted(values, key=lambda k: values[k], reverse=True):
            amt = min(raw_bid[n], budget)
            if amt > 0:
                out[n] = amt
                budget -= amt
        return out

    # ------------------------------------------------------------------
    # negotiation
    # ------------------------------------------------------------------

    def quote(self, obs) -> tuple:
        # Honest centering at the minimum legal width is provably
        # optimal here: the maker-obligation transfer (RULEBOOK.md sec
        # 7.2) nets to zero EV under honest centering for ANY width,
        # while WIDTH_PREMIUM charges for width above the floor
        # unconditionally -- so the floor dominates. Not a heuristic.
        S_hat, _ = self._estimate_S(obs)
        v = round(S_hat)
        cap = obs.final_cap
        lo = v - cap // 2
        return (lo, lo + cap)

    def respond(self, obs, quote: tuple, turn: int):
        # Only the very first response of the round reads a clean, honest
        # opening quote; later ranges are contaminated by both sides.
        if turn == 2:
            self._opp_anchor[obs.round] = (quote[0] + quote[1]) / 2.0

        bid_p, ask_p = quote
        live_quote = quote if turn == 2 else None
        S_hat, S_var = self._estimate_S(obs, live_quote)
        se = S_var ** 0.5

        substitute = "SUBSTITUTE" in obs.powers_mine
        edge_buy = S_hat - ask_p
        edge_sell = bid_p - S_hat
        true_buy = self._true_ev(edge_buy, se, substitute)
        true_sell = self._true_ev(edge_sell, se, substitute)

        if turn == obs.n_turns:
            return self._final_turn(obs, bid_p, ask_p, S_hat, se, edge_buy, edge_sell,
                                     substitute)

        # With an unbiased S_hat, "true edge > 0" is already EV-maximal
        # under this game's pure-PnL-sum scoring -- no confidence bar
        # needed.
        if true_buy > 0 and true_buy >= true_sell:
            return "ACCEPT_BUY"
        if true_sell > 0:
            return "ACCEPT_SELL"

        # RULEBOOK.md sec 6: max_width = min(ask-bid, max(final_cap,
        # (ask-bid) - MIN_REDUCTION)). Never shrink past the round's floor.
        width = ask_p - bid_p
        max_width = min(width, max(obs.final_cap, width - self.config.MIN_REDUCTION))
        center = max(bid_p, min(round(S_hat), ask_p - max_width))
        return ("COUNTER", center, center + max_width)

    def _final_turn(self, obs, bid_p, ask_p, S_hat, se, edge_buy, edge_sell, substitute):
        """Countering here is a forced fill: my range fixes the midpoint,
        I go short (last quoter sells, RULEBOOK.md sec 6/15), I pay the
        forcing fee. Choose the legal range that maximises that price,
        then take whichever of force/accept has the higher true EV --
        no margin: with an unbiased estimator, taking the larger of two
        EV estimates is already the policy that maximises expected total
        PnL over repeated trials, and this game scores exactly that
        (RULEBOOK.md sec 1/13)."""
        width = ask_p - bid_p
        max_width = min(width, max(obs.final_cap, width - self.config.MIN_REDUCTION))
        f_bid = ask_p - max_width
        f_ask = ask_p

        shift = 0
        for power, mag in SHIFT_POWERS.items():
            if power in obs.powers_mine:
                shift += mag
            if power in obs.powers_theirs:
                shift -= mag

        forced_price = (f_bid + f_ask) // 2 + shift
        raw_force_contract = forced_price - S_hat
        true_force_contract = self._true_ev(raw_force_contract, se, substitute)
        ev_force = true_force_contract - self.config.FORCED_FILL_FEE

        true_buy = self._true_ev(edge_buy, se, substitute)
        true_sell = self._true_ev(edge_sell, se, substitute)
        best_accept = max(true_buy, true_sell)

        if ev_force > best_accept:
            return ("COUNTER", f_bid, f_ask)

        if true_buy >= true_sell:
            return "ACCEPT_BUY"
        return "ACCEPT_SELL"

    def use_transform(self, obs) -> bool:
        return abs(obs.k_mine) <= FLAT_THRESHOLD