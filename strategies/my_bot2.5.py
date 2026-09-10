# Name: Heet Nisar
# College: Dwarkadas J. Sanghvi College of Engineering
# Roll Number: 60005240044

"""
heet_bot.py — Divided Oracle entry (v2)
========================================

v1 baseline result: avg win 68%, with realized PnL negative in rounds
1-2 and positive (peaking) in rounds 3-5. That shape is diagnostic,
not random noise: rounds 1-2 are exactly where the opponent-sum
estimate is weakest (fewest FORESIGHT samples, no anchor yet or a
high-variance one), so a rule that trades on any edge > 0 is partly
trading on estimation noise rather than real mispricing in those
rounds. By round 3-5, enough signal has accumulated that the same
edge > 0 rule is mostly catching real edge -- hence the improving,
then peak-at-5, curve.

v2 makes three changes, all pulled out of the noise model this bot
already builds for itself -- not tuned against any particular
opponent, since chasing a specific bot's behavior is exactly what
produces a curve that's great late (well-sampled) and shaky early
(under-sampled, easy to overfit blindly):

1. SHRINKAGE PRIOR. _estimate_S previously combined FORESIGHT / anchor
   / live-quote signals only when at least one existed, and hard-fell
   back to k_theirs_hat = 0 otherwise. That's a discontinuity: the
   instant an anchor is first recorded, the estimate can jump, because
   nothing was regularizing it beforehand. Now a flat prior (mean 0,
   var 20 -- the full unconditional variance of an unknown 20-coin
   hand under the stated ±1-per-coin model) is always one of the
   inverse-variance components, so thin early information is shrunk
   toward 0 instead of taken near face value, and the estimate moves
   smoothly as real signal arrives. (At the very first quote, with no
   other components, this reduces to the old k_theirs_hat = 0 exactly
   -- so round-1 opening behavior is unchanged; only what happens
   after that differs.)

2. CONFIDENCE-SCALED TRADING THRESHOLD. Market-taking (ACCEPT_BUY /
   ACCEPT_SELL) previously required only edge > thresh, a flat
   constant. The estimator already carries its own posterior variance
   (the reciprocal of the summed inverse-variances), so its standard
   error is known at decision time for free. respond() now requires
   edge to clear thresh PLUS a multiple of that standard error: the
   edge has to be distinguishable from estimation noise, not merely
   positive. Because SE shrinks automatically as FORESIGHT/anchor/live
   signal accumulate over the course of a deal, this bar comes out
   strict in rounds 1-2 and relaxes by rounds 3-5 on its own -- it
   reproduces the "hold back early, press late" shape v1 found
   empirically, except earned from the noise model instead of hard-
   coded to round number (so it won't silently mis-generalize if
   final_cap, round count, or TE budgets change).

3. The same standard error feeds the final-turn force-vs-accept
   margin (previously a flat FORCE_MARGIN), so the forcing turn is
   held to the same risk standard as an ordinary trade -- with the
   original constant kept as a floor so it's never looser than what
   v1 already established as safe.

Left unchanged: the auction's calibrated per-round power values and
0.60 first-price shade (explicitly sanctioned reuse per RULEBOOK.md
sec 12: baseline code and its constants are not evidence of copying),
the forcing-turn mechanics themselves, and TRANSFORM denial staying
zeroed out (v1 measured it at or below zero and re-guessing that
without a live backtester risks reintroducing a losing feature).

Two tunables worth sweeping in your own backtester rather than taking
on faith: Z_ACCEPT and Z_FORCE below (0.5 SE is a reasonable, not
sacred, starting point -- I don't have access to your simulator or
RULEBOOK.md to verify this against realized variance directly).
"""

import random

# Calibrated tick value of each power, per round -- lifted from
# strategies/adaptive_bidder.py (a provided baseline; its constants are
# explicitly reusable, see RULEBOOK.md sec 12).
POWER_VALUES = {
    "FORESIGHT":    {1: 0.76, 2: 1.16, 3: 1.48, 4: 1.97, 5: 2.02},
    "TRICK_ROOM":   {1: 1.14, 2: 0.00, 3: 0.00, 4: 0.60, 5: 0.52},
    "SUBSTITUTE":   {1: 1.46, 2: 1.15, 3: 0.95, 4: 0.57, 5: 0.29},
    "STEALTH_ROCK": {1: 1.51, 2: 0.75, 3: 0.75, 4: 0.75, 5: 0.00},
    "TRANSFORM":    {1: 1.58, 2: 1.24, 3: 1.31, 4: 0.00, 5: 0.00},
}

SHADE = 0.60                 # first-price shade, see adaptive_bidder.py
FLAT_THRESHOLD = 1            # |k_mine| this small -> hand is worth swapping away
SHIFT_POWERS = {"TRICK_ROOM": 3, "STEALTH_ROCK": 2}

PRIOR_VAR = 20.0              # unconditional variance of an unknown 20-coin ±1 hand
Z_ACCEPT = 0.50                # required edge, in standard errors, to market-take
Z_FORCE = 0.50                 # same standard applied to the forcing-turn decision
FORCE_MARGIN_FLOOR = 0.3      # v1's constant, kept as a floor under the SE-based margin


class Bot:
    name = "DividedOracle"

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def reset(self, seat, config, seed) -> None:
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)
        self._opp_anchor = {}     # round -> opponent revealed-sum estimate (their opening quote)
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
        """Most recent opponent-revealed-sum read from an earlier round's
        opening quote (round < obs.round only -- this round's is not in
        yet unless passed explicitly via a live quote)."""
        earlier = [r for r in self._opp_anchor if r < obs.round]
        if not earlier:
            return None, None
        r = max(earlier)
        return self._opp_anchor[r], r

    def _estimate_S(self, obs, live_quote=None):
        """Inverse-variance combination of every signal about the
        opponent's hand (always including a flat shrinkage prior),
        added to our own exactly-known revealed sum.

        Returns (S_hat, S_var): S_var is the posterior variance of the
        combined estimate, used downstream to size how much edge is
        required before actually trading on it."""
        self._update_foresight(obs)

        # Always-present shrinkage prior -- keeps thin early information
        # from being taken at face value (see module docstring, item 1).
        components = [(0.0, PRIOR_VAR)]  # (mean, variance) of k_theirs

        if self._foresight_n > 0:
            var = max(1.0, 20 - self._foresight_n)
            components.append((self._foresight_sum, var))

        anchor, anchor_r = self._best_anchor(obs)
        if anchor is not None:
            var = max(1.0, 20 - 4 * anchor_r)
            components.append((anchor, var))

        if live_quote is not None and not obs.is_maker:
            live_mid = (live_quote[0] + live_quote[1]) / 2.0
            var = max(1.0, 20 - 4 * obs.round)
            components.append((live_mid, var))

        w_sum = sum(1.0 / v for _, v in components)
        k_theirs_hat = sum(m / v for m, v in components) / w_sum
        S_var = 1.0 / w_sum

        return obs.k_mine + k_theirs_hat, S_var

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
        S_hat, S_var = self._estimate_S(obs, quote)
        se = S_var ** 0.5

        substitute = "SUBSTITUTE" in obs.powers_mine
        base_thresh = -1.0 if substitute else 0.0
        # Required edge now scales with how noisy our own estimate still
        # is, not just a flat constant -- see module docstring, item 2.
        thresh = base_thresh + Z_ACCEPT * se

        edge_buy = S_hat - ask_p
        edge_sell = bid_p - S_hat

        if turn == obs.n_turns:
            return self._final_turn(obs, bid_p, ask_p, S_hat, se, edge_buy, edge_sell,
                                     substitute)

        if edge_buy > thresh and edge_buy >= edge_sell:
            return "ACCEPT_BUY"
        if edge_sell > thresh:
            return "ACCEPT_SELL"

        w = max(0, (ask_p - bid_p) - self.config.MIN_REDUCTION)
        center = max(bid_p, min(round(S_hat), ask_p - w))
        return ("COUNTER", center, center + w)

    def _final_turn(self, obs, bid_p, ask_p, S_hat, se, edge_buy, edge_sell, substitute):
        """Countering here is a forced fill: my range fixes the midpoint,
        I go short, I pay the forcing fee. Choose the legal range that
        maximises that price, then compare its EV to simply accepting --
        holding the forcing turn to the same noise-adjusted standard as
        any other trade (module docstring, item 3)."""
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
        ev_force = (forced_price - S_hat) - self.config.FORCED_FILL_FEE

        ev_buy, ev_sell = edge_buy, edge_sell
        if substitute:
            ev_buy = max(ev_buy, -2.0)
            ev_sell = max(ev_sell, -2.0)
            ev_force = max(ev_force, -2.0)

        best_accept = max(ev_buy, ev_sell)
        force_margin = max(FORCE_MARGIN_FLOOR, Z_FORCE * se)
        if ev_force > best_accept + force_margin:
            return ("COUNTER", f_bid, f_ask)

        if ev_buy >= ev_sell:
            return "ACCEPT_BUY"
        return "ACCEPT_SELL"

    def use_transform(self, obs) -> bool:
        return abs(obs.k_mine) <= FLAT_THRESHOLD