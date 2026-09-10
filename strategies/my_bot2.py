# Name: Heet Nisar
# College: Dwarkadas J. Sanghvi College of Engineering
# Roll Number: 60005240044

"""
heet_bot.py — Divided Oracle entry
====================================

Three ideas on top of the provided baselines, each derived from the spec
rather than tuned against any particular opponent:

1. VALUE ESTIMATION is an inverse-variance combination of every signal
   about the opponent's hand, not a fixed 50/50 blend:
     - FORESIGHT: an exact partial sum of their revealed coins. The part
       it doesn't sample has mean 0, so the sum is already an unbiased
       estimator of their whole 20-coin hand, with variance
       (20 - n_sampled).
     - The Maker's OPENING quote midpoint, read only on the first
       response of a round (later ranges are contaminated by both
       sides, per RULEBOOK.md sec 9), is itself the maker's own estimate
       of their revealed sum under honest centering. Variance shrinks
       with the round it was read in: (20 - 4*r).
     - The live quote on the table this round, when I am Taker, is the
       freshest version of the same signal.
   Weighting by inverse variance is the closed-form optimum for
   combining independent unbiased estimators of the same quantity, so
   this isn't a guess at a blend ratio -- it's the correct one given the
   stated noise model.

2. AUCTION BIDDING reuses the calibrated per-round power values measured
   in strategies/adaptive_bidder.py (explicitly free to reuse per
   RULEBOOK.md sec 12: baseline code and its constants are not evidence
   of copying) with the same 0.60 first-price shade. If a round's total
   bid would exceed remaining TE, spend is reallocated to the
   highest-value power(s) first rather than being zeroed out entirely.

3. THE FORCING TURN is played as a distinct decision, not folded into
   the ordinary accept/counter logic. Countering on the final turn is
   not "wait one more round" -- it deterministically fixes your side
   (short), your price (the midpoint of whatever range you just
   proposed, shifted by any TRICK_ROOM/STEALTH_ROCK you or your
   opponent hold), and a 2-tick fee, all at once. Since the *range you
   propose* sets that midpoint, and the rules only cap width from
   above, the range that maximises your forced price while staying
   legal is to hold the current ask fixed and take the widest still-
   legal width downward from it. Whether to take that guaranteed outcome
   over simply accepting is then a plain EV comparison against the
   current bid/ask -- with a small margin so the comparison isn't
   decided by estimation noise.
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
OPP_FLAT_THRESHOLD = 2.0      # how flat the opponent must look before TRANSFORM denial matters
SHIFT_POWERS = {"TRICK_ROOM": 3, "STEALTH_ROCK": 2}
FORCE_MARGIN = 0.3            # required EV edge before preferring a forced fill to accepting


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
        opponent's hand, added to our own exactly-known revealed sum."""
        self._update_foresight(obs)

        components = []  # (mean, variance) pairs, each an estimator of k_theirs

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

        if components:
            w_sum = sum(1.0 / v for _, v in components)
            k_theirs_hat = sum(m / v for m, v in components) / w_sum
        else:
            k_theirs_hat = 0.0

        return obs.k_mine + k_theirs_hat

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
        v = round(self._estimate_S(obs))
        cap = obs.final_cap
        lo = v - cap // 2
        return (lo, lo + cap)

    def respond(self, obs, quote: tuple, turn: int):
        # Only the very first response of the round reads a clean, honest
        # opening quote; later ranges are contaminated by both sides.
        if turn == 2:
            self._opp_anchor[obs.round] = (quote[0] + quote[1]) / 2.0

        bid_p, ask_p = quote
        S_hat = self._estimate_S(obs, quote)

        substitute = "SUBSTITUTE" in obs.powers_mine
        thresh = -1.0 if substitute else 0.0

        edge_buy = S_hat - ask_p
        edge_sell = bid_p - S_hat

        if turn == obs.n_turns:
            return self._final_turn(obs, bid_p, ask_p, S_hat, edge_buy, edge_sell,
                                     substitute, thresh)

        if edge_buy > thresh and edge_buy >= edge_sell:
            return "ACCEPT_BUY"
        if edge_sell > thresh:
            return "ACCEPT_SELL"

        w = max(0, (ask_p - bid_p) - self.config.MIN_REDUCTION)
        center = max(bid_p, min(round(S_hat), ask_p - w))
        return ("COUNTER", center, center + w)

    def _final_turn(self, obs, bid_p, ask_p, S_hat, edge_buy, edge_sell,
                     substitute, thresh):
        """Countering here is a forced fill: my range fixes the midpoint,
        I go short, I pay the forcing fee. Choose the legal range that
        maximises that price, then compare its EV to simply accepting."""
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
        if ev_force > best_accept + FORCE_MARGIN:
            return ("COUNTER", f_bid, f_ask)

        if ev_buy >= ev_sell:
            return "ACCEPT_BUY"
        return "ACCEPT_SELL"

    def use_transform(self, obs) -> bool:
        return abs(obs.k_mine) <= FLAT_THRESHOLD
