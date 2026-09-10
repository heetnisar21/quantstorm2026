# Name: Heet Nisar
# College: Dwarkadas J. Sanghvi College of Engineering
# Roll Number: 60005240044

"""
heet_bot.py — Divided Oracle entry (v4)
========================================

v1 baseline: avg win 68%, PnL negative in rounds 1-2, positive and
peaking in rounds 3-5.

v2 added a confidence-scaled ("edge must clear Z * standard-error")
threshold before accepting a trade, reasoning that early-round edges
were mostly noise. Tested result: Win 2 / Loss 6, every round red,
round 5 -- previously the best round -- now the worst. That result
is diagnostic, and it overturned the reasoning behind v2's own
change:

RULEBOOK.md sec 1/13 state the objective plainly -- summed PnL across
all matches, no risk adjustment, no variance penalty, no normalisation.
Under that objective, if S_hat is unbiased, "accept any edge > 0" is
already EV-maximal; there is no theoretical reason to also demand the
edge clear a confidence bar, because summed-over-many-trials PnL
doesn't care about the variance of any one trade, only its sign in
expectation. Round 5 is the bot's highest-confidence round (least
residual uncertainty), which is exactly where v1 was profitably taking
many small, real edges -- and v2's confidence bar filtered a chunk of
those out specifically there. Best round becoming worst is the
signature of that mechanism, not sampling noise.

v4 reverts the ordinary accept/counter threshold to v1's flat rule
(edge > 0, or > -1 under SUBSTITUTE's loss cap) and keeps three fixes
that ARE grounded in the rulebook rather than in an unvalidated risk
preference:

1. S_var (used only for the forcing-turn decision now, see #3) was
   missing its own biggest early-round term. RULEBOOK.md sec 2: each
   hand is N_PRIVATE=20 coins, and you only see REVEAL_PER_ROUND*round
   of YOUR OWN by a given round -- the rest of your own hand is exactly
   as unknown to you as your opponent's, until round 5. Fixed: S_var
   now always includes `N_PRIVATE - len(obs.my_revealed)` as a
   structural term alongside the opponent-side estimate.

2. The live opening quote was being read past its shelf life.
   RULEBOOK.md sec 9 is explicit: "open_bid/open_ask are the only
   clean read of a Maker's information. Later ranges are negotiated
   objects contaminated by both sides." v1/v2 fed *every* turn's
   on-the-table quote into the value estimate with the same trust as
   the honest turn-2 read. Turn 2 is always the Taker's first response
   to the Maker's turn-1 opening (turns alternate starting with the
   Taker), so it's the only turn guaranteed to be that untouched
   opening. Fixed: the live-quote signal is now only added at turn==2.

3. Counter-width had a floor bug. RULEBOOK.md sec 6's exact rule:
   `max_width = min(ask-bid, max(final_cap, (ask-bid) - MIN_REDUCTION))`.
   _final_turn already implemented this correctly; the ordinary
   counter logic in respond() used `max(0, (ask-bid) - MIN_REDUCTION)`
   instead of `max(final_cap, ...)`, so a round's counters could keep
   shrinking past the legal floor toward 0 width, giving away
   negotiating room the rules never required giving away. This bug
   was ALSO present in the original v1 file, so it isn't what caused
   the v2 regression -- but it's still a real deviation from sec 6,
   fixed here regardless.

The forcing-turn decision keeps an SE-scaled margin (Z_FORCE * se,
floored at v1's flat constant) rather than reverting fully to flat,
because forcing is different from an ordinary accept: it's a single
discrete, irreversible commitment, and the comparison it's based on
(forced-fill EV vs. best accept EV) amplifies estimation error rather
than cancelling it -- a S_hat error of size e shifts edge_buy by +e
but shifts ev_force by -e, so the *gap* between them moves by 2e, not
0. That is a genuine, derivable reason to demand extra clearance
specifically there, unlike the blanket accept-threshold change that
regressed round 5.

CONFIRMED, NOT CHANGED: quote()'s choice to always center at S_hat and
open at exactly final_cap width. RULEBOOK.md sec 7.2's maker-obligation
formula nets to precisely zero expected value under honest centering,
for *any* width -- the straddle-probability terms cancel algebraically
-- while WIDTH_PREMIUM charges 0.22 ticks per tick of width above the
floor, unconditionally. Opening wider than the floor has zero upside
and a certain, strictly negative cost.

LEFT ALONE, FLAGGED RATHER THAN RE-GUESSED: the calibrated POWER_VALUES
table, reused per RULEBOOK.md sec 12 (baseline constants are explicit
fair game). TRICK_ROOM and STEALTH_ROCK only pay off on a forced fill
(sec 5), so their fair value is coupled to how often *this* bot
actually forces. Since v4's forcing behavior differs from whatever
produced the reused table, those two entries are probably slightly
stale. Not re-derived here without a live backtester to validate a
replacement against -- see the closing note in my reply for how to
check this yourself.
"""

import random

# Calibrated tick value of each power, per round -- lifted from
# strategies/adaptive_bidder.py (a provided baseline; its constants are
# explicitly reusable, see RULEBOOK.md sec 12). See docstring note above
# on TRICK_ROOM / STEALTH_ROCK being coupled to forcing frequency.
POWER_VALUES = {
    "FORESIGHT":    {1: 0.76, 2: 1.16, 3: 1.48, 4: 1.97, 5: 2.02},
    "TRICK_ROOM":   {1: 1.14, 2: 0.00, 3: 0.00, 4: 0.60, 5: 0.52},
    "SUBSTITUTE":   {1: 1.46, 2: 1.15, 3: 0.95, 4: 0.57, 5: 0.29},
    "STEALTH_ROCK": {1: 1.51, 2: 0.75, 3: 0.75, 4: 0.75, 5: 0.00},
    "TRANSFORM":    {1: 1.58, 2: 1.24, 3: 1.31, 4: 0.00, 5: 0.00},
}

SHADE = 0.60                  # first-price shade, see adaptive_bidder.py
FLAT_THRESHOLD = 1             # |k_mine| this small -> hand is worth swapping away
# Shift magnitudes per RULEBOOK.md sec 15 / sec 5. Not available via config
# (engine.shift_sources cannot be imported), must be reimplemented per rules.
SHIFT_POWERS = {"TRICK_ROOM": 3, "STEALTH_ROCK": 2}

Z_FORCE = 0.50                  # SE multiplier for the forcing-turn margin only
FORCE_MARGIN_FLOOR = 0.3       # v1's constant, kept as a floor under the SE-based margin


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
        """Most recent opponent-whole-hand-sum read from an earlier
        round's opening quote (round < obs.round only)."""
        earlier = [r for r in self._opp_anchor if r < obs.round]
        if not earlier:
            return None, None
        r = max(earlier)
        return self._opp_anchor[r], r

    def _estimate_S(self, obs, live_quote=None):
        """Inverse-variance combination of every signal about the
        opponent's whole 20-coin hand (always including a flat
        shrinkage prior), added to our own exactly-known revealed sum.

        Returns (S_hat, S_var). S_var includes both our remaining
        uncertainty about the opponent's hand AND the structural,
        signal-proof uncertainty about our own still-unrevealed coins
        (RULEBOOK.md sec 2). It is used only by the forcing-turn
        decision -- see module docstring for why ordinary accept
        decisions use a flat threshold instead."""
        n_private = self.config.N_PRIVATE
        per_round = self.config.REVEAL_PER_ROUND

        self._update_foresight(obs)

        # Always-present shrinkage prior on the opponent's whole hand --
        # keeps thin early information from being taken at face value.
        components = [(0.0, float(n_private))]  # (mean, variance) of k_theirs

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

        w_sum = sum(1.0 / v for _, v in components)
        k_theirs_hat = sum(m / v for m, v in components) / w_sum
        opp_var = 1.0 / w_sum

        my_unseen_var = max(0, n_private - len(obs.my_revealed))
        S_var = opp_var + my_unseen_var

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
        # (SLOTS_PER_ROUND == 1 in the current rules, so `offered` is
        # always a single power -- this branch is defensive, not dead
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
        # optimal here -- see module docstring. Not a heuristic.
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
        # Flat threshold: with an unbiased S_hat, "edge > 0" is already
        # EV-maximal under this game's pure-PnL-sum scoring (no risk
        # adjustment) -- see module docstring for why a confidence bar
        # here was tested and reverted.
        thresh = -1.0 if substitute else 0.0

        edge_buy = S_hat - ask_p
        edge_sell = bid_p - S_hat

        if turn == obs.n_turns:
            return self._final_turn(obs, bid_p, ask_p, S_hat, se, edge_buy, edge_sell,
                                     substitute)

        if edge_buy > thresh and edge_buy >= edge_sell:
            return "ACCEPT_BUY"
        if edge_sell > thresh:
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
        then compare its EV to simply accepting -- with an SE-scaled
        margin, since this comparison amplifies S_hat error rather than
        cancelling it (see module docstring)."""
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