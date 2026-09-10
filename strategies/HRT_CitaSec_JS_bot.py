# Name: Heet Vijay Nisar
# College: Dwarkadas J. Sanghvi College of Engineering
# Roll Number: 60005240044

"""
v1: avg win 68%, rounds 1-2 negative, 3-5 positive, round 5 best.
v2-v4: progressively fixed the accept threshold and rulebook-mechanics
bugs (own-hand variance, quote freshness, counter-width floor). Tested
result: round 5 -- previously the BEST round -- turned into the WORST
round, and stayed the worst across every subsequent variant tested,
including a hand-modified descendant (bot28.py) built on top of v4.
Three unrelated variants sharing one ancestor all breaking the same
way, in the same round, means a bug in the shared code, not chance.

THE BUG (found by re-deriving _estimate_S from scratch): v2 introduced
a flat prior (mean 0, var N_PRIVATE) and always blended it in alongside
whatever real signal existed (FORESIGHT / anchor / live quote), meaning
to fix a discontinuity between "no signal" and "first signal arrives."
That discontinuity was never actually a problem -- a Bayesian estimate
is SUPPOSED to jump when real evidence first arrives, that's not a bug
to smooth over. And blending a redundant prior in on top of a real
signal IS a bug: every one of those signals' own variance formulas
(e.g. FORESIGHT's `N_PRIVATE - n_sampled`) already assumes the unseen
remainder is mean-zero -- that's where the "prior" already lives. Adding
a second, independent mean-zero prior on top double-counts that
assumption and drags the estimate back toward 0, on top of otherwise-
good evidence. This is invisible when signals are weak (early rounds)
and gets WORSE exactly as signal strength grows -- which is round 5,
the round with the most accumulated FORESIGHT/anchor information. A
round-5 read showing a strongly one-sided hand was getting ~15-20% of
its estimate silently pulled back toward zero. Fixed: the flat prior is
now used ONLY as the fallback when zero real signals exist yet -- not
blended in alongside them.

SUBSTITUTE handling was also a guess (a flat -1.0 acceptance threshold,
flat -2.0 EV clamps) rather than a derivation. Now exact: SUBSTITUTE
caps a round's contract loss at -2.0 (RULEBOOK.md sec 5, magnitude 2)
and refunds the shortfall, which means the TRUE expected value of a
trade is the raw edge plus the expected refund -- the expected value of
a normal-distributed variable's shortfall below -2, i.e. a textbook
"protective put" expectation. S is a sum of up to 40 independent +-1
coins, so a normal approximation for its posterior (mean S_hat, var
S_var, both already computed) is accurate. This replaces every
substitute-specific constant with one closed-form calculation.

The final-turn force-vs-accept margin (Z_FORCE * se, from v2-v4) is
also removed. The reasoning for it was that the comparison amplifies
S_hat's estimation error -- true, but under this game's actual scoring
(summed PnL, no risk adjustment, RULEBOOK.md sec 1/13), that's the same
mistake the SE-scaled accept threshold already turned out to be: with
an unbiased estimator, comparing two EV estimates and taking the larger
is already correct, with no margin, because that policy maximises
EXPECTED total PnL over repeated trials regardless of any single
estimate's variance. Kept for that decision: pure ev_force vs
best_accept, no margin.

CONFIRMED, STILL UNCHANGED FROM v3/v4: quote() centers at S_hat and
opens at exactly final_cap width -- provably optimal given RULEBOOK.md
sec 7.2's maker-obligation formula, which nets to zero EV under honest
centering for any width, while WIDTH_PREMIUM charges for width above
the floor unconditionally. Also unchanged: live-quote signal restricted
to turn==2 (sec 9: later ranges are contaminated), counter-width
respecting the round's floor (sec 6's exact max_width formula), and
own-hand residual variance (N_PRIVATE - len(my_revealed)) included in
S_var.

REVERTED (not carried over from the bot28.py variant you shared):
- Dynamic per-round auction shade (0.60->0.95). There IS a real
  argument for shading less aggressively as rounds progress -- TE spent
  now forecloses the OPTION to bid on a later round's still-unknown
  draw, and that option value shrinks to zero after round 5's auction,
  which argues for LESS shading late. But the specific schedule in
  bot28.py wasn't derived from that or anything else, and the one data
  point available (bot28.py's own result) is not a positive signal for
  it once its other bugs are accounted for. Reverted to the flat 0.60
  reused from adaptive_bidder.py (explicitly sanctioned, RULEBOOK.md
  sec 12) rather than ship an unvalidated schedule. Worth testing as
  its own isolated change in your backtester if you want to chase it.
- `getattr(self.config, 'LOSS_CAP', -1.5)` -- LOSS_CAP is not a real
  GameConfig attribute (see RULEBOOK.md sec 15's complete list), so
  this always silently fell back to -1.5. SUBSTITUTE's real cap is
  -2.0 (sec 5); now used exactly, via the put-expectation formula above
  rather than as a bare constant.
- TRANSFORM_THRESHOLD = 0.5 vs FLAT_THRESHOLD = 1 -- these are
  identical in practice. k_mine sums exactly REVEAL_PER_ROUND*round
  coins, always a multiple of 4, hence always even -- so `abs(k_mine)
  <= 0.5` and `abs(k_mine) <= 1` both only ever mean "k_mine is exactly
  0." The change was a no-op either way; kept at 1 to match the
  originally-tested value.

LEFT ALONE, STILL FLAGGED: TRICK_ROOM / STEALTH_ROCK's reused
POWER_VALUES are coupled to how often this bot actually forces, which
has changed again in this version (no more margin gating it). Not
re-derived here -- see the closing note in my reply.
"""

import math
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
SUBSTITUTE_CAP = -2.0          # RULEBOOK.md sec 5: SUBSTITUTE's exact loss cap magnitude
# Shift magnitudes per RULEBOOK.md sec 15 / sec 5. Not available via config
# (engine.shift_sources cannot be imported), must be reimplemented per rules.
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
        """Most recent opponent-whole-hand-sum read from an earlier
        round's opening quote (round < obs.round only)."""
        earlier = [r for r in self._opp_anchor if r < obs.round]
        if not earlier:
            return None, None
        r = max(earlier)
        return self._opp_anchor[r], r

    def _estimate_S(self, obs, live_quote=None):
        """Inverse-variance combination of every REAL signal about the
        opponent's whole 20-coin hand. The flat (mean 0, var N_PRIVATE)
        prior is used ONLY when no real signal exists yet -- blending it
        in alongside real signals double-counts the mean-zero assumption
        each signal's own variance formula already makes. See module
        docstring for why this was the main bug in v2-v4.

        Returns (S_hat, S_var). S_var includes both remaining
        uncertainty about the opponent's hand AND the structural,
        signal-proof uncertainty about our own still-unrevealed coins
        (RULEBOOK.md sec 2)."""
        n_private = self.config.N_PRIVATE
        per_round = self.config.REVEAL_PER_ROUND

        # Keep only the largest Foresight sample seen so far in this deal.
        self._update_foresight(obs)

        # Each component is (estimated opponent sum, uncertainty variance).
        components = []  # (mean, variance) of k_theirs, real signals only

        if self._foresight_n > 0:
            # Unseen opponent coins contribute the remaining variance.
            var = max(1.0, n_private - self._foresight_n)
            components.append((self._foresight_sum, var))

        anchor, anchor_r = self._best_anchor(obs)
        if anchor is not None:
            # Earlier opening quotes contain less information than later ones.
            var = max(1.0, n_private - per_round * anchor_r)
            components.append((anchor, var))

        # Only ever trust the live quote as a value signal on turn 2 --
        # the one turn guaranteed to be the Maker's untouched opening
        # (RULEBOOK.md sec 9: later ranges are contaminated by both sides).
        if live_quote is not None and not obs.is_maker:
            # Only the untouched opening quote is used as a clean opponent signal.
            live_mid = (live_quote[0] + live_quote[1]) / 2.0
            var = max(1.0, n_private - per_round * obs.round)
            components.append((live_mid, var))

        if components:
            # Inverse-variance weighting gives more influence to the more precise signal.
            w_sum = sum(1.0 / v for _, v in components)
            k_theirs_hat = sum(m / v for m, v in components) / w_sum
            opp_var = 1.0 / w_sum
        else:
            k_theirs_hat = 0.0
            opp_var = float(n_private)

        # Our unrevealed coins are also part of the uncertainty in total score S.
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
        # Expected value of the loss-cap refund under a normal approximation.
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

        # Convert a power value in PnL ticks into its TE-equivalent value.
        fair_te = {n: v / self.config.TE_SALVAGE for n, v in values.items()}
        # Shade a first-price bid so we do not pay full theoretical value.
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
        # Maker price is centered on the current estimate of total score S.
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
        # Estimate both fair value and uncertainty for this negotiation state.
        S_hat, S_var = self._estimate_S(obs, live_quote)
        se = S_var ** 0.5

        substitute = "SUBSTITUTE" in obs.powers_mine
        # Positive buy edge means the contract is cheaper than our estimate.
        edge_buy = S_hat - ask_p
        # Positive sell edge means the contract is more expensive than our estimate.
        edge_sell = bid_p - S_hat
        true_buy = self._true_ev(edge_buy, se, substitute)
        true_sell = self._true_ev(edge_sell, se, substitute)

        # The last turn is special: countering creates a forced fill.
        if turn == obs.n_turns:
            return self._final_turn(obs, bid_p, ask_p, S_hat, se, edge_buy, edge_sell,
                                     substitute)

        # With an unbiased S_hat, "true edge > 0" is already EV-maximal
        # under this game's pure-PnL-sum scoring -- no confidence bar
        # needed (see module docstring history).
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
        no margin, see module docstring for why."""
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

        # Choose the action with the highest expected PnL.
        if ev_force > best_accept:
            return ("COUNTER", f_bid, f_ask)

        if true_buy >= true_sell:
            return "ACCEPT_BUY"
        return "ACCEPT_SELL"

    def use_transform(self, obs) -> bool:
        # Transform is most valuable when our revealed hand is close to neutral.
        return abs(obs.k_mine) <= FLAT_THRESHOLD
